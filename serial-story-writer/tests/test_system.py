from __future__ import annotations

import pytest

from serial_writer.checker import hard_checks
from serial_writer.context_builder import build_context
from serial_writer.directives import add_directive, handle_feedback
from serial_writer.memory import characters_view, facts_view, threads_view
from serial_writer.models import DirectiveClassification, ScenePlan
from serial_writer.pipeline import AutoReviewer, Pipeline, ReviewDecision
from serial_writer.planner import PlanError, export_plan_yaml, import_plan_yaml, validate_plan
from serial_writer.reconcile import apply_edit, resolve
from tests.conftest import reopen, write_n
from tests.fakes import FakeProvider


# --- planning --------------------------------------------------------------------
def test_plan_has_200_unique_beats_and_round_trips_yaml(planned_story):
    s = planned_story
    assert s.db.scalar("SELECT COUNT(DISTINCT ep) FROM beats") == 200
    assert validate_plan(s) == []
    text = export_plan_yaml(s).read_text().replace("Ravi and Meera chase clue 7.", "Ravi loses the ledger.")
    import_plan_yaml(s, text)
    assert s.db.beat(7) == "Ravi loses the ledger."
    assert s.db.scalar("SELECT COUNT(*) FROM beat_history WHERE ep=7") == 1
    with pytest.raises(PlanError):
        import_plan_yaml(s, text.replace("    7: Ravi loses the ledger.\n", ""))


# --- context builder -------------------------------------------------------------
def test_context_respects_budget_and_keeps_required(planned_story):
    s = planned_story
    write_n(s, 6)
    add_directive(s, "no swearing", 3, DirectiveClassification(
        kind="style", scope="global", needs_replan=False, normalized_text="Characters never swear."))
    full = build_context(s, 7, budget=100_000)
    assert full.log["dropped"] == []
    small_budget = full.section("bible").tokens() + 600
    ctx = build_context(s, 7, budget=small_budget)
    assert ctx.log["dropped"], "something must be trimmed"
    assert ctx.total_tokens() <= small_budget or all(
        not sec.items for sec in ctx.sections if not sec.required)
    assert ctx.section("bible").items
    assert any("Characters never swear." in i for i in ctx.section("directives").items)
    assert ctx.section("beats").items[0].startswith("THIS EPISODE (7) BEAT")
    # facts are trimmed before the previous episode
    assert len(ctx.section("facts").items) <= len(full.section("facts").items)


# --- hard checks -----------------------------------------------------------------
def _plan(**kw) -> ScenePlan:
    base = dict(title="t", in_story_day=5, characters=["Ravi Kumar"], scenes=["s"], goal="g", conflict="c",
                hook_type="threat", hook_idea="h")
    base.update(kw)
    return ScenePlan(**base)


def test_hard_checks_catch_dead_character_wordcount_timeline(planned_story):
    s = planned_story
    write_n(s, 3)  # timeline days 1..3
    s.db.execute("INSERT INTO char_events(episode_id,name,kind,value,detail) VALUES(2,'Meera Shah','status','dead','shot')")
    chars = characters_view(s.db, 4)
    ok_text = "word " * 500

    issues = hard_checks(s, 4, ok_text, _plan(), chars)
    assert not [i for i in issues if i.severity == "error"]

    kinds = lambda iss: {i.kind for i in iss if i.severity == "error"}  # noqa: E731
    assert "length" in kinds(hard_checks(s, 4, "too short", _plan(), chars))
    assert "length" in kinds(hard_checks(s, 4, "word " * 900, _plan(), chars))
    assert "continuity" in kinds(hard_checks(s, 4, ok_text, _plan(characters=["Ravi Kumar", "Meera"]), chars))
    assert "continuity" in kinds(hard_checks(s, 4, ok_text + " Meera said hello.", _plan(), chars))
    assert "continuity" not in kinds(hard_checks(s, 4, ok_text + " He missed Meera.", _plan(), chars))
    assert "timeline" in kinds(hard_checks(s, 4, ok_text, _plan(in_story_day=1), chars))
    assert "timeline" not in kinds(hard_checks(s, 4, ok_text, _plan(in_story_day=1, is_flashback=True), chars))
    assert "characters" in kinds(hard_checks(s, 4, ok_text, _plan(characters=["Ravi", "Meeraa Shaw"]), chars))
    unknown = hard_checks(s, 4, ok_text, _plan(characters=["Ravi", "Zoya"]), chars)
    assert any(i.kind == "characters" and i.severity == "warn" for i in unknown)
    declared = hard_checks(s, 4, ok_text, _plan(characters=["Ravi", "Zoya"], new_characters=["Zoya"]), chars)
    assert not any(i.kind == "characters" for i in declared)


# --- directives --------------------------------------------------------------------
def test_directive_from_ep5_is_in_context_at_ep10(planned_story):
    s = planned_story
    write_n(s, 4)
    did = add_directive(s, "slow down the romance", 5, DirectiveClassification(
        kind="pacing", scope="global", needs_replan=False, normalized_text="Slow the Ravi–Meera romance."))
    temp = add_directive(s, "rain every day", 5, DirectiveClassification(
        kind="style", scope="until_episode", until_episode=7, needs_replan=False, normalized_text="It rains."))
    write_n(s, 5)  # eps 5..9
    ctx = build_context(s, 10)
    assert did in ctx.directive_ids and temp not in ctx.directive_ids
    assert any("Slow the Ravi–Meera romance." in i for i in ctx.section("directives").items)
    # propagation is recorded per episode with adherence
    eps = [r["episode"] for r in s.db.query("SELECT episode FROM directive_impact WHERE directive_id=?", (did,))]
    assert eps == [5, 6, 7, 8, 9]
    assert [r["episode"] for r in s.db.query("SELECT episode FROM directive_impact WHERE directive_id=?", (temp,))] == [5, 6, 7]


def test_kill_directive_replans_and_schedules_fate(planned_story):
    s = planned_story
    write_n(s, 3)
    out = handle_feedback(s, "kill off Meera", 4, approve=lambda changes, why: True)
    assert out.applied and out.changes[0].ep == 5
    assert s.db.beat(5) == "Meera is killed in the tram depot."
    fate = s.db.one("SELECT * FROM fates")
    assert fate["name"] == "Meera Shah" and fate["status"] == "dead" and fate["by_ep"] == 6
    ctx = build_context(s, 6)
    assert any("MUST HAPPEN IN THIS EPISODE" in i for i in ctx.section("beats").items)


# --- resume --------------------------------------------------------------------------
def test_resume_after_crash_mid_episode(planned_story, tmp_path):
    s = planned_story
    write_n(s, 2)
    crashing = FakeProvider(fail_on={"check:consistency"})
    s = reopen(s, crashing)
    with pytest.raises(Exception):
        Pipeline(s, AutoReviewer(on_fail="approve")).run_episode(3)
    assert s.db.get_episode(3)["status"] == "drafted"  # draft saved before the crash

    fresh = FakeProvider()
    s = reopen(s, fresh)
    pipe = Pipeline(s, AutoReviewer(on_fail="approve"))
    assert pipe.next_episode() == 3
    assert pipe.run_episode(3) == "approved"
    assert "draft" not in fresh.calls and "refine_beat" not in fresh.calls  # resumed, not redone


def test_resume_mid_review(planned_story):
    s = planned_story

    class Pauser:
        def review(self, story, ep, checks):
            return ReviewDecision("pause")

    assert Pipeline(s, Pauser()).run_episode(1) == "paused"
    assert s.db.get_episode(1)["status"] == "review"
    fresh = FakeProvider()
    s = reopen(s, fresh)
    assert Pipeline(s, AutoReviewer(on_fail="approve")).run_episode(1) == "approved"
    assert fresh.calls == ["extract"]  # straight to commit


# --- retroactive edit -----------------------------------------------------------------
def test_retro_edit_removes_and_replays_derived_state(planned_story):
    s = planned_story
    write_n(s, 4)
    subjects = lambda: {f["subject"] for f in facts_view(s.db, 99)}  # noqa: E731
    assert subjects() == {"clue1", "clue2", "clue3", "clue4"}

    report = apply_edit(s, 2, "EDITED " + s.db.get_episode(2)["text"])
    assert subjects() == {"clue1", "clue2b", "clue3", "clue4"}  # ep 2 re-extracted; 3,4 replayed
    assert s.db.get_episode(2)["human_edited"] == 1
    assert report.later == [3, 4] and report.checked == [3, 4]
    assert {t.description for t in threads_view(s.db, 99)} == {
        "Who hid clue1?", "Who hid clue2b?", "Who hid clue3?", "Who hid clue4?"}
    assert s.db.scalar("SELECT COUNT(*) FROM episode_archive WHERE ep=2") == 1

    resolve(s, report, "regenerate")
    assert s.db.approved_eps() == [1, 2]
    assert subjects() == {"clue1", "clue2b"}
    write_n(s, 1)
    assert s.db.approved_eps() == [1, 2, 3]


def test_reject_regenerates_with_reason(planned_story):
    s = planned_story

    class RejectOnce:
        def __init__(self):
            self.n = 0

        def review(self, story, ep, checks):
            self.n += 1
            return ReviewDecision("reject", "too much exposition") if self.n == 1 else ReviewDecision("approve")

    assert Pipeline(s, RejectOnce()).run_episode(1) == "approved"
    assert s.db.scalar("SELECT COUNT(*) FROM episode_archive WHERE ep=1") == 1


def test_cost_cap_and_bounded_revisions(tmp_path):
    from tests.conftest import make_story
    from serial_writer.planner import generate_plan

    # Judge always fails -> must stop after max_revisions and hand to the human.
    bad = lambda u: '{"hook_strength":1,"momentum":1,"voice":1,"beat_adherence":1,"directive_adherence":1,"summary":"x"}'  # noqa: E731
    provider = FakeProvider(overrides={"check:rubric": bad})
    s = make_story(tmp_path, provider)
    generate_plan(s, premise="p")
    s.db.set_meta("plan_status", "approved")
    seen = {}

    class Capture:
        def review(self, story, ep, checks):
            seen["checks"] = checks
            return ReviewDecision("pause")

    Pipeline(s, Capture()).run_episode(1)
    assert provider.calls.count("revise") == 2
    assert seen["checks"].passed is False
