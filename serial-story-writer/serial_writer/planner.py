"""Story bible + hierarchical 200-episode plan, plan validation, YAML round-trip, re-planning.

Planning is hierarchical so no single call has to hold 200 episodes:
bible (1 call) -> 5 acts (1 call) -> 10 arcs (1 call) -> 20 beats per arc (10 calls).
Each beats call sees the acts, a one-line view of every arc, and the tail of the
previous arc's beats, which is enough for local continuity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import yaml

from .db import now_iso
from .memory import characters_view, seed_characters
from .models import ActPlan, ArcPlan, Beat, BeatList, Replan, StoryBible
from .story import Story, load_prompt

Progress = Callable[[str], None]


class PlanError(ValueError):
    pass


def _fb(feedback: list[str]) -> str:
    if not feedback:
        return ""
    return "SHOWRUNNER FEEDBACK ON THE PLAN (must be honoured):\n" + "\n".join(f"- {f}" for f in feedback)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_bible(story: Story, premise: str, feedback: list[str] | None = None) -> StoryBible:
    fb = _fb(feedback or [])
    previous = story.db.get_meta("bible")
    if previous and fb:
        fb += ("\n\nCURRENT BIBLE (revise it minimally to honour the feedback; keep names stable):\n"
               + json.dumps(previous, ensure_ascii=False))
    bible = story.llm.complete_json(
        StoryBible, step="plan:bible", episode=0, tier="writer",
        system="You are a showrunner and story architect for long-form serial fiction.",
        user=load_prompt("bible", premise=premise, feedback=fb),
    )
    story.db.set_meta("premise", premise)
    story.db.set_meta("bible", bible.model_dump())
    seed_characters(story.db, bible.characters)
    return bible


def plan_acts(story: Story, feedback: list[str]) -> ActPlan:
    s = story.settings
    previous = ""
    old = story.db.query("SELECT * FROM acts ORDER BY act_no")
    if old and feedback:
        previous = "PREVIOUS VERSION OF THE ACTS (revise it according to the feedback):\n" + "\n".join(
            f"Act {a['act_no']}: {a['title']} — {a['summary']} Turning point: {a['turning_point']}" for a in old)

    def check(p: ActPlan) -> None:
        if len(p.acts) != s.n_acts:
            raise ValueError(f"expected exactly {s.n_acts} acts, got {len(p.acts)}")

    names = ", ".join(c.name for c in story.bible.characters)
    plan = story.llm.complete_json(
        ActPlan, step="plan:acts", episode=0, tier="writer", system=story.writer_system(),
        user=load_prompt("acts", n_acts=s.n_acts, feedback=_fb(feedback), previous=previous, characters=names),
        validate=check,
    )
    for i, a in enumerate(plan.acts, 1):
        a.act_no = i
    return plan


def plan_arcs(story: Story, acts: ActPlan, feedback: list[str]) -> ArcPlan:
    s = story.settings

    def check(p: ArcPlan) -> None:
        if len(p.arcs) != s.n_arcs:
            raise ValueError(f"expected exactly {s.n_arcs} arcs, got {len(p.arcs)}")

    plan = story.llm.complete_json(
        ArcPlan, step="plan:arcs", episode=0, tier="writer", system=story.writer_system(),
        user=load_prompt("arcs", n_arcs=s.n_arcs, acts=_acts_text(acts), feedback=_fb(feedback)),
        validate=check,
    )
    per_act = s.n_arcs // s.n_acts
    for i, arc in enumerate(plan.arcs, 1):
        arc.arc_no = i
        arc.act_no = (i - 1) // per_act + 1
        for t in arc.threads_opened:
            t.payoff_arc = max(i, min(s.n_arcs, t.payoff_arc))
    return plan


def arc_range(story: Story, arc_no: int) -> tuple[int, int]:
    per = story.settings.total_episodes // story.settings.n_arcs
    return (arc_no - 1) * per + 1, arc_no * per


def plan_beats(story: Story, acts: ActPlan, arcs: ArcPlan, arc_no: int, prev_beats: list[Beat],
               feedback: list[str]) -> list[Beat]:
    arc = arcs.arcs[arc_no - 1]
    start, end = arc_range(story, arc_no)
    n = end - start + 1
    payoff = [t.description for a in arcs.arcs for t in a.threads_opened if t.payoff_arc == arc_no]

    def check(bl: BeatList) -> None:
        if len(bl.beats) != n:
            raise ValueError(f"expected exactly {n} beats for episodes {start}-{end}, got {len(bl.beats)}")

    bl = story.llm.complete_json(
        BeatList, step=f"plan:beats:arc{arc_no}", episode=0, tier="writer", system=story.writer_system(),
        user=load_prompt(
            "beats", arc_no=arc_no, arc_title=arc.title, start_ep=start, end_ep=end,
            acts=_acts_text(acts), arcs=_arcs_text(story, arcs), arc_goal=arc.goal,
            threads_open="; ".join(t.description + f" (pays off arc {t.payoff_arc})" for t in arc.threads_opened) or "none",
            threads_payoff="; ".join(payoff) or "none",
            prev_beats="\n".join(f"{b.ep}. {b.beat}" for b in prev_beats[-5:]) or "(this is the first arc)",
            feedback=_fb(feedback),
        ),
        validate=check,
    )
    # Numbering is enforced by code: the model is good at content, not at counting.
    return [Beat(ep=start + i, beat=b.beat.strip()) for i, b in enumerate(bl.beats)]


def generate_plan(story: Story, premise: str | None = None, feedback: list[str] | None = None,
                  progress: Progress = lambda m: None) -> list[str]:
    """Create (or fully re-create, with feedback) the plan. Returns validation warnings."""
    feedback = feedback or []
    premise = premise or story.db.get_meta("premise")
    if story.db.get_meta("bible") is None or feedback:
        progress("Building story bible…" if not feedback else "Revising story bible with feedback…")
        build_bible(story, premise, feedback)
    progress("Planning 5 acts…")
    acts = plan_acts(story, feedback)
    progress("Planning 10 arcs…")
    arcs = plan_arcs(story, acts, feedback)
    beats: list[Beat] = []
    for arc in arcs.arcs:
        progress(f"Planning beats for arc {arc.arc_no}/{len(arcs.arcs)}: {arc.title}")
        beats.extend(plan_beats(story, acts, arcs, arc.arc_no, beats, feedback))
    save_plan(story, acts, arcs, beats)
    story.db.set_meta("plan_status", "pending")
    warnings = validate_plan(story)
    export_plan_yaml(story)
    return warnings


def _acts_text(acts: ActPlan) -> str:
    out = []
    for a in acts.acts:
        out.append(f"Act {a.act_no}: {a.title}\n  {a.summary}\n  Turning point: {a.turning_point}\n  "
                   + "\n  ".join(a.character_arcs))
    return "\n".join(out)


def _arcs_text(story: Story, arcs: ArcPlan) -> str:
    out = []
    for a in arcs.arcs:
        s, e = arc_range(story, a.arc_no)
        out.append(f"Arc {a.arc_no} (act {a.act_no}, eps {s}-{e}): {a.title} — {a.goal}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Persistence + validation
# ---------------------------------------------------------------------------
def save_plan(story: Story, acts: ActPlan, arcs: ArcPlan, beats: list[Beat]) -> None:
    db = story.db
    with db.transaction():
        db.execute("DELETE FROM acts")
        db.execute("DELETE FROM arcs")
        db.execute("DELETE FROM beats")
        db.execute("DELETE FROM plan_threads")
        for a in acts.acts:
            db.execute("INSERT INTO acts VALUES(?,?,?,?,?)",
                       (a.act_no, a.title, a.summary, a.turning_point, "\n".join(a.character_arcs)))
        for arc in arcs.arcs:
            s, e = arc_range(story, arc.arc_no)
            db.execute("INSERT INTO arcs VALUES(?,?,?,?,?,?)", (arc.arc_no, arc.act_no, arc.title, arc.goal, s, e))
            for t in arc.threads_opened:
                db.execute("INSERT INTO plan_threads(description, opened_arc, payoff_arc) VALUES(?,?,?)",
                           (t.description, arc.arc_no, t.payoff_arc))
        for b in beats:
            arc = (b.ep - 1) // (story.settings.total_episodes // story.settings.n_arcs) + 1
            db.execute("INSERT INTO beats(ep, arc_no, beat) VALUES(?,?,?)", (b.ep, arc, b.beat))


def validate_plan(story: Story) -> list[str]:
    """Hard rules raise PlanError; soft problems come back as warnings for the human."""
    db, s = story.db, story.settings
    eps = [r[0] for r in db.query("SELECT ep FROM beats ORDER BY ep")]
    if eps != list(range(1, s.total_episodes + 1)):
        raise PlanError(f"plan must have exactly {s.total_episodes} uniquely numbered beats (got {len(eps)})")
    empty = [r[0] for r in db.query("SELECT ep FROM beats WHERE trim(COALESCE(beat,''))=''")]
    if empty:
        raise PlanError(f"empty beats: {empty[:10]}")
    for arc in db.query("SELECT * FROM arcs"):
        n = db.scalar("SELECT COUNT(*) FROM beats WHERE ep BETWEEN ? AND ?", (arc["start_ep"], arc["end_ep"]))
        if n != arc["end_ep"] - arc["start_ep"] + 1:
            raise PlanError(f"arc {arc['arc_no']} has {n} beats")
    warnings = []
    arcs_text = " ".join((r["character_arcs"] or "") for r in db.query("SELECT character_arcs FROM acts")).lower()
    for c in story.bible.characters:
        if c.name.lower() not in arcs_text and c.name.split()[0].lower() not in arcs_text:
            warnings.append(f"main character {c.name} has no arc in any act")
    for t in db.query("SELECT * FROM plan_threads"):
        if t["payoff_arc"] is None or not (t["opened_arc"] <= t["payoff_arc"] <= s.n_arcs):
            warnings.append(f"thread '{t['description']}' has no valid planned payoff")
    if not db.scalar("SELECT COUNT(*) FROM plan_threads"):
        warnings.append("plan opens no long-range threads")
    return warnings


def export_plan_yaml(story: Story, path: Path | None = None) -> Path:
    db = story.db
    path = path or story.dir / "plan.yaml"
    data = {
        "title": story.bible.title,
        "premise": db.get_meta("premise"),
        "plan_status": db.get_meta("plan_status"),
        "bible": story.bible.model_dump(),
        "acts": [
            {"act_no": a["act_no"], "title": a["title"], "summary": a["summary"],
             "turning_point": a["turning_point"],
             "character_arcs": [x for x in (a["character_arcs"] or "").split("\n") if x]}
            for a in db.query("SELECT * FROM acts ORDER BY act_no")
        ],
        "arcs": [
            {"arc_no": r["arc_no"], "act_no": r["act_no"], "title": r["title"], "goal": r["goal"],
             "episodes": f"{r['start_ep']}-{r['end_ep']}",
             "threads": [{"description": t["description"], "payoff_arc": t["payoff_arc"]}
                         for t in db.query("SELECT * FROM plan_threads WHERE opened_arc=? ORDER BY id", (r["arc_no"],))],
             "beats": {b["ep"]: b["beat"] for b in db.query(
                 "SELECT ep, beat FROM beats WHERE ep BETWEEN ? AND ? ORDER BY ep", (r["start_ep"], r["end_ep"]))}}
            for r in db.query("SELECT * FROM arcs ORDER BY arc_no")
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return path


def import_plan_yaml(story: Story, text: str) -> list[str]:
    """Load a human-edited plan. Validates before committing; raises PlanError on bad input."""
    try:
        data = yaml.safe_load(text)
        bible = StoryBible.model_validate(data["bible"])
        acts = ActPlan.model_validate({"acts": data["acts"]})
        arcs = ArcPlan.model_validate({"arcs": [
            {**{k: v for k, v in a.items() if k not in ("beats", "threads", "episodes")},
             "threads_opened": a.get("threads", [])} for a in data["arcs"]]})
        beats = [Beat(ep=int(ep), beat=str(b)) for a in data["arcs"] for ep, b in (a.get("beats") or {}).items()]
    except Exception as exc:  # yaml errors, KeyErrors, pydantic errors
        raise PlanError(f"could not parse edited plan: {exc}") from exc
    eps = sorted(b.ep for b in beats)
    if eps != list(range(1, story.settings.total_episodes + 1)):
        raise PlanError(f"edited plan must keep exactly {story.settings.total_episodes} beats numbered 1..N")
    old_beats = {r["ep"]: r["beat"] for r in story.db.query("SELECT ep, beat FROM beats")}
    with story.db.transaction():
        story.db.set_meta("bible", bible.model_dump())
        seed_characters(story.db, bible.characters)
        save_plan(story, acts, arcs, beats)
        for b in beats:
            if old_beats.get(b.ep) not in (None, b.beat):
                story.db.execute(
                    "INSERT INTO beat_history(ep, old_beat, new_beat, directive_id, reason, created_at) "
                    "VALUES(?,?,?,NULL,'human plan edit',?)", (b.ep, old_beats[b.ep], b.beat, now_iso()))
    warnings = validate_plan(story)
    export_plan_yaml(story)
    return warnings


# ---------------------------------------------------------------------------
# Re-planning for directives
# ---------------------------------------------------------------------------
def replan_beats(story: Story, directive: dict, start: int, end: int, fates: list[dict]) -> Replan:
    db = story.db
    beats = db.query("SELECT ep, beat FROM beats WHERE ep BETWEEN ? AND ? ORDER BY ep", (start, end))
    arcs = db.query("SELECT * FROM arcs WHERE end_ep >= ? AND start_ep <= ? ORDER BY arc_no", (start, end))
    chars = characters_view(db, start)
    state = [db.get_meta("story_so_far") or "",
             "Characters: " + "; ".join(f"{c.name} ({c.role}, {c.status})" for c in chars.values())]
    last = db.query("SELECT ep, summary FROM episodes WHERE status='approved' AND ep < ? ORDER BY ep DESC LIMIT 3", (start,))
    state += [f"Ep {r['ep']}: {r['summary']}" for r in reversed(last)]
    fate_text = "\n".join(f"SCHEDULED: {f['name']} must become {f['status']} by episode {f['by_ep']}." for f in fates)

    def check(r: Replan) -> None:
        bad = [b.ep for b in r.changed_beats if not (start <= b.ep <= end)]
        if bad:
            raise ValueError(f"changed beats outside {start}-{end}: {bad}")

    return story.llm.complete_json(
        Replan, step="directive:replan", episode=start, tier="writer", system=story.writer_system(),
        user=load_prompt(
            "replan", directive_id=directive["id"], kind=directive["kind"], directive=directive["text"],
            fates=fate_text, state="\n".join(x for x in state if x), start_ep=start, end_ep=end,
            beats="\n".join(f"{b['ep']}. {b['beat']}" for b in beats),
            arcs="\n".join(f"Arc {a['arc_no']} (eps {a['start_ep']}-{a['end_ep']}): {a['title']} — {a['goal']}" for a in arcs),
        ),
        validate=check,
    )


def apply_beat_changes(story: Story, changes: list[Beat], directive_id: int | None, reason: str) -> None:
    db = story.db
    with db.transaction():
        for b in changes:
            old = db.beat(b.ep)
            if old == b.beat:
                continue
            db.execute("UPDATE beats SET beat=?, revision=revision+1 WHERE ep=?", (b.beat, b.ep))
            db.execute(
                "INSERT INTO beat_history(ep, old_beat, new_beat, directive_id, reason, created_at) VALUES(?,?,?,?,?,?)",
                (b.ep, old, b.beat, directive_id, reason, now_iso()))
    export_plan_yaml(story)
