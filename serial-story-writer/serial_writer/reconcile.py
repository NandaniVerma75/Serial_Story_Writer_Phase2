"""Retroactive edits: the human rewrites approved episode K. What happens to K+1..?

1. The old text is archived and the new text becomes canon.
2. All derived state with episode_id >= K is deleted (event-sourced memory).
3. Episode K is re-extracted with the LLM from the human's text.
4. Later episodes' stored extractions are *replayed* (no LLM) on top, so memory
   is rebuilt in order; replay surfaces mechanical conflicts (e.g. a later
   episode resolves a thread that K no longer opens).
5. Invalidated arc / story-so-far summaries are regenerated.
6. Episodes K+1..K+window are consistency-checked against the rebuilt state as it
   stood before each of them, and the human chooses: keep, auto-revise the
   flagged ones, or regenerate everything after K.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .checker import consistency_check, dead_character_issues
from .context_builder import build_context
from .db import now_iso
from .extractor import extract_and_apply, rebuild_summaries
from .memory import apply_extraction, delete_derived_from, last_timeline, words_in
from .models import ScenePlan
from .story import Story
from .writer import revise

Progress = Callable[[str], None]


@dataclass
class EditReport:
    k: int
    later: list[int] = field(default_factory=list)
    checked: list[int] = field(default_factory=list)
    flagged: dict[int, list[str]] = field(default_factory=dict)
    replay_warnings: dict[int, list[str]] = field(default_factory=dict)


def _archive(story: Story, ep: int, reason: str) -> None:
    row = story.db.get_episode(ep)
    if row is not None and row["text"]:
        story.db.execute("INSERT INTO episode_archive(ep, title, text, reason, archived_at) VALUES(?,?,?,?,?)",
                         (ep, row["title"], row["text"], reason, now_iso()))


def replay_from(story: Story, start: int) -> dict[int, list[str]]:
    """Re-apply stored extractions of approved episodes >= start, in order (no LLM calls)."""
    warnings: dict[int, list[str]] = {}
    for j in story.db.approved_eps():
        if j < start:
            continue
        data = story.db.scalar("SELECT data FROM extractions WHERE episode_id=?", (j,))
        if data:
            w = apply_extraction(story.db, j, json.loads(data))
            if w:
                warnings[j] = w
    return warnings


def apply_edit(story: Story, k: int, new_text: str, window: int | None = None,
               progress: Progress = lambda m: None) -> EditReport:
    db = story.db
    row = db.get_episode(k)
    if row is None:
        raise ValueError(f"episode {k} does not exist")
    report = EditReport(k)
    if row["status"] != "approved":
        # Draft not yet canon: just replace the text; it goes through review as usual.
        db.upsert_episode(k, text=new_text, word_count=words_in(new_text), human_edited=1)
        return report

    _archive(story, k, "replaced by human retroactive edit")
    db.upsert_episode(k, text=new_text, word_count=words_in(new_text), human_edited=1)
    story.tracer.decision(k, "reconcile", "human retroactive edit; rebuilding memory from this episode")

    progress(f"Deleting derived state for episodes >= {k}")
    delete_derived_from(db, k)
    progress(f"Re-extracting episode {k} from the edited text")
    extract_and_apply(story, k, new_text)
    story.write_episode_md(k)

    report.later = [j for j in db.approved_eps() if j > k]
    progress(f"Replaying stored extractions for {len(report.later)} later episodes")
    report.replay_warnings = replay_from(story, k + 1)
    progress("Regenerating invalidated summaries")
    rebuild_summaries(story)

    window = window if window is not None else story.settings.reconcile_window
    report.checked = report.later[:window]
    for j in report.checked:
        progress(f"Checking episode {j} against the rebuilt canon")
        issues = check_downstream(story, j) + report.replay_warnings.get(j, [])
        if issues:
            report.flagged[j] = issues
    story.tracer.decision(k, "reconcile",
                          f"checked {len(report.checked)} later eps; flagged {sorted(report.flagged)}")
    return report


def check_downstream(story: Story, j: int) -> list[str]:
    db = story.db
    row = db.get_episode(j)
    ctx = build_context(story, j)
    issues = []
    if row["scene_plan"]:
        plan = ScenePlan.model_validate_json(row["scene_plan"])
        issues += [i.message for i in dead_character_issues(row["text"], plan, ctx.characters)]
    day = db.one("SELECT day FROM timeline WHERE episode_id=?", (j,))
    last = last_timeline(db, j)
    if day and last and day["day"] < last["day"]:
        issues.append(f"timeline: Day {day['day']} comes after Day {last['day']} (ep {last['episode_id']})")
    report = consistency_check(story, ctx, row["text"])
    issues += [f"{c.claim} — contradicts: {c.conflicts_with}" for c in report.contradictions if c.severity != "low"]
    return issues


def resolve(story: Story, report: EditReport, option: str, progress: Progress = lambda m: None) -> list[int]:
    """option: keep | revise | regenerate. Returns the episodes that changed."""
    db = story.db
    if option == "keep" or (option == "revise" and not report.flagged):
        story.tracer.decision(report.k, "reconcile", "human chose keep")
        return []
    if option == "revise":
        changed = []
        for j in sorted(report.flagged):
            progress(f"Revising episode {j} to fit the new canon")
            ctx = build_context(story, j)
            row = db.get_episode(j)
            plan = ScenePlan.model_validate_json(row["scene_plan"])
            title, text = revise(story, ctx, plan, row["text"],
                                 [f"[continuity after retro edit of ep {report.k}] {i}" for i in report.flagged[j]])
            _archive(story, j, f"auto-revised after retro edit of ep {report.k}")
            db.upsert_episode(j, title=title, text=text, word_count=words_in(text))
            delete_derived_from(db, j)
            extract_and_apply(story, j, text)
            replay_from(story, j + 1)
            story.write_episode_md(j)
            changed.append(j)
        rebuild_summaries(story)
        story.tracer.decision(report.k, "reconcile", f"auto-revised {changed}")
        return changed
    if option == "regenerate":
        later = report.later
        with db.transaction():
            for j in later:
                _archive(story, j, f"discarded: regenerate after retro edit of ep {report.k}")
            db.execute("DELETE FROM episodes WHERE ep > ?", (report.k,))
            db.execute("DELETE FROM extractions WHERE episode_id > ?", (report.k,))
            db.execute("DELETE FROM directive_impact WHERE episode > ?", (report.k,))
        delete_derived_from(db, report.k + 1)
        for j in later:
            story.episode_path(j).unlink(missing_ok=True)
        story.tracer.decision(report.k, "reconcile", f"discarded eps {later[:1]}..{later[-1:]} for regeneration")
        return later
    raise ValueError(f"unknown option {option!r}")
