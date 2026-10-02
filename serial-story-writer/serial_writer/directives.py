"""Human feedback -> persistent directives -> re-planned beats -> measurable impact.

Flow for one piece of feedback:
  1. classify (kind + scope) with the cheap model
  2. store as a persistent directive (included in every in-scope episode's context)
  3. schedule character fates (e.g. "kill off Meera by ep 11")
  4. if needed, re-plan the affected future beats with the writer model and show
     the diff for approval
Propagation is then *measured*: each episode records which directives were in
its context and the judge's per-directive adherence score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .db import now_iso
from .memory import characters_view, match_name
from .models import Beat, DirectiveClassification
from .planner import apply_beat_changes, replan_beats
from .story import Story, load_prompt

STATUS_ALIASES = {"killed": "dead", "deceased": "dead", "died": "dead", "death": "dead", "gone": "departed"}


@dataclass
class BeatChange:
    ep: int
    old: str
    new: str


@dataclass
class FeedbackOutcome:
    directive_id: int
    classification: DirectiveClassification
    changes: list[BeatChange] = field(default_factory=list)
    applied: bool = False
    rationale: str = ""


def classify(story: Story, text: str, at_ep: int) -> DirectiveClassification:
    db = story.db
    chars = characters_view(db, at_ep)
    arcs = db.query("SELECT arc_no, title, start_ep, end_ep FROM arcs ORDER BY arc_no")
    return story.llm.complete_json(
        DirectiveClassification, step="directive:classify", episode=at_ep, tier="cheap",
        system=story.checker_system(),
        user=load_prompt(
            "classify_directive", text=text, ep=at_ep, total=story.settings.total_episodes,
            characters=", ".join(f"{c.name} ({c.status})" for c in chars.values()),
            arcs="; ".join(f"{a['arc_no']}: {a['title']} (eps {a['start_ep']}-{a['end_ep']})" for a in arcs),
        ),
    )


def add_directive(story: Story, raw_text: str, at_ep: int, cls: DirectiveClassification) -> int:
    db = story.db
    known = list(characters_view(db, at_ep + 1))
    with db.transaction():
        cur = db.execute(
            "INSERT INTO directives(text, kind, scope, scope_target, until_ep, created_at_ep, active, created_at, "
            "classification) VALUES(?,?,?,?,?,?,1,?,?)",
            (cls.normalized_text or raw_text, cls.kind, cls.scope, cls.scope_target, cls.until_episode, at_ep,
             now_iso(), json.dumps({"raw": raw_text, **cls.model_dump()})))
        did = cur.lastrowid
        for f in cls.fates:
            name = match_name(f.name, known) or f.name
            status = STATUS_ALIASES.get(f.status, f.status)
            by_ep = max(at_ep, f.by_episode or at_ep + 3)
            db.execute("INSERT INTO fates(name, status, by_ep, directive_id) VALUES(?,?,?,?)",
                       (name, status, by_ep, did))
    story.tracer.decision(at_ep, "directive:add", f"D{did} {cls.kind}/{cls.scope}: {cls.normalized_text[:120]}")
    return did


def propose_replan(story: Story, directive_id: int, at_ep: int) -> tuple[list[BeatChange], str]:
    db, s = story.db, story.settings
    d = dict(db.one("SELECT * FROM directives WHERE id=?", (directive_id,)))
    fates = [dict(r) for r in db.query("SELECT * FROM fates WHERE directive_id=?", (directive_id,))]
    end = at_ep + s.replan_window - 1
    if d["scope"] == "until_episode" and d["until_ep"]:
        end = d["until_ep"]
    if fates:
        end = max(end, max(f["by_ep"] for f in fates) + 5)
    end = min(end, s.total_episodes)
    if at_ep > end:
        return [], "nothing left to re-plan"
    replan = replan_beats(story, d, at_ep, end, fates)
    changes = []
    for b in replan.changed_beats:
        old = db.beat(b.ep)
        if old and b.beat.strip() and b.beat.strip() != old:
            changes.append(BeatChange(b.ep, old, b.beat.strip()))
    return sorted(changes, key=lambda c: c.ep), replan.rationale


def handle_feedback(story: Story, text: str, at_ep: int,
                    approve: Callable[[list[BeatChange], str], bool]) -> FeedbackOutcome:
    """Classify, persist, schedule fates, re-plan (human approves the beat diff)."""
    cls = classify(story, text, at_ep)
    did = add_directive(story, text, at_ep, cls)
    outcome = FeedbackOutcome(did, cls)
    if cls.needs_replan or cls.fates or cls.kind in ("plot", "character_fate"):
        changes, rationale = propose_replan(story, did, at_ep)
        outcome.changes, outcome.rationale = changes, rationale
        if changes and approve(changes, rationale):
            apply_beat_changes(story, [Beat(ep=c.ep, beat=c.new) for c in changes], did, f"directive D{did}")
            outcome.applied = True
            story.tracer.decision(at_ep, "directive:replan", f"D{did}: {len(changes)} beats changed (approved)")
        elif changes:
            story.tracer.decision(at_ep, "directive:replan", f"D{did}: beat changes rejected by human")
    return outcome


def set_active(story: Story, directive_id: int, active: bool) -> None:
    story.db.execute("UPDATE directives SET active=? WHERE id=?", (1 if active else 0, directive_id))


# ---------------------------------------------------------------------------
# Impact reporting
# ---------------------------------------------------------------------------
def impact_report(story: Story) -> list[dict]:
    db = story.db
    out = []
    for d in db.query("SELECT * FROM directives ORDER BY id"):
        eps = db.query("SELECT episode, adherence, note FROM directive_impact WHERE directive_id=? ORDER BY episode",
                       (d["id"],))
        beats = db.query("SELECT ep, old_beat, new_beat FROM beat_history WHERE directive_id=? ORDER BY ep", (d["id"],))
        fates = db.query("SELECT * FROM fates WHERE directive_id=?", (d["id"],))
        scores = [e["adherence"] for e in eps if e["adherence"] is not None]
        out.append({
            "id": d["id"], "text": d["text"], "raw": json.loads(d["classification"] or "{}").get("raw", ""),
            "kind": d["kind"], "scope": d["scope"], "created_at_ep": d["created_at_ep"], "active": bool(d["active"]),
            "episodes": [dict(e) for e in eps],
            "mean_adherence": round(sum(scores) / len(scores), 2) if scores else None,
            "beats_changed": [dict(b) for b in beats],
            "fates": [dict(f) for f in fates],
        })
    return out


def impact_markdown(story: Story) -> str:
    lines = ["# Human interventions and their effects", ""]
    for d in impact_report(story):
        lines += [f"## D{d['id']} — given before episode {d['created_at_ep']}",
                  "", f"**Human said:** \"{d['raw'] or d['text']}\"", "",
                  f"**Stored directive** ({d['kind']}, scope {d['scope']}): {d['text']}", ""]
        for f in d["fates"]:
            done = f"fulfilled in ep {f['fulfilled_ep']}" if f["fulfilled_ep"] else "not yet fulfilled"
            lines.append(f"- Scheduled fate: **{f['name']} → {f['status']} by ep {f['by_ep']}** ({done})")
        if d["beats_changed"]:
            lines += ["", f"**Beats re-planned ({len(d['beats_changed'])}):**", ""]
            for b in d["beats_changed"]:
                lines += [f"- Ep {b['ep']}", f"  - before: {b['old_beat']}", f"  - after: {b['new_beat']}"]
        if d["episodes"]:
            lines += ["", "**Episodes written with this directive in context (judge adherence 1-5):**", ""]
            lines += [f"- Ep {e['episode']}: {e['adherence']} — {e['note'] or ''}" for e in d["episodes"]]
            lines.append(f"\nMean adherence: {d['mean_adherence']}")
        lines.append("")
    return "\n".join(lines)
