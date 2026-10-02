"""Post-approval memory update: extract what changed, apply it, maintain summaries."""
from __future__ import annotations

from .memory import apply_extraction, characters_view, last_timeline, resolve_thread_refs, threads_view
from .models import HOOK_TYPES, Extraction
from .story import Story, load_prompt


def extract(story: Story, ep: int, text: str) -> Extraction:
    db = story.db
    chars = characters_view(db, ep)
    threads = [t for t in threads_view(db, ep) if t.status == "open"]
    last = last_timeline(db, ep)
    return story.llm.complete_json(
        Extraction, step="extract", episode=ep, tier="cheap", system=story.checker_system(),
        user=load_prompt(
            "extract", ep=ep, text=text,
            characters=", ".join(f"{c.name} ({c.status})" for c in chars.values()) or "none",
            threads="\n".join(f"{t.id}: {t.description}" for t in threads) or "(none)",
            last_day=f"Day {last['day']}" if last else "none (this is the start: Day 1)",
            hook_types=", ".join(HOOK_TYPES),
        ),
    )


def extract_and_apply(story: Story, ep: int, text: str) -> list[str]:
    """LLM extraction + write to memory + summary embedding. Returns warnings."""
    ex = extract(story, ep, text)
    data = resolve_thread_refs(story.db, ep, ex)
    warnings = apply_extraction(story.db, ep, data)
    vec = story.llm.embed(ex.summary, episode=ep, step="extract:embed")
    story.db.upsert_episode(ep, embedding=vec.astype("float32").tobytes())
    return warnings


def update_summaries(story: Story, ep: int, force: bool = False) -> None:
    """Rolling arc summary every 5 episodes and at arc end; refresh 'story so far' at arc end."""
    db = story.db
    arc = db.arc_for_ep(ep)
    if arc is None:
        return
    arc_end = ep == arc["end_ep"]
    if not (force or arc_end or (ep - arc["start_ep"] + 1) % 5 == 0):
        return
    rows = db.query("SELECT ep, summary FROM episodes WHERE status='approved' AND ep BETWEEN ? AND ? ORDER BY ep",
                    (arc["start_ep"], ep))
    if not rows:
        return
    res = story.llm.complete(
        step="summary:arc", episode=ep, tier="cheap", system=story.checker_system(),
        user=load_prompt("arc_summary", arc_no=arc["arc_no"], arc_title=arc["title"], start_ep=arc["start_ep"],
                         upto_ep=ep, arc_goal=arc["goal"],
                         summaries="\n".join(f"Ep {r['ep']}: {r['summary']}" for r in rows)),
    )
    db.execute("INSERT OR REPLACE INTO arc_summaries(arc_no, summary, upto_ep) VALUES(?,?,?)",
               (arc["arc_no"], res.text.strip(), ep))
    if arc_end:
        previous = db.get_meta("story_so_far")
        if not previous:  # invalidated by a retro edit: rebuild from earlier arc summaries
            earlier = db.query("SELECT arc_no, summary FROM arc_summaries WHERE arc_no < ? ORDER BY arc_no",
                               (arc["arc_no"],))
            previous = "\n".join(f"Arc {r['arc_no']}: {r['summary']}" for r in earlier) or "(none yet)"
        res2 = story.llm.complete(
            step="summary:story_so_far", episode=ep, tier="cheap", system=story.checker_system(),
            user=load_prompt("story_so_far", upto_ep=ep, previous=previous,
                             arc_no=arc["arc_no"], arc_summary=res.text.strip()),
        )
        db.set_meta("story_so_far", res2.text.strip())
        db.set_meta("story_so_far_upto", ep)


def rebuild_summaries(story: Story) -> None:
    """After a retroactive edit: regenerate any arc summaries / story-so-far that were invalidated."""
    db = story.db
    last = db.last_approved_ep()
    for arc in db.query("SELECT * FROM arcs WHERE start_ep <= ? ORDER BY arc_no", (last,)):
        if db.one("SELECT 1 FROM arc_summaries WHERE arc_no=?", (arc["arc_no"],)):
            continue
        upto = min(arc["end_ep"], last)
        if upto == arc["end_ep"]:
            update_summaries(story, upto, force=True)
        else:
            # Rolling summary at the last multiple of 5 reached in this arc (if any).
            n = (upto - arc["start_ep"] + 1) // 5 * 5
            if n:
                update_summaries(story, arc["start_ep"] + n - 1, force=True)
