"""Assemble the writing context for episode N under a token budget.

Priority order (1 = most important). Sections 1-3 are never dropped; the rest are
trimmed item-by-item from the bottom (9 first) until the pack fits the budget.

  1. Story bible (style + world rules)            -> lives in the cached system prompt
  2. Active human directives
  3. Beat N, beats N+1..N+3, arc goal, scheduled events
  4. Full text of episode N-1
  5. Summaries of episodes N-5..N-2
  6. "Story so far", current arc rolling summary, earlier arc summaries
  7. Retrieved character cards
  8. Retrieved open threads (due this arc / overdue / untouched > 15 eps)
  9. Relevant fact-ledger entries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import estimate_tokens
from .memory import (
    CharacterState, active_directives, characters_view, directive_line, fact_line, facts_view,
    pending_fates, threads_view,
)
from .story import Story


@dataclass
class Section:
    key: str
    title: str
    items: list[str]
    required: bool = False
    in_system: bool = False  # already in the system prompt: counted, not repeated

    def render(self) -> str:
        return f"# {self.title}\n" + "\n".join(self.items) if self.items else ""

    def tokens(self) -> int:
        return estimate_tokens(self.render()) if self.items else 0


@dataclass
class ContextPack:
    ep: int
    sections: list[Section]
    directive_ids: list[int] = field(default_factory=list)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    retrieved_names: list[str] = field(default_factory=list)
    log: dict = field(default_factory=dict)

    def text(self, keys: list[str] | None = None) -> str:
        parts = [s.render() for s in self.sections
                 if not s.in_system and s.items and (keys is None or s.key in keys)]
        return "\n\n".join(p for p in parts if p)

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)

    def total_tokens(self) -> int:
        return sum(s.tokens() for s in self.sections)


def _mentions(name: str, text: str) -> bool:
    for token in {name, name.split()[0]}:
        if len(token) > 2 and re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
            return True
    return False


def build_context(story: Story, ep: int, budget: int | None = None) -> ContextPack:
    db, s = story.db, story.settings
    budget = budget or s.context_budget_tokens
    sections: list[Section] = []

    # 1. Bible (system prompt)
    sections.append(Section("bible", "STORY BIBLE", [story.writer_system()], required=True, in_system=True))

    # 2. Directives — always included
    directives = active_directives(db, ep)
    sections.append(Section(
        "directives", "ACTIVE HUMAN DIRECTIVES (binding)",
        [directive_line(d) for d in directives] or ["(none)"], required=True))

    # 3. Beats + arc goal + scheduled events
    arc = db.arc_for_ep(ep)
    beat = db.beat(ep)
    beat_items = [f"THIS EPISODE ({ep}) BEAT: {beat}"]
    if arc:
        beat_items.append(
            f"Current arc {arc['arc_no']} '{arc['title']}' (eps {arc['start_ep']}-{arc['end_ep']}, this is "
            f"episode {ep - arc['start_ep'] + 1} of {arc['end_ep'] - arc['start_ep'] + 1}). Arc goal: {arc['goal']}")
    for nxt in range(ep + 1, min(ep + 3, s.total_episodes) + 1):
        beat_items.append(f"Upcoming ep {nxt} (set up, don't execute): {db.beat(nxt)}")
    for f in pending_fates(db, ep):
        if f["by_ep"] is None:
            continue
        if f["by_ep"] == ep:
            beat_items.append(f"SCHEDULED EVENT — MUST HAPPEN IN THIS EPISODE: {f['name']} becomes {f['status']} "
                              f"(directive D{f['directive_id']}).")
        elif f["by_ep"] < ep:
            beat_items.append(f"OVERDUE SCHEDULED EVENT — make it happen now: {f['name']} becomes {f['status']} "
                              f"(was due by ep {f['by_ep']}, directive D{f['directive_id']}).")
        elif f["by_ep"] - ep <= 5:
            beat_items.append(f"Scheduled: {f['name']} becomes {f['status']} by ep {f['by_ep']} "
                              f"(directive D{f['directive_id']}); build toward it.")
    sections.append(Section("beats", "PLAN", beat_items, required=True))

    # 4. Previous episode full text
    prev = db.get_episode(ep - 1)
    prev_items = []
    if prev is not None and prev["status"] == "approved":
        prev_items = [f"Episode {ep - 1}: {prev['title']}\n{prev['text']}"]
    sections.append(Section("prev_episode", "PREVIOUS EPISODE (full text — continue from its last moment)", prev_items))

    # 5. Recent summaries N-2..N-5 (closest first so trimming drops the oldest)
    recent = db.query(
        "SELECT ep, title, summary FROM episodes WHERE status='approved' AND ep BETWEEN ? AND ? ORDER BY ep DESC",
        (ep - 5, ep - 2))
    sections.append(Section("recent", "RECENT EPISODES (summaries)",
                            [f"Ep {r['ep']} '{r['title']}': {r['summary']}" for r in recent]))

    # 6. Story so far + arc summaries
    long_items = []
    so_far = db.get_meta("story_so_far")
    if so_far:
        long_items.append(f"STORY SO FAR (through ep {db.get_meta('story_so_far_upto')}): {so_far}")
    for r in db.query("SELECT a.arc_no, a.title, s.summary, s.upto_ep FROM arc_summaries s JOIN arcs a "
                      "ON a.arc_no=s.arc_no WHERE s.upto_ep < ? ORDER BY a.arc_no DESC", (ep,)):
        long_items.append(f"Arc {r['arc_no']} '{r['title']}' (through ep {r['upto_ep']}): {r['summary']}")
    sections.append(Section("long", "LONG-TERM MEMORY", long_items))

    # 7. Characters: named in beat, active in last 3 eps, and their relationship partners
    chars = characters_view(db, ep)
    lookahead = " ".join([beat] + [db.beat(n) for n in range(ep + 1, min(ep + 1, s.total_episodes) + 1)])
    ordered: list[str] = [n for n in chars if _mentions(n, beat)]
    ordered += [n for n in chars if _mentions(n, lookahead) and n not in ordered]
    ordered += [n for n, c in sorted(chars.items(), key=lambda kv: -(kv[1].last_seen_ep or 0))
                if c.last_seen_ep and c.last_seen_ep >= ep - 3 and n not in ordered]
    for n in list(ordered):
        for other in chars[n].relationships:
            if other in chars and other not in ordered:
                ordered.append(other)
    for f in pending_fates(db, ep):
        if f["name"] in chars and f["name"] not in ordered:
            ordered.insert(0, f["name"])
    facts = facts_view(db, ep)
    cards = []
    for n in ordered:
        own = [fact_line(f) for f in facts if _mentions(n, f["subject"])][:5]
        cards.append(chars[n].card(own))
    sections.append(Section("characters", "CHARACTERS (canon as of now)", cards))

    # 8. Threads: due this arc / overdue first, then untouched for too long, then planned
    arc_no = arc["arc_no"] if arc else 0
    open_threads = [t for t in threads_view(db, ep) if t.status == "open"]
    due = [t for t in open_threads if t.expected_payoff_arc and t.expected_payoff_arc <= arc_no]
    stale = [t for t in open_threads if t not in due and ep - t.last_touched_ep > s.untouched_thread_eps]
    rest = [t for t in open_threads if t not in due and t not in stale]
    thread_items = [f"[T{t.id}] DUE (payoff arc {t.expected_payoff_arc}): {t.description}" for t in due]
    thread_items += [f"[T{t.id}] DON'T FORGET (untouched since ep {t.last_touched_ep}): {t.description}" for t in stale]
    for p in db.query("SELECT * FROM plan_threads WHERE opened_arc=? OR payoff_arc=?", (arc_no, arc_no)):
        verb = "to open" if p["opened_arc"] == arc_no else "to pay off"
        thread_items.append(f"Planned thread {verb} in this arc: {p['description']}")
    thread_items += [f"[T{t.id}] open since ep {t.opened_ep}: {t.description}" for t in rest]
    sections.append(Section("threads", "OPEN THREADS", thread_items))

    # 9. Facts about retrieved characters and places named in the beat
    keywords = {w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", beat)}
    shown = {line for card in cards for line in card.split("\n")}
    fact_items = []
    for f in facts:
        text = f"{f['subject']} {f['object']}"
        if (any(_mentions(n, text) for n in ordered[:8]) or any(k.lower() in text.lower() for k in keywords)) \
                and f"- {fact_line(f)}" not in shown:
            fact_items.append(fact_line(f))
        if len(fact_items) >= 40:
            break
    sections.append(Section("facts", "ESTABLISHED FACTS (do not contradict)", fact_items))

    pack = ContextPack(ep, sections, [d["id"] for d in directives], chars, ordered)
    _fit_budget(pack, budget)
    return pack


TRIM_ORDER = ["facts", "threads", "characters", "long", "recent", "prev_episode"]


def _fit_budget(pack: ContextPack, budget: int) -> None:
    dropped: list[str] = []
    total = pack.total_tokens()
    for key in TRIM_ORDER:
        sec = pack.section(key)
        while total > budget and sec and sec.items:
            if key == "prev_episode" and not any("[trimmed]" in i for i in sec.items):
                # Keep the ending (the hook we continue from) before dropping it entirely.
                text = sec.items[0]
                sec.items[0] = "[trimmed] …" + text[-len(text) // 3:]
                dropped.append("prev_episode:first two-thirds")
            else:
                item = sec.items.pop()
                dropped.append(f"{key}:{item[:60]}")
            total = pack.total_tokens()
        if total <= budget:
            break
    pack.log = {
        "budget": budget,
        "total_tokens": total,
        "over_budget": total > budget,
        "included": {s.key: {"items": len(s.items), "tokens": s.tokens()} for s in pack.sections},
        "dropped": dropped,
        "directives": pack.directive_ids,
    }
