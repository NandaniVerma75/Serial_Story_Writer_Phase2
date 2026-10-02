"""Read/write access to layered story memory, always *as of* a given episode.

Every query takes `before_ep`: the state visible when writing episode N is the
state produced by episodes < N. Because state is event-sourced, the same
functions serve normal writing, consistency checks of old episodes after a
retroactive edit, and replay.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from .db import StoryDB
from .models import Extraction


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------
@dataclass
class CharacterState:
    name: str
    role: str = ""
    traits: str = ""
    voice_notes: str = ""
    arc: str = ""
    origin_ep: int = 0
    status: str = "alive"
    status_detail: str = ""
    status_ep: int | None = None
    relationships: dict[str, tuple[str, str, int]] = field(default_factory=dict)
    last_seen_ep: int | None = None
    appearances: list[int] = field(default_factory=list)

    def card(self, facts: list[str] | None = None) -> str:
        lines = [f"## {self.name} — {self.role}"]
        status = self.status.upper()
        if self.status_ep:
            status += f" (since ep {self.status_ep}{': ' + self.status_detail if self.status_detail else ''})"
        lines.append(f"Status: {status}")
        if self.traits:
            lines.append(f"Traits: {self.traits}")
        if self.voice_notes:
            lines.append(f"Voice: {self.voice_notes}")
        if self.relationships:
            rels = "; ".join(
                f"{other}: {typ}{' — ' + state if state else ''} (ep {ep})"
                for other, (typ, state, ep) in self.relationships.items()
            )
            lines.append(f"Relationships: {rels}")
        lines.append(f"Last seen: ep {self.last_seen_ep}" if self.last_seen_ep else "Not yet on page")
        for f in facts or []:
            lines.append(f"- {f}")
        return "\n".join(lines)


def characters_view(db: StoryDB, before_ep: int) -> dict[str, CharacterState]:
    """Replay character rows + events (< before_ep) into current character state."""
    chars: dict[str, CharacterState] = {}
    for r in db.query("SELECT * FROM characters WHERE origin_ep < ? ORDER BY origin_ep, name", (before_ep,)):
        chars[r["name"]] = CharacterState(
            name=r["name"], role=r["role"] or "", traits=r["traits"] or "",
            voice_notes=r["voice_notes"] or "", arc=r["arc"] or "", origin_ep=r["origin_ep"],
        )
    for e in db.query("SELECT * FROM char_events WHERE episode_id < ? ORDER BY episode_id, id", (before_ep,)):
        c = chars.get(e["name"])
        if c is None:
            continue
        if e["kind"] == "status":
            c.status, c.status_detail, c.status_ep = e["value"], e["detail"] or "", e["episode_id"]
        elif e["kind"] == "relationship":
            c.relationships[e["other"]] = (e["value"], e["detail"] or "", e["episode_id"])
        elif e["kind"] == "seen":
            c.last_seen_ep = e["episode_id"]
            c.appearances.append(e["episode_id"])
    return chars


def match_name(name: str, known: list[str]) -> str | None:
    """Resolve a mention ("Ravi", "ravi kumar") to a canonical character name."""
    n = name.strip().lower()
    if not n:
        return None
    lower = {k.lower(): k for k in known}
    if n in lower:
        return lower[n]
    first = {}
    for k in known:
        first.setdefault(k.split()[0].lower(), []).append(k)
    tok = n.split()[0]
    if tok in first and len(first[tok]) == 1:
        return first[tok][0]
    for k in known:  # "Inspector Rao" -> "Rao"
        if k.lower() in n.split() or n in k.lower().split():
            return k
    close = difflib.get_close_matches(n, list(lower), n=1, cutoff=0.88)
    return lower[close[0]] if close else None


DEAD_STATUSES = {"dead", "deceased", "killed", "died"}


def is_dead(c: CharacterState) -> bool:
    return c.status in DEAD_STATUSES


# ---------------------------------------------------------------------------
# Facts, threads, timeline
# ---------------------------------------------------------------------------
def facts_view(db: StoryDB, before_ep: int) -> list[dict]:
    rows = db.query(
        "SELECT * FROM facts WHERE episode_id < ? AND (superseded_ep IS NULL OR superseded_ep >= ?) "
        "ORDER BY episode_id DESC, id DESC",
        (before_ep, before_ep),
    )
    return [dict(r) for r in rows]


def fact_line(f: dict) -> str:
    return f"{f['subject']} {f['predicate']} {f['object']} (ep {f['episode_id']})"


@dataclass
class ThreadState:
    id: int
    description: str
    opened_ep: int
    expected_payoff_arc: int | None
    status: str = "open"
    last_touched_ep: int = 0


def threads_view(db: StoryDB, before_ep: int) -> list[ThreadState]:
    threads = {
        r["id"]: ThreadState(r["id"], r["description"], r["opened_ep"], r["expected_payoff_arc"],
                             last_touched_ep=r["opened_ep"])
        for r in db.query("SELECT * FROM threads WHERE opened_ep < ? ORDER BY id", (before_ep,))
    }
    for e in db.query("SELECT * FROM thread_events WHERE episode_id < ? ORDER BY episode_id, id", (before_ep,)):
        t = threads.get(e["thread_id"])
        if t is None:
            continue
        t.last_touched_ep = max(t.last_touched_ep, e["episode_id"])
        if e["action"] == "resolve":
            t.status = "resolved"
        elif e["action"] == "abandon":
            t.status = "abandoned"
    return list(threads.values())


def last_timeline(db: StoryDB, before_ep: int) -> dict | None:
    row = db.one("SELECT * FROM timeline WHERE episode_id < ? ORDER BY episode_id DESC LIMIT 1", (before_ep,))
    return dict(row) if row else None


def recent_hook_types(db: StoryDB, before_ep: int, n: int) -> list[str]:
    rows = db.query(
        "SELECT hook_type FROM episodes WHERE status='approved' AND ep < ? ORDER BY ep DESC LIMIT ?",
        (before_ep, n),
    )
    return [r[0] for r in rows if r[0]]


# ---------------------------------------------------------------------------
# Directives and scheduled fates
# ---------------------------------------------------------------------------
def active_directives(db: StoryDB, ep: int) -> list[dict]:
    out = []
    for r in db.query("SELECT * FROM directives WHERE active=1 AND created_at_ep <= ? ORDER BY id", (ep,)):
        if r["scope"] == "until_episode" and r["until_ep"] and ep > r["until_ep"]:
            continue
        if r["scope"] == "arc" and r["scope_target"]:
            arc = db.one("SELECT * FROM arcs WHERE arc_no=?", (_int(r["scope_target"]),))
            if arc and not (arc["start_ep"] <= ep <= arc["end_ep"]):
                continue
        out.append(dict(r))
    return out


def directive_line(d: dict) -> str:
    scope = d["scope"]
    if scope == "until_episode" and d["until_ep"]:
        scope = f"until ep {d['until_ep']}"
    elif scope in ("character", "arc") and d["scope_target"]:
        scope = f"{scope}: {d['scope_target']}"
    return f"[D{d['id']}] ({d['kind']}, {scope}, since ep {d['created_at_ep']}) {d['text']}"


def pending_fates(db: StoryDB, ep: int) -> list[dict]:
    """Scheduled character fates not yet fulfilled before episode `ep`."""
    rows = db.query("SELECT * FROM fates WHERE fulfilled_ep IS NULL OR fulfilled_ep >= ? ORDER BY by_ep", (ep,))
    return [dict(r) for r in rows]


def _int(v) -> int | None:
    try:
        return int(str(v).strip().split()[-1])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Writing extractions into memory (and replaying them)
# ---------------------------------------------------------------------------
def delete_derived_for(db: StoryDB, ep: int) -> None:
    _delete_derived(db, "=", ep)


def delete_derived_from(db: StoryDB, k: int) -> None:
    """Remove all derived state produced by episodes >= k (retroactive edit / regenerate)."""
    _delete_derived(db, ">=", k)
    db.execute("DELETE FROM arc_summaries WHERE upto_ep >= ?", (k,))
    if (db.get_meta("story_so_far_upto") or 0) >= k:
        db.del_meta("story_so_far")
        db.del_meta("story_so_far_upto")


def _delete_derived(db: StoryDB, op: str, ep: int) -> None:
    with db.transaction():
        db.execute(f"DELETE FROM char_events WHERE episode_id {op} ?", (ep,))
        db.execute(f"DELETE FROM facts WHERE episode_id {op} ?", (ep,))
        db.execute(f"UPDATE facts SET superseded_ep=NULL WHERE superseded_ep {op} ?", (ep,))
        db.execute(f"DELETE FROM thread_events WHERE episode_id {op} ?", (ep,))
        db.execute(f"DELETE FROM thread_events WHERE thread_id IN (SELECT id FROM threads WHERE opened_ep {op} ?)", (ep,))
        db.execute(f"DELETE FROM threads WHERE opened_ep {op} ?", (ep,))
        db.execute(f"DELETE FROM timeline WHERE episode_id {op} ?", (ep,))
        db.execute(f"DELETE FROM characters WHERE origin_ep {op} ? AND origin_ep > 0", (ep,))
        db.execute(f"UPDATE fates SET fulfilled_ep=NULL WHERE fulfilled_ep {op} ?", (ep,))


def resolve_thread_refs(db: StoryDB, ep: int, ex: Extraction) -> dict:
    """Turn the model's thread ids into descriptions so the extraction can be replayed later
    even after thread ids are regenerated by a retroactive edit."""
    data = ex.model_dump()
    by_id = {t.id: t.description for t in threads_view(db, ep)}
    data["threads_touched_desc"] = [by_id[i] for i in ex.threads_touched if i in by_id]
    data["threads_resolved_desc"] = [by_id[i] for i in ex.threads_resolved if i in by_id]
    return data


def apply_extraction(db: StoryDB, ep: int, data: dict) -> list[str]:
    """Write one episode's extraction into memory. Idempotent for that episode.

    Returns warnings (e.g. a referenced thread no longer exists after a retro edit).
    """
    warnings: list[str] = []
    ex = Extraction.model_validate(data)
    with db.transaction():
        delete_derived_for(db, ep)
        known = list(characters_view(db, ep + 1).keys())

        for nc in ex.new_characters:
            if match_name(nc.name, known) is None:
                db.execute(
                    "INSERT OR IGNORE INTO characters(name, role, traits, voice_notes, arc, origin_ep) "
                    "VALUES(?,?,?,?,?,?)",
                    (nc.name.strip(), nc.role, nc.traits, nc.voice_notes, "", ep),
                )
                known.append(nc.name.strip())

        def canon(name: str) -> str | None:
            m = match_name(name, known)
            if m is None and name.strip():
                db.execute(
                    "INSERT OR IGNORE INTO characters(name, role, traits, voice_notes, arc, origin_ep) "
                    "VALUES(?,?,?,?,?,?)", (name.strip(), "minor (auto-added)", "", "", "", ep))
                known.append(name.strip())
                m = name.strip()
            return m

        for name in ex.characters_present:
            c = canon(name)
            if c:
                db.execute("INSERT INTO char_events(episode_id,name,kind) VALUES(?,?,'seen')", (ep, c))
        for sc in ex.status_changes:
            c = canon(sc.name)
            if c:
                db.execute(
                    "INSERT INTO char_events(episode_id,name,kind,value,detail) VALUES(?,?,'status',?,?)",
                    (ep, c, sc.status, sc.detail))
                for fate in db.query("SELECT * FROM fates WHERE fulfilled_ep IS NULL"):
                    if match_name(fate["name"], [c]) and fate["status"] == sc.status:
                        db.execute("UPDATE fates SET fulfilled_ep=? WHERE id=?", (ep, fate["id"]))
        for rc in ex.relationship_changes:
            a, b = canon(rc.a), canon(rc.b)
            if a and b:
                for x, y in ((a, b), (b, a)):
                    db.execute(
                        "INSERT INTO char_events(episode_id,name,kind,other,value,detail) "
                        "VALUES(?,?,'relationship',?,?,?)", (ep, x, y, rc.type, rc.state))
        for f in ex.facts:
            db.execute(
                "UPDATE facts SET superseded_ep=? WHERE lower(subject)=lower(?) AND lower(predicate)=lower(?) "
                "AND lower(object)!=lower(?) AND superseded_ep IS NULL AND episode_id < ?",
                (ep, f.subject, f.predicate, f.object, ep))
            db.execute(
                "INSERT INTO facts(subject,predicate,object,episode_id,confidence) VALUES(?,?,?,?,?)",
                (f.subject, f.predicate, f.object, ep, f.confidence))
        for t in ex.threads_opened:
            cur = db.execute(
                "INSERT INTO threads(description, opened_ep, expected_payoff_arc) VALUES(?,?,?)",
                (t.description, ep, t.expected_payoff_arc))
            db.execute("INSERT INTO thread_events(thread_id, episode_id, action) VALUES(?,?,'open')",
                       (cur.lastrowid, ep))
        for action, key in (("touch", "threads_touched_desc"), ("resolve", "threads_resolved_desc")):
            for desc in data.get(key, []):
                row = db.one("SELECT id FROM threads WHERE description=? AND opened_ep < ? ORDER BY id DESC",
                             (desc, ep))
                if row is None:
                    warnings.append(f"ep {ep} {action}s thread that no longer exists: '{desc}'")
                    continue
                db.execute("INSERT INTO thread_events(thread_id, episode_id, action) VALUES(?,?,?)",
                           (row["id"], ep, action))
        db.execute("INSERT OR REPLACE INTO timeline(episode_id, day, time_label) VALUES(?,?,?)",
                   (ep, ex.in_story_day, ex.time_label))
        prev = last_timeline(db, ep)
        if prev and ex.in_story_day < prev["day"]:
            warnings.append(f"ep {ep} timeline day {ex.in_story_day} is before ep {prev['episode_id']} day {prev['day']}")
        db.execute("INSERT OR REPLACE INTO extractions(episode_id, data) VALUES(?,?)", (ep, json.dumps(data)))
        db.upsert_episode(ep, summary=ex.summary, hook_type=ex.hook_type or None)
    return warnings


def seed_characters(db: StoryDB, seeds) -> None:
    """Characters from the bible are canon from 'episode 0'."""
    with db.transaction():
        for c in seeds:
            db.execute(
                "INSERT OR REPLACE INTO characters(name, role, traits, voice_notes, arc, origin_ep) "
                "VALUES(?,?,?,?,?,0)", (c.name, c.role, c.traits, c.voice_notes, c.arc))
        db.execute("DELETE FROM char_events WHERE episode_id=0")
        names = [c.name for c in seeds]
        for c in seeds:
            for r in c.relationships:
                other = match_name(r.other, names) or r.other
                db.execute(
                    "INSERT INTO char_events(episode_id,name,kind,other,value,detail) VALUES(0,?,'relationship',?,?,?)",
                    (c.name, other, r.type, r.state))


def words_in(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))
