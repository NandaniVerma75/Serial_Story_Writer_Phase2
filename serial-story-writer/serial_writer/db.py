"""SQLite storage. One file per story (`stories/<id>/story.db`).

Every piece of derived story state (character events, facts, threads, timeline)
carries the `episode_id` that produced it. That is what makes retroactive edits
possible: delete everything with `episode_id >= K`, then replay.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- Plan --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS acts (
    act_no INTEGER PRIMARY KEY, title TEXT, summary TEXT, turning_point TEXT,
    character_arcs TEXT);
CREATE TABLE IF NOT EXISTS arcs (
    arc_no INTEGER PRIMARY KEY, act_no INTEGER, title TEXT, goal TEXT,
    start_ep INTEGER, end_ep INTEGER);
CREATE TABLE IF NOT EXISTS beats (
    ep INTEGER PRIMARY KEY, arc_no INTEGER, beat TEXT, revision INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS beat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ep INTEGER, old_beat TEXT, new_beat TEXT,
    directive_id INTEGER, reason TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS plan_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT, opened_arc INTEGER,
    payoff_arc INTEGER);

-- Memory (event-sourced, every row has the episode that caused it) ----------
CREATE TABLE IF NOT EXISTS characters (
    name TEXT PRIMARY KEY, role TEXT, traits TEXT, voice_notes TEXT, arc TEXT,
    origin_ep INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS char_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER, name TEXT,
    kind TEXT, other TEXT, value TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_char_events ON char_events(episode_id, name);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, predicate TEXT, object TEXT,
    episode_id INTEGER, confidence REAL, superseded_ep INTEGER);
CREATE INDEX IF NOT EXISTS ix_facts ON facts(episode_id);
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT, opened_ep INTEGER,
    expected_payoff_arc INTEGER);
CREATE TABLE IF NOT EXISTS thread_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER, episode_id INTEGER,
    action TEXT);
CREATE TABLE IF NOT EXISTS timeline (
    episode_id INTEGER PRIMARY KEY, day INTEGER, time_label TEXT);
CREATE TABLE IF NOT EXISTS arc_summaries (
    arc_no INTEGER PRIMARY KEY, summary TEXT, upto_ep INTEGER);
CREATE TABLE IF NOT EXISTS extractions (episode_id INTEGER PRIMARY KEY, data TEXT);

-- Episodes ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    ep INTEGER PRIMARY KEY, status TEXT, title TEXT, text TEXT, scene_plan TEXT,
    word_count INTEGER, summary TEXT, hook_type TEXT, revisions INTEGER DEFAULT 0,
    checks TEXT, directives TEXT, context_log TEXT, embedding BLOB,
    human_edited INTEGER DEFAULT 0, created_at TEXT, approved_at TEXT);
CREATE TABLE IF NOT EXISTS episode_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ep INTEGER, title TEXT, text TEXT,
    reason TEXT, archived_at TEXT);

-- Human directives ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS directives (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, kind TEXT, scope TEXT,
    scope_target TEXT, until_ep INTEGER, created_at_ep INTEGER, active INTEGER DEFAULT 1,
    created_at TEXT, classification TEXT);
CREATE TABLE IF NOT EXISTS fates (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, status TEXT, by_ep INTEGER,
    directive_id INTEGER, fulfilled_ep INTEGER);
CREATE TABLE IF NOT EXISTS directive_impact (
    episode INTEGER, directive_id INTEGER, adherence REAL, note TEXT,
    PRIMARY KEY (episode, directive_id));

-- Observability ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, episode INTEGER,
    step TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER,
    cache_read_tokens INTEGER, cache_write_tokens INTEGER, cost_usd REAL,
    latency_ms INTEGER, retry_count INTEGER, decision TEXT);
CREATE INDEX IF NOT EXISTS ix_traces_ep ON traces(episode);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class StoryDB:
    """Thin wrapper over a sqlite3 connection with a few typed helpers."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._tx_depth = 0

    # -- basic access -----------------------------------------------------
    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        if self._tx_depth == 0:
            self.conn.commit()
        return cur

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return None if row is None else row[0]

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group writes atomically. Nested calls join the outer transaction."""
        self._tx_depth += 1
        try:
            yield
        except Exception:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.rollback()
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- meta -------------------------------------------------------------
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        if row is None:
            return default
        return json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def del_meta(self, key: str) -> None:
        self.execute("DELETE FROM meta WHERE key=?", (key,))

    # -- episodes -----------------------------------------------------------
    def get_episode(self, ep: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM episodes WHERE ep=?", (ep,))

    def upsert_episode(self, ep: int, **fields: Any) -> None:
        existing = self.get_episode(ep)
        if existing is None:
            fields.setdefault("created_at", now_iso())
            cols = ["ep", *fields.keys()]
            marks = ",".join("?" for _ in cols)
            self.execute(
                f"INSERT INTO episodes({','.join(cols)}) VALUES({marks})",
                (ep, *fields.values()),
            )
        elif fields:
            sets = ",".join(f"{k}=?" for k in fields)
            self.execute(f"UPDATE episodes SET {sets} WHERE ep=?", (*fields.values(), ep))

    def last_approved_ep(self) -> int:
        return int(self.scalar("SELECT COALESCE(MAX(ep), 0) FROM episodes WHERE status='approved'"))

    def in_progress_ep(self) -> int | None:
        return self.scalar("SELECT MIN(ep) FROM episodes WHERE status != 'approved'")

    def approved_eps(self) -> list[int]:
        return [r[0] for r in self.query("SELECT ep FROM episodes WHERE status='approved' ORDER BY ep")]

    # -- plan helpers -------------------------------------------------------
    def beat(self, ep: int) -> str:
        return self.scalar("SELECT beat FROM beats WHERE ep=?", (ep,)) or ""

    def arc_for_ep(self, ep: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM arcs WHERE start_ep<=? AND end_ep>=?", (ep, ep))
