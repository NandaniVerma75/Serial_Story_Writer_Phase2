"""Observability: every LLM call and pipeline decision becomes one trace row.

Rows go to the `traces` table (queried by `story stats` / `story estimate`) and are
mirrored to `stories/<id>/trace.jsonl` for grepping and for the demo bundle.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .db import StoryDB, now_iso


class Tracer:
    def __init__(self, db: StoryDB | None, jsonl_path: Path | None, run_id: str | None = None):
        self.db = db
        self.jsonl_path = jsonl_path
        self.run_id = run_id or uuid.uuid4().hex[:10]

    def log(
        self,
        *,
        episode: int,
        step: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        retry_count: int = 0,
        decision: str = "",
    ) -> None:
        row = {
            "run_id": self.run_id,
            "ts": now_iso(),
            "episode": episode,
            "step": step,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "decision": decision,
        }
        if self.db is not None:
            cols = ",".join(row)
            marks = ",".join("?" for _ in row)
            self.db.execute(f"INSERT INTO traces({cols}) VALUES({marks})", tuple(row.values()))
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

    def decision(self, episode: int, step: str, decision: str) -> None:
        """Record a non-LLM decision (pass / revise / cap hit / human approve ...)."""
        self.log(episode=episode, step=step, decision=decision)

    def run_cost(self) -> float:
        if self.db is None:
            return 0.0
        return float(
            self.db.scalar("SELECT COALESCE(SUM(cost_usd),0) FROM traces WHERE run_id=?", (self.run_id,))
        )
