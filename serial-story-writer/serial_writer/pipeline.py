"""Per-episode state machine:

    plan_beat -> draft -> self-check -> (revise <= 2) -> HUMAN REVIEW -> commit

Each stage persists before the next starts (episodes.status = planned | drafted |
review | approved), so a crash or a pause at any point resumes exactly where it
stopped — including mid-review of an already-drafted episode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol

from .checker import CheckResult, run_checks
from .context_builder import build_context
from .db import now_iso
from .extractor import extract_and_apply, update_summaries
from .memory import characters_view, is_dead, last_timeline, match_name, words_in
from .models import ScenePlan
from .story import Story
from .writer import draft, refine_beat, revise


@dataclass
class ReviewDecision:
    kind: str  # approve | edit | reject | regenerate | pause
    text: str = ""


class Reviewer(Protocol):
    def review(self, story: Story, ep: int, checks: CheckResult | None) -> ReviewDecision: ...


class AutoReviewer:
    """Approves episodes that pass every check. On failure: hand to `fallback` (a human),
    approve anyway (`on_fail='approve'`, recorded in the trace), or pause."""

    def __init__(self, on_fail: str = "pause", fallback: Reviewer | None = None):
        self.on_fail = on_fail
        self.fallback = fallback

    def review(self, story: Story, ep: int, checks: CheckResult | None) -> ReviewDecision:
        if checks is not None and checks.passed:
            story.tracer.decision(ep, "review", "auto-approved (all checks passed)")
            return ReviewDecision("approve")
        if self.on_fail == "approve":
            msgs = "; ".join(i.message[:80] for i in (checks.errors() if checks else []))
            story.tracer.decision(ep, "review", f"auto-approved WITH open issues: {msgs}")
            return ReviewDecision("approve")
        if self.fallback is not None:
            story.tracer.decision(ep, "review", "check failure -> paused for human review")
            return self.fallback.review(story, ep, checks)
        story.tracer.decision(ep, "review", "check failure -> paused")
        return ReviewDecision("pause")


class Pipeline:
    def __init__(self, story: Story, reviewer: Reviewer, log: Callable[[str], None] = lambda m: None):
        self.story = story
        self.db = story.db
        self.reviewer = reviewer
        self.log = log

    # -- helpers ---------------------------------------------------------------
    def next_episode(self) -> int:
        in_progress = self.db.in_progress_ep()
        return in_progress if in_progress is not None else self.db.last_approved_ep() + 1

    def episode_cost(self, ep: int) -> float:
        row = self.db.get_episode(ep)
        since = row["created_at"] if row else ""
        return float(self.db.scalar(
            "SELECT COALESCE(SUM(cost_usd),0) FROM traces WHERE episode=? AND ts>=?", (ep, since or "")))

    def _extra(self, ep: int) -> str:
        reasons = self.db.get_meta(f"reject:{ep}") or []
        if not reasons:
            return ""
        return ("The showrunner rejected previous attempts at this episode. Do not repeat these problems:\n"
                + "\n".join(f"- {r}" for r in reasons))

    def _plan_obj(self, ep: int) -> ScenePlan:
        return ScenePlan.model_validate_json(self.db.get_episode(ep)["scene_plan"])

    # -- stages ------------------------------------------------------------------
    def run_episode(self, ep: int) -> str:
        """Drive episode `ep` to approval. Returns 'approved' or 'paused'."""
        if ep > self.story.settings.total_episodes:
            return "complete"
        while True:
            row = self.db.get_episode(ep)
            if row is None:
                self.stage_plan(ep)
                row = self.db.get_episode(ep)
            if row["status"] == "planned":
                self.stage_draft(ep)
                row = self.db.get_episode(ep)
            if row["status"] == "drafted":
                self.stage_check(ep)
                row = self.db.get_episode(ep)
            if row["status"] == "approved":
                return "approved"
            checks = CheckResult.from_json(row["checks"])
            decision = self.reviewer.review(self.story, ep, checks)
            if decision.kind == "approve":
                self.commit(ep)
                return "approved"
            if decision.kind == "edit":
                self.db.upsert_episode(ep, text=decision.text, word_count=words_in(decision.text), human_edited=1)
                self.story.tracer.decision(ep, "review", "human edited text; edited version becomes canon")
                self.commit(ep)
                return "approved"
            if decision.kind in ("reject", "regenerate"):
                if decision.kind == "reject":
                    reasons = self.db.get_meta(f"reject:{ep}") or []
                    self.db.set_meta(f"reject:{ep}", reasons + [decision.text])
                self.story.tracer.decision(ep, "review", f"{decision.kind}: {decision.text[:150]}")
                self.discard(ep, decision.kind + (": " + decision.text if decision.text else ""))
                continue
            self.story.tracer.decision(ep, "review", "paused by human")
            return "paused"

    def stage_plan(self, ep: int) -> None:
        self.log(f"ep {ep}: refining beat into a scene plan")
        started = now_iso()
        ctx = build_context(self.story, ep)
        self.story.tracer.decision(
            ep, "context",
            f"tokens {ctx.log['total_tokens']}/{ctx.log['budget']}; dropped {len(ctx.log['dropped'])} items; "
            f"directives {ctx.directive_ids}")
        plan = refine_beat(self.story, ctx, self._extra(ep))
        self.db.upsert_episode(ep, status="planned", title=plan.title, scene_plan=plan.model_dump_json(),
                               context_log=json.dumps(ctx.log), directives=json.dumps(ctx.directive_ids),
                               created_at=started)

    def stage_draft(self, ep: int) -> None:
        self.log(f"ep {ep}: drafting")
        ctx = build_context(self.story, ep)
        plan = self._plan_obj(ep)
        title, text = draft(self.story, ctx, plan, self._extra(ep))
        self.db.upsert_episode(ep, status="drafted", title=title, text=text, word_count=words_in(text))

    def stage_check(self, ep: int) -> None:
        s = self.story.settings
        ctx = build_context(self.story, ep)
        while True:
            row = self.db.get_episode(ep)
            plan = self._plan_obj(ep)
            self.log(f"ep {ep}: checking (revision {row['revisions']})")
            result, _ = run_checks(self.story, ctx, row["text"], plan)
            self.db.upsert_episode(ep, checks=result.to_json())
            if result.passed:
                self.story.tracer.decision(ep, "check", f"pass (rubric {result.rubric_mean}, sim {result.max_similarity})")
                break
            problems = "; ".join(i.message[:90] for i in result.errors())
            if row["revisions"] >= s.max_revisions:
                self.story.tracer.decision(ep, "check", f"max revisions reached -> human: {problems}")
                break
            cost = self.episode_cost(ep)
            if cost >= s.episode_cost_cap_usd:
                self.story.tracer.decision(ep, "check", f"episode cost cap hit (${cost:.3f}) -> human: {problems}")
                break
            self.story.tracer.decision(ep, "check", f"revise: {problems}")
            self.log(f"ep {ep}: revising — {problems[:160]}")
            plan = self._repair_plan(ep, plan, result)
            title, text = revise(self.story, ctx, plan, row["text"], result.fix_list(), self._extra(ep))
            self.db.upsert_episode(ep, title=title, text=text, word_count=words_in(text),
                                   revisions=row["revisions"] + 1, scene_plan=plan.model_dump_json())
        self.db.upsert_episode(ep, status="review")

    def _repair_plan(self, ep: int, plan: ScenePlan, result: CheckResult) -> ScenePlan:
        """Deterministic fixes for plan-level hard failures before asking for a revision."""
        kinds = {i.kind for i in result.errors()}
        if "timeline" in kinds:
            last = last_timeline(self.db, ep)
            if last:
                plan.in_story_day = last["day"]
        if "continuity" in kinds:
            chars = characters_view(self.db, ep)
            plan.characters = [n for n in plan.characters
                               if not ((m := match_name(n, list(chars))) and is_dead(chars[m]))]
        return plan

    # -- commit / discard ------------------------------------------------------------
    def commit(self, ep: int) -> list[str]:
        self.log(f"ep {ep}: approved — extracting facts, characters, threads, timeline")
        row = self.db.get_episode(ep)
        warnings = extract_and_apply(self.story, ep, row["text"])
        checks = CheckResult.from_json(row["checks"])
        scores = {}
        if checks and checks.rubric:
            scores = {d["directive_id"]: d for d in checks.rubric.get("directive_scores", [])}
        with self.db.transaction():
            for did in json.loads(row["directives"] or "[]"):
                d = scores.get(did, {})
                self.db.execute(
                    "INSERT OR REPLACE INTO directive_impact(episode, directive_id, adherence, note) VALUES(?,?,?,?)",
                    (ep, did, d.get("score"), d.get("note", "")))
            self.db.upsert_episode(ep, status="approved", approved_at=now_iso())
        self.db.del_meta(f"reject:{ep}")
        self.story.write_episode_md(ep)
        update_summaries(self.story, ep)
        self.story.tracer.decision(ep, "commit", "approved" + (f"; warnings: {warnings}" if warnings else ""))
        return warnings

    def discard(self, ep: int, reason: str) -> None:
        row = self.db.get_episode(ep)
        if row is None:
            return
        with self.db.transaction():
            if row["text"]:
                self.db.execute("INSERT INTO episode_archive(ep, title, text, reason, archived_at) VALUES(?,?,?,?,?)",
                                (ep, row["title"], row["text"], reason, now_iso()))
            self.db.execute("DELETE FROM episodes WHERE ep=?", (ep,))
