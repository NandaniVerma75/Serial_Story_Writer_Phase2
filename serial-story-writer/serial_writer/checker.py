"""Self-check of a draft before a human sees it.

Layers, cheapest first:
  * hard checks (code): word count, dead characters acting, unknown/misspelt
    characters, timeline running backwards, banned phrases, overdue scheduled fates
  * consistency (cheap LLM): draft vs retrieved canon -> contradictions with evidence
  * repetition: embedding similarity to every past episode summary + hook-type variety
  * quality rubric (cheap LLM judge, 1-5): hook, momentum, voice, beat & directive adherence
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .context_builder import ContextPack
from .llm import cosine
from .memory import (
    CharacterState, is_dead, last_timeline, match_name, pending_fates, recent_hook_types, words_in,
)
from .models import ConsistencyReport, RubricScore, ScenePlan
from .story import Story, load_prompt

ACTION_VERBS = (
    "said|says|asked|asks|replied|replies|whispered|shouted|yelled|smiled|smiles|nodded|nods|walked|walks|"
    "ran|runs|grabbed|grabs|looked|looks|turned|turns|laughed|laughs|stood|stands|sat|sits|took|takes|opened|"
    "opens|stepped|steps|reached|reaches|answered|answers|muttered|mutters|grinned|grins|shrugged|shrugs"
)


@dataclass
class Issue:
    kind: str
    severity: str  # error | warn
    message: str


@dataclass
class CheckResult:
    passed: bool
    issues: list[Issue] = field(default_factory=list)
    word_count: int = 0
    rubric: dict | None = None
    rubric_mean: float | None = None
    contradictions: list[dict] = field(default_factory=list)
    max_similarity: float = 0.0
    most_similar_ep: int | None = None
    summary: str = ""
    judge_skipped: bool = False

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def fix_list(self) -> list[str]:
        """Issues to hand to the reviser: all errors plus warnings worth fixing while at it."""
        return [f"[{i.kind}] {i.message}" for i in self.issues if i.severity == "error" or i.kind == "style"]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str | None) -> "CheckResult | None":
        if not s:
            return None
        d = json.loads(s)
        d["issues"] = [Issue(**i) for i in d.get("issues", [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# Hard checks (pure code)
# ---------------------------------------------------------------------------
def hard_checks(story: Story, ep: int, text: str, plan: ScenePlan,
                chars: dict[str, CharacterState]) -> list[Issue]:
    s = story.settings
    issues: list[Issue] = []

    wc = words_in(text)
    if wc < s.words_min or wc > s.words_max:
        issues.append(Issue("length", "error", f"word count {wc} is outside {s.words_min}-{s.words_max}"))

    issues += dead_character_issues(text, plan, chars)

    known = list(chars)
    new = {n.lower() for n in plan.new_characters}
    for name in plan.characters:
        if match_name(name, known) or name.lower() in new:
            continue
        close = difflib.get_close_matches(name.lower(), [k.lower() for k in known], n=1, cutoff=0.75)
        if close:
            issues.append(Issue("characters", "error",
                                f"'{name}' is not in the character store — misspelling of '{close[0]}'?"))
        else:
            issues.append(Issue("characters", "warn",
                                f"'{name}' is not in the character store and not declared as new"))

    last = last_timeline(story.db, ep)
    if last and plan.in_story_day < last["day"] and not plan.is_flashback:
        issues.append(Issue("timeline", "error",
                            f"timeline runs backwards: Day {plan.in_story_day} after ep {last['episode_id']} "
                            f"was Day {last['day']} (not marked as flashback)"))

    lowered = text.lower()
    hits = [p for p in story.banned_phrases() if p.lower() in lowered]
    if hits:
        issues.append(Issue("style", "warn", f"banned phrases used: {', '.join(hits)}"))

    for f in pending_fates(story.db, ep):
        if f["by_ep"] is not None and f["by_ep"] < ep:
            issues.append(Issue("directive", "warn",
                                f"scheduled fate overdue: {f['name']} should be {f['status']} by ep {f['by_ep']}"))
    return issues


def dead_character_issues(text: str, plan: ScenePlan, chars: dict[str, CharacterState]) -> list[Issue]:
    issues = []
    present = {match_name(n, list(chars)) for n in plan.characters}
    for name, c in chars.items():
        if not is_dead(c) or plan.is_flashback:
            continue
        if name in present:
            issues.append(Issue("continuity", "error",
                                f"{name} died in ep {c.status_ep} but is listed as present in the scene plan"))
            continue
        for token in {name, name.split()[0]}:
            pattern = rf"\b{re.escape(token)}\b(?:\s+\w+){{0,2}}\s+(?:{ACTION_VERBS})\b|(?:{ACTION_VERBS})\s+{re.escape(token)}\b"
            m = re.search(pattern, text)
            if m:
                issues.append(Issue("continuity", "error",
                                    f"{name} died in ep {c.status_ep} but appears to act alive: \"{m.group(0)}\""))
                break
    return issues


# ---------------------------------------------------------------------------
# LLM checks
# ---------------------------------------------------------------------------
def consistency_check(story: Story, ctx: ContextPack, text: str) -> ConsistencyReport:
    db = story.db
    canon_parts = ["World rules:\n" + "\n".join(f"- {r}" for r in story.bible.world_rules)]
    last = last_timeline(db, ctx.ep)
    if last:
        canon_parts.append(f"Timeline: previous episode was Day {last['day']} ({last['time_label']}).")
    prev = db.get_episode(ctx.ep - 1)
    if prev is not None and prev["summary"]:
        canon_parts.append(f"Previous episode summary: {prev['summary']}")
    canon_parts.append(ctx.text(keys=["characters", "facts", "threads"]))
    return story.llm.complete_json(
        ConsistencyReport, step="check:consistency", episode=ctx.ep, tier="cheap",
        system=story.checker_system(),
        user=load_prompt("consistency", ep=ctx.ep, canon="\n\n".join(canon_parts), draft=text),
    )


def rubric_check(story: Story, ctx: ContextPack, text: str, plan: ScenePlan) -> RubricScore:
    directives = ctx.section("directives")
    return story.llm.complete_json(
        RubricScore, step="check:rubric", episode=ctx.ep, tier="cheap", system=story.checker_system(),
        user=load_prompt(
            "rubric", ep=ctx.ep, beat=story.db.beat(ctx.ep), hook=f"{plan.hook_type}: {plan.hook_idea}",
            directives="\n".join(directives.items) if directives else "(none)", draft=text,
        ),
    )


def repetition_check(story: Story, ep: int, summary: str, plan: ScenePlan) -> tuple[list[Issue], float, int | None, np.ndarray]:
    s = story.settings
    issues: list[Issue] = []
    vec = story.llm.embed(summary, episode=ep, step="check:embed")
    best, best_ep = 0.0, None
    for r in story.db.query("SELECT ep, embedding FROM episodes WHERE status='approved' AND ep < ? AND embedding IS NOT NULL", (ep,)):
        other = np.frombuffer(r["embedding"], dtype=np.float32)
        sim = cosine(vec, other)
        if sim > best:
            best, best_ep = sim, r["ep"]
    if best > s.repetition_threshold:
        issues.append(Issue("repetition", "error",
                            f"this episode's events are too similar to ep {best_ep} (cosine {best:.2f} > "
                            f"{s.repetition_threshold}); change what happens, not just the wording"))
    recent = recent_hook_types(story.db, ep, s.hook_window)
    if recent and plan.hook_type == recent[0]:
        issues.append(Issue("repetition", "error",
                            f"hook type '{plan.hook_type}' repeats the previous episode's hook; end differently"))
    elif plan.hook_type in recent:
        issues.append(Issue("repetition", "warn",
                            f"hook type '{plan.hook_type}' was used in the last {s.hook_window} episodes"))
    return issues, best, best_ep, vec


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_checks(story: Story, ctx: ContextPack, text: str, plan: ScenePlan) -> tuple[CheckResult, np.ndarray]:
    s = story.settings
    issues = hard_checks(story, ctx.ep, text, plan, ctx.characters)
    result = CheckResult(passed=False, word_count=words_in(text))

    report = consistency_check(story, ctx, text)
    for c in report.contradictions:
        sev = "warn" if c.severity == "low" else "error"
        issues.append(Issue("consistency", sev,
                            f"{c.claim} — contradicts canon: {c.conflicts_with}"
                            + (f" (draft: \"{c.evidence}\")" if c.evidence else "")))
    result.contradictions = [c.model_dump() for c in report.contradictions]

    hard_errors = any(i.severity == "error" for i in issues)
    if s.skip_judge_if_hard_pass and not hard_errors:
        result.judge_skipped = True
        summary = f"{plan.goal} {plan.conflict} {' '.join(plan.scenes)}"
    else:
        rubric = rubric_check(story, ctx, text, plan)
        result.rubric = rubric.model_dump()
        result.rubric_mean = round(rubric.mean(), 2)
        summary = rubric.summary or plan.goal
        weak = {k: v for k, v in rubric.dims().items() if v < s.rubric_min_dim}
        if rubric.mean() < s.rubric_pass_mean or weak:
            issues.append(Issue("quality", "error",
                                f"rubric mean {rubric.mean():.2f} (pass {s.rubric_pass_mean}); weak: "
                                f"{weak or 'none'}; judge notes: {rubric.notes}"))
        for d in rubric.directive_scores:
            if d.score < 3 and d.directive_id in ctx.directive_ids:
                issues.append(Issue("directive", "error", f"directive D{d.directive_id} not honoured: {d.note}"))
    result.summary = summary

    rep_issues, sim, sim_ep, vec = repetition_check(story, ctx.ep, summary, plan)
    issues += rep_issues
    result.max_similarity, result.most_similar_ep = round(sim, 3), sim_ep
    result.issues = issues
    result.passed = not any(i.severity == "error" for i in issues)
    return result, vec
