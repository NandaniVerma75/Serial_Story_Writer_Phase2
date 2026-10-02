"""Stats, 200-episode cost/time projection, and full-story export."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .checker import CheckResult
from .config import price_for
from .story import Story


def _step_group(step: str) -> str:
    step = step.split(":retry")[0]
    for prefix in ("plan:beats", "check:", "summary:", "directive:", "extract"):
        if step.startswith(prefix):
            return step if prefix == "check:" else prefix.rstrip(":")
    return step


def episode_stats(story: Story) -> list[dict]:
    db = story.db
    agg = {r["episode"]: dict(r) for r in db.query(
        "SELECT episode, SUM(cost_usd) cost, SUM(input_tokens) tin, SUM(output_tokens) tout, "
        "SUM(cache_read_tokens) tcache, SUM(latency_ms) lat, SUM(CASE WHEN model!='' THEN 1 ELSE 0 END) calls, "
        "SUM(retry_count) retries FROM traces GROUP BY episode")}
    out = []
    for r in db.query("SELECT * FROM episodes ORDER BY ep"):
        a = agg.get(r["ep"], {})
        checks = CheckResult.from_json(r["checks"])
        attempts = 1 + int(db.scalar("SELECT COUNT(*) FROM episode_archive WHERE ep=? AND (reason LIKE 'reject%' OR reason LIKE 'regenerate%')", (r["ep"],)) or 0)
        out.append({
            "ep": r["ep"], "status": r["status"], "title": r["title"], "words": r["word_count"],
            "revisions": r["revisions"], "attempts": attempts,
            "cost": round(a.get("cost") or 0, 4), "input_tokens": a.get("tin") or 0,
            "output_tokens": a.get("tout") or 0, "cache_read_tokens": a.get("tcache") or 0,
            "latency_s": round((a.get("lat") or 0) / 1000, 1), "calls": a.get("calls") or 0,
            "retries": a.get("retries") or 0,
            "passed": None if checks is None else checks.passed,
            "rubric": None if checks is None else checks.rubric_mean,
            "similarity": None if checks is None else checks.max_similarity,
            "errors": 0 if checks is None else len(checks.errors()),
            "human_edited": bool(r["human_edited"]),
        })
    return out


def totals(story: Story) -> dict:
    db = story.db
    t = dict(db.one(
        "SELECT COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(input_tokens),0) tin, COALESCE(SUM(output_tokens),0) tout, "
        "COALESCE(SUM(cache_read_tokens),0) tcache, COALESCE(SUM(latency_ms),0) lat, "
        "COALESCE(SUM(CASE WHEN model!='' THEN 1 ELSE 0 END),0) calls, COALESCE(SUM(retry_count),0) retries FROM traces"))
    t["planning_cost"] = float(db.scalar("SELECT COALESCE(SUM(cost_usd),0) FROM traces WHERE episode=0"))
    by_step: dict[str, float] = defaultdict(float)
    for r in db.query("SELECT step, SUM(cost_usd) c FROM traces WHERE episode>0 AND model!='' GROUP BY step"):
        by_step[_step_group(r["step"])] += r["c"] or 0
    t["by_step"] = dict(sorted(by_step.items(), key=lambda kv: -kv[1]))
    t["approved"] = len(db.approved_eps())
    return t


def stats_markdown(story: Story) -> str:
    rows = episode_stats(story)
    t = totals(story)
    lines = [f"# Stats — {story.bible.title}", "",
             "| Ep | Status | Words | Revisions | Attempts | Cost $ | In tok | Out tok | LLM time s | Checks | Rubric | Max sim | Human edit |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        check = "-" if r["passed"] is None else ("pass" if r["passed"] else f"fail ({r['errors']})")
        lines.append(f"| {r['ep']} | {r['status']} | {r['words']} | {r['revisions']} | {r['attempts']} | {r['cost']:.4f} | "
                     f"{r['input_tokens']} | {r['output_tokens']} | {r['latency_s']} | {check} | {r['rubric']} | "
                     f"{r['similarity']} | {'yes' if r['human_edited'] else ''} |")
    lines += ["", "## Totals", "",
              f"- Approved episodes: {t['approved']}",
              f"- Total cost: ${t['cost']:.4f} (planning ${t['planning_cost']:.4f})",
              f"- Tokens: {t['tin']} in (+{t['tcache']} cache reads), {t['tout']} out; {t['calls']} LLM calls, {t['retries']} retries",
              f"- LLM time: {t['lat'] / 1000:.0f}s", "", "## Cost by step (episodes only)", ""]
    lines += [f"- {k}: ${v:.4f}" for k, v in t["by_step"].items()]
    return "\n".join(lines) + "\n"


def estimate(story: Story) -> dict:
    s, db = story.settings, story.db
    t = totals(story)
    n = t["approved"]
    ep_cost = float(db.scalar("SELECT COALESCE(SUM(cost_usd),0) FROM traces WHERE episode BETWEEN 1 AND ?", (max(n, 0),)))
    ep_lat = float(db.scalar("SELECT COALESCE(SUM(latency_ms),0) FROM traces WHERE episode BETWEEN 1 AND ?", (max(n, 0),)))
    measured = n > 0
    avg_cost = ep_cost / n if measured else s.default_episode_cost_usd
    avg_sec = ep_lat / 1000 / n if measured else s.default_episode_seconds
    total_eps = s.total_episodes
    by_step = t["by_step"]
    spent = sum(by_step.values()) or 1.0

    def share(*keys: str) -> float:
        return sum(v for k, v in by_step.items() if any(k.startswith(x) for x in keys)) / spent

    writer_in, writer_out = price_for(s.model_writer)
    cheaper = "claude-sonnet-5-5" if s.provider == "anthropic" else "gpt-4.1-mini"
    c_in, c_out = price_for(cheaper)
    writer_ratio = (c_in + c_out) / (writer_in + writer_out) if writer_in + writer_out else 1
    full = avg_cost * total_eps
    levers = [
        ("Skip the LLM judge when hard + consistency checks pass (SKIP_JUDGE_IF_HARD_PASS=1)",
         share("check:rubric") * 0.8 * full, "loses quality signal + directive adherence measurement"),
        (f"Draft/revise with {cheaper} instead of {s.model_writer}",
         share("draft", "revise", "refine_beat") * (1 - writer_ratio) * full, "prose quality drop; measure with the judge"),
        ("Batch API (50% off) for extraction + summaries, run asynchronously after approval",
         share("extract", "summary") * 0.5 * full, "adds latency before the next episode can start"),
        ("Allow 1 revision instead of 2 (MAX_REVISIONS=1)",
         share("revise") * 0.5 * full, "more issues reach the human"),
        ("Smaller context budget (CONTEXT_BUDGET_TOKENS=8000)",
         share("draft", "refine_beat", "revise") * 0.15 * full, "weaker continuity on long-range details"),
        ("Prompt caching of bible + plan prefix (already on for Anthropic; verify cache reads > 0)",
         0.0, f"measured cache-read tokens so far: {t['tcache']}"),
    ]
    return {
        "measured": measured, "approved": n, "avg_cost": avg_cost, "avg_seconds": avg_sec,
        "planning_cost": t["planning_cost"], "projected_total": t["planning_cost"] + full,
        "projected_remaining": avg_cost * (total_eps - n), "projected_hours": avg_sec * total_eps / 3600,
        "remaining_hours": avg_sec * (total_eps - n) / 3600, "by_step": by_step,
        "levers": sorted(levers, key=lambda x: -x[1]), "total_eps": total_eps,
    }


def estimate_markdown(story: Story) -> str:
    e = estimate(story)
    src = f"measured over {e['approved']} approved episodes" if e["measured"] else "config defaults (nothing measured yet)"
    lines = [
        f"# Cost & time projection — {e['total_eps']} episodes", "", f"Basis: {src}.", "",
        f"- Planning (bible + acts + arcs + 200 beats): ${e['planning_cost']:.3f}",
        f"- Average per episode (all drafts, checks, revisions, rejects, extraction, summaries): ${e['avg_cost']:.4f}",
        f"- Average LLM time per episode: {e['avg_seconds']:.0f}s (excludes human review time)",
        f"- **Projected total for {e['total_eps']} episodes: ${e['projected_total']:.2f}** "
        f"(remaining: ${e['projected_remaining']:.2f})",
        f"- **Projected LLM time: {e['projected_hours']:.1f} h** (remaining: {e['remaining_hours']:.1f} h)", "",
        "## Where the money goes (per-episode steps)", "",
    ]
    total = sum(e["by_step"].values()) or 1
    lines += [f"- {k}: {v / total:.0%}" for k, v in e["by_step"].items()]
    lines += ["", "## Cost-reduction levers (estimated savings over the full run)", "",
              "| Lever | Est. saving | Trade-off |", "|---|---|---|"]
    lines += [f"| {name} | ${saving:.2f} | {tradeoff} |" for name, saving, tradeoff in e["levers"]]
    return "\n".join(lines) + "\n"


def export_markdown(story: Story, path: Path | None = None) -> Path:
    db = story.db
    path = path or story.dir / "story.md"
    b = story.bible
    lines = [f"# {b.title}", "", f"_{b.premise}_", "", f"**Genre:** {b.genre} · **Tone:** {b.tone} · "
             f"**POV:** {b.pov} · **Tense:** {b.tense}", "", "## Plan", ""]
    for a in db.query("SELECT * FROM acts ORDER BY act_no"):
        lines += [f"### Act {a['act_no']}: {a['title']}", "", a["summary"], "", f"_Turning point:_ {a['turning_point']}", ""]
        for arc in db.query("SELECT * FROM arcs WHERE act_no=? ORDER BY arc_no", (a["act_no"],)):
            lines += [f"#### Arc {arc['arc_no']}: {arc['title']} (eps {arc['start_ep']}-{arc['end_ep']})", "", arc["goal"], ""]
            lines += [f"{r['ep']}. {r['beat']}" for r in db.query(
                "SELECT ep, beat FROM beats WHERE ep BETWEEN ? AND ? ORDER BY ep", (arc["start_ep"], arc["end_ep"]))]
            lines.append("")
    lines += ["## Episodes", ""]
    for r in db.query("SELECT * FROM episodes WHERE status='approved' ORDER BY ep"):
        lines += [f"### Episode {r['ep']}: {r['title']}", "", r["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
