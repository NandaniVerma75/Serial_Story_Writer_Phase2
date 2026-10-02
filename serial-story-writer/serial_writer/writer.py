"""Beat refinement, drafting and revision (writer-tier model)."""
from __future__ import annotations

import re

from .context_builder import ContextPack
from .memory import last_timeline, recent_hook_types
from .models import HOOK_TYPES, ScenePlan
from .story import Story, load_prompt


def split_title(text: str, fallback: str) -> tuple[str, str]:
    """Separate a leading '# Title' line from the prose."""
    text = text.strip()
    m = re.match(r"^#+\s*(?:Episode\s+\d+\s*[:.\-–—]\s*)?(.+?)\s*\n", text)
    if m:
        return m.group(1).strip().strip('"*'), text[m.end():].strip()
    return fallback, text


def render_plan(plan: ScenePlan) -> str:
    lines = [
        f"Title: {plan.title}",
        f"When: Day {plan.in_story_day}{', ' + plan.time_label if plan.time_label else ''}"
        f"{' (FLASHBACK)' if plan.is_flashback else ''}",
        f"Characters present: {', '.join(plan.characters)}",
    ]
    if plan.new_characters:
        lines.append(f"New characters: {', '.join(plan.new_characters)}")
    lines += [f"Goal: {plan.goal}", f"Conflict: {plan.conflict}", "Scenes:"]
    lines += [f"  {i}. {sc}" for i, sc in enumerate(plan.scenes, 1)]
    if plan.threads_to_touch:
        lines.append(f"Threads to advance: {'; '.join(plan.threads_to_touch)}")
    lines.append(f"Hook ({plan.hook_type}): {plan.hook_idea}")
    return "\n".join(lines)


def _extra(extra: str) -> str:
    return f"# EXTRA INSTRUCTIONS FOR THIS ATTEMPT\n{extra}" if extra else ""


def refine_beat(story: Story, ctx: ContextPack, extra: str = "") -> ScenePlan:
    db, s = story.db, story.settings
    last = last_timeline(db, ctx.ep)
    recent = recent_hook_types(db, ctx.ep, s.hook_window)
    return story.llm.complete_json(
        ScenePlan, step="refine_beat", episode=ctx.ep, tier="writer", system=story.writer_system(),
        user=load_prompt(
            "refine_beat", ep=ctx.ep, total=s.total_episodes, context=ctx.text(),
            last_day=f"Day {last['day']} ({last['time_label']})" if last else "none yet (start at Day 1)",
            recent_hooks=", ".join(recent) or "none", hook_types=", ".join(HOOK_TYPES),
            extra=f"- {extra}" if extra else "",
        ),
    )


def draft(story: Story, ctx: ContextPack, plan: ScenePlan, extra: str = "") -> tuple[str, str]:
    res = story.llm.complete(
        step="draft", episode=ctx.ep, tier="writer", system=story.writer_system(),
        user=load_prompt(
            "draft", ep=ctx.ep, total=story.settings.total_episodes, context=ctx.text(),
            scene_plan=render_plan(plan), extra=_extra(extra), hook_type=plan.hook_type,
            banned="; ".join(story.banned_phrases()),
        ),
    )
    return split_title(res.text, plan.title)


def revise(story: Story, ctx: ContextPack, plan: ScenePlan, text: str, issues: list[str],
           extra: str = "") -> tuple[str, str]:
    res = story.llm.complete(
        step="revise", episode=ctx.ep, tier="writer", system=story.writer_system(),
        user=load_prompt(
            "revise", ep=ctx.ep, context=ctx.text() + ("\n\n" + _extra(extra) if extra else ""),
            scene_plan=render_plan(plan), issues="\n".join(f"- {i}" for i in issues), draft=text,
            banned="; ".join(story.banned_phrases()),
        ),
    )
    return split_title(res.text, plan.title)
