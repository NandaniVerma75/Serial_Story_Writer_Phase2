"""`story` command-line interface."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import hitl, reports
from .config import load_settings
from .directives import impact_report, impact_markdown, set_active
from .hitl import console
from .llm import BudgetExceeded, LLMError
from .pipeline import AutoReviewer, Pipeline
from .planner import generate_plan
from .reconcile import apply_edit, resolve
from .story import Story

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Agentic serial story writer: 200 episodes, human in the loop.")


# ---------------------------------------------------------------------------
# Story selection
# ---------------------------------------------------------------------------
def _current_file() -> Path:
    s = load_settings()
    s.stories_dir.mkdir(parents=True, exist_ok=True)
    return s.stories_dir / ".current"


def _set_current(story_id: str) -> None:
    _current_file().write_text(story_id)


def _open(story_id: Optional[str]) -> Story:
    sid = story_id or (_current_file().read_text().strip() if _current_file().exists() else None)
    if not sid:
        console.print("[red]No story selected. Use `story new` or pass --story <id>.[/]")
        raise typer.Exit(1)
    try:
        return Story.open(sid)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)


StoryOpt = typer.Option(None, "--story", "-s", help="Story id (defaults to the current story)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@app.command()
def new(premise: str = typer.Argument(..., help="One-paragraph premise"),
        story_id: Optional[str] = typer.Option(None, "--id", help="Folder name for the story"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Approve the generated plan without review")):
    """Create a story: bible + hierarchical 200-episode plan, then plan approval."""
    settings = load_settings()
    sid = story_id or Story.new_id(premise)
    story = Story.open(sid, settings, create=True)
    _set_current(sid)
    console.print(f"Story [bold]{sid}[/] → {story.dir}  (writer: {settings.model_writer}, cheap: {settings.model_cheap})")
    try:
        with console.status("Planning…") as status:
            warnings = generate_plan(story, premise=premise, progress=lambda m: status.update(m))
    except (LLMError, BudgetExceeded) as exc:
        console.print(f"[red]Planning failed: {exc}[/]")
        raise typer.Exit(1)
    story.db.set_meta("plan_warnings", warnings)
    cost = story.tracer.run_cost()
    console.print(f"[green]Plan created[/]: 5 acts, 10 arcs, 200 beats · planning cost ${cost:.3f} · "
                  f"{story.dir / 'plan.yaml'}")
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/]")
    if yes:
        story.db.set_meta("plan_status", "approved")
        story.tracer.decision(0, "plan", "plan auto-approved (--yes)")
        reports.export_markdown(story)
        console.print("Plan approved (--yes). Next: story write")
    elif hitl.plan_approval(story):
        console.print("Next: [bold]story write[/]  (or story write --auto 5)")


def _ensure_plan(story: Story) -> bool:
    if story.db.get_meta("plan_status") == "approved":
        return True
    console.print("[yellow]The plan has not been approved yet.[/]")
    return hitl.plan_approval(story)


def _write(story: Story, auto: int, on_fail: str, count: int) -> None:
    if not _ensure_plan(story):
        return
    interactive = sys.stdin.isatty()
    if auto:
        fallback = hitl.InteractiveReviewer() if (on_fail == "pause" and interactive) else None
        reviewer = AutoReviewer(on_fail=on_fail, fallback=fallback)
        target = auto
    else:
        reviewer = hitl.InteractiveReviewer()
        target = count
    pipe = Pipeline(story, reviewer, log=lambda m: console.print(f"[dim]{m}[/]"))
    done = 0
    try:
        while done < target:
            ep = pipe.next_episode()
            if ep > story.settings.total_episodes:
                console.print("[green]All episodes written.[/]")
                return
            console.rule(f"Episode {ep}: {story.db.beat(ep)[:90]}")
            result = pipe.run_episode(ep)
            if result != "approved":
                console.print(f"[yellow]Paused at episode {ep}. Resume with: story resume {story.id}[/]")
                return
            row = story.db.get_episode(ep)
            console.print(f"[green]✓ Episode {ep} approved[/] — {row['title']} · {row['word_count']} words · "
                          f"${pipe.episode_cost(ep):.4f} · revisions {row['revisions']}")
            done += 1
            if not auto and done >= target and interactive and Confirm.ask("Write the next episode?", default=True):
                target += 1
    except BudgetExceeded as exc:
        console.print(f"[yellow]{exc} Run paused; everything is saved. Resume with: story resume {story.id}[/]")
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted. State is saved after every step. Resume with: story resume {story.id}[/]")
    except LLMError as exc:
        console.print(f"[red]LLM error: {exc}. State is saved; retry with: story resume {story.id}[/]")
        raise typer.Exit(1)
    console.print(f"Run cost: ${story.tracer.run_cost():.4f}")


@app.command()
def write(auto: int = typer.Option(0, "--auto", help="Write N episodes without stopping (pauses on check failure / caps)"),
          on_fail: str = typer.Option("pause", "--on-fail", help="Auto mode on check failure: pause | approve"),
          count: int = typer.Option(1, "--count", "-n", help="Interactive: episodes to write before asking"),
          story_id: Optional[str] = StoryOpt):
    """Write the next episode(s), with human review."""
    _write(_open(story_id), auto, on_fail, count)


@app.command()
def resume(story_id: str = typer.Argument(...),
           auto: int = typer.Option(0, "--auto"),
           on_fail: str = typer.Option("pause", "--on-fail")):
    """Continue a story exactly where it stopped (plan approval, mid-draft, mid-review)."""
    story = _open(story_id)
    _set_current(story_id)
    nxt = Pipeline(story, AutoReviewer()).next_episode()
    row = story.db.get_episode(nxt)
    stage = row["status"] if row else "not started"
    console.print(f"Resuming [bold]{story_id}[/]: {story.db.last_approved_ep()} episodes approved; "
                  f"episode {nxt} is at stage '{stage}'.")
    _write(story, auto, on_fail, 1)


@app.command()
def edit(episode: int,
         from_file: Optional[Path] = typer.Option(None, "--file", help="Use this file's text instead of $EDITOR"),
         option: Optional[str] = typer.Option(None, "--option", help="keep | revise | regenerate (skip the prompt)"),
         window: Optional[int] = typer.Option(None, "--window", help="How many later episodes to re-check"),
         story_id: Optional[str] = StoryOpt):
    """Retroactively edit an episode and reconcile everything after it."""
    story = _open(story_id)
    row = story.db.get_episode(episode)
    if row is None:
        console.print(f"[red]Episode {episode} does not exist.[/]")
        raise typer.Exit(1)
    if from_file:
        new_text = from_file.read_text(encoding="utf-8").strip()
    else:
        new_text = click.edit(row["text"], extension=".md", require_save=True)
        if not new_text or new_text.strip() == row["text"].strip():
            console.print("No changes.")
            return
        new_text = new_text.strip()
    with console.status("Reconciling…") as status:
        report = apply_edit(story, episode, new_text, window, progress=lambda m: status.update(m))
    if not report.later:
        console.print(f"[green]Episode {episode} updated; memory rebuilt. No later episodes to reconcile.[/]")
        return
    t = Table("Ep", "Conflicts with the new canon", title=f"Downstream check after editing ep {episode}", show_lines=True)
    for j in report.checked:
        issues = report.flagged.get(j)
        t.add_row(str(j), "\n".join(f"[red]• {i}[/]" for i in issues) if issues else "[green]consistent[/]")
    console.print(t)
    unchecked = [j for j in report.later if j not in report.checked]
    if unchecked:
        console.print(f"[dim]Not re-checked (outside window): eps {unchecked[0]}-{unchecked[-1]} "
                      f"(memory was still replayed for them).[/]")
    if not option:
        option = Prompt.ask("[k]eep as is  [r]evise flagged episodes  [g]enerate again from ep "
                            f"{episode + 1}", choices=["k", "r", "g"], default="k" if not report.flagged else "r")
        option = {"k": "keep", "r": "revise", "g": "regenerate"}[option]
    with console.status(f"Applying '{option}'…") as status:
        changed = resolve(story, report, option, progress=lambda m: status.update(m))
    if option == "regenerate":
        console.print(f"Discarded {len(changed)} episodes (archived). Run `story write` to regenerate from ep {episode + 1}.")
    elif changed:
        console.print(f"[green]Revised episodes {changed} to fit the edited canon.[/]")
    else:
        console.print("Kept later episodes unchanged.")


@app.command()
def feedback(text: str, yes: bool = typer.Option(False, "--yes", "-y", help="Apply re-planned beats without asking"),
             at_ep: Optional[int] = typer.Option(None, "--at", help="Episode the directive starts at (default: next)"),
             story_id: Optional[str] = StoryOpt):
    """Add a persistent directive outside the review loop (re-plans affected beats)."""
    story = _open(story_id)
    ep = at_ep or Pipeline(story, AutoReviewer()).next_episode()
    try:
        hitl.feedback_flow(story, text, ep, auto_yes=yes)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)


@app.command()
def show(what: str = typer.Argument(..., help="plan | characters | threads | facts | directives | episode | arc"),
         n: Optional[int] = typer.Argument(None, help="Episode / arc number"),
         subject: Optional[str] = typer.Option(None, "--subject", help="Filter facts by subject"),
         story_id: Optional[str] = StoryOpt):
    """Inspect the plan, memory, or an episode."""
    story = _open(story_id)
    upto = story.db.last_approved_ep() + 1
    if what == "plan":
        if n:
            hitl.show_arc_beats(story, n)
        else:
            hitl.show_plan_overview(story)
            console.print("[dim]`story show arc <n>` for beats; full plan in plan.yaml[/]")
    elif what == "arc":
        hitl.show_arc_beats(story, n or 1)
    elif what == "characters":
        hitl.show_characters(story, upto)
    elif what == "threads":
        hitl.show_threads(story, upto)
    elif what == "facts":
        hitl.show_facts(story, upto, subject, limit=200)
    elif what == "directives":
        hitl.show_directives(story, upto)
    elif what == "episode":
        hitl.show_episode(story, n or story.db.last_approved_ep())
    else:
        console.print(f"[red]Unknown: {what}[/]")


@app.command()
def directives(impact: bool = typer.Option(False, "--impact", help="Show which episodes each directive influenced"),
               md: Optional[Path] = typer.Option(None, "--md", help="Write the impact report as Markdown"),
               deactivate: Optional[int] = typer.Option(None, "--deactivate"),
               activate: Optional[int] = typer.Option(None, "--activate"),
               story_id: Optional[str] = StoryOpt):
    """List directives; --impact shows propagation (beats changed, episodes, adherence)."""
    story = _open(story_id)
    if deactivate:
        set_active(story, deactivate, False)
    if activate:
        set_active(story, activate, True)
    if md:
        md.write_text(impact_markdown(story), encoding="utf-8")
        console.print(f"Wrote {md}")
    if not impact:
        hitl.show_directives(story, story.db.last_approved_ep() + 1)
        return
    t = Table("ID", "Directive", "Beats changed", "Episodes influenced (adherence)", "Mean", show_lines=True)
    for d in impact_report(story):
        eps = ", ".join(f"{e['episode']}({e['adherence'] if e['adherence'] is not None else '-'})" for e in d["episodes"])
        fates = "".join(f"\nfate: {f['name']}→{f['status']} by {f['by_ep']}, "
                        f"{'done ep ' + str(f['fulfilled_ep']) if f['fulfilled_ep'] else 'pending'}" for f in d["fates"])
        t.add_row(f"D{d['id']}", f"{d['text']}{fates}",
                  ", ".join(str(b["ep"]) for b in d["beats_changed"]) or "-", eps or "-", str(d["mean_adherence"] or "-"))
    console.print(t)


@app.command()
def stats(md: Optional[Path] = typer.Option(None, "--md", help="Also write stats as Markdown"),
          story_id: Optional[str] = StoryOpt):
    """Per-episode cost, tokens, latency, revisions and check results."""
    story = _open(story_id)
    rows = reports.episode_stats(story)
    t = Table("Ep", "Status", "Words", "Rev", "Cost $", "In tok", "Out tok", "LLM s", "Checks", "Rubric", "Sim",
              title=f"Stats — {story.bible.title}")
    for r in rows:
        check = "-" if r["passed"] is None else ("[green]pass[/]" if r["passed"] else f"[red]fail({r['errors']})[/]")
        t.add_row(str(r["ep"]), r["status"], str(r["words"]), str(r["revisions"]), f"{r['cost']:.4f}",
                  str(r["input_tokens"]), str(r["output_tokens"]), str(r["latency_s"]), check,
                  str(r["rubric"]), str(r["similarity"]))
    console.print(t)
    tot = reports.totals(story)
    console.print(Panel(
        f"Approved: {tot['approved']} · Total ${tot['cost']:.4f} (planning ${tot['planning_cost']:.4f}) · "
        f"{tot['tin']} in / {tot['tout']} out tokens · {tot['tcache']} cache-read · {tot['calls']} calls · "
        f"{tot['retries']} retries · {tot['lat'] / 1000:.0f}s LLM time\n"
        + " · ".join(f"{k} ${v:.3f}" for k, v in tot["by_step"].items()), title="Totals"))
    if md:
        md.write_text(reports.stats_markdown(story), encoding="utf-8")
        console.print(f"Wrote {md}")


@app.command()
def estimate(md: Optional[Path] = typer.Option(None, "--md"), story_id: Optional[str] = StoryOpt):
    """Project cost and time for all 200 episodes from measured averages."""
    story = _open(story_id)
    text = reports.estimate_markdown(story)
    console.print(text)
    if md:
        md.write_text(text, encoding="utf-8")


@app.command()
def export(out: Optional[Path] = typer.Option(None, "--out"), story_id: Optional[str] = StoryOpt):
    """Export the full plan + all approved episodes into one Markdown file."""
    story = _open(story_id)
    path = reports.export_markdown(story, out)
    console.print(f"Wrote {path}")


@app.command(name="list")
def list_stories():
    """List stories."""
    s = load_settings()
    t = Table("ID", "Title", "Plan", "Approved", "Cost $")
    current = _current_file().read_text().strip() if _current_file().exists() else ""
    for d in sorted(s.stories_dir.glob("*/story.db")):
        story = Story.open(d.parent.name, s)
        title = (story.db.get_meta("bible") or {}).get("title", "?")
        cost = story.db.scalar("SELECT COALESCE(SUM(cost_usd),0) FROM traces")
        mark = " *" if d.parent.name == current else ""
        t.add_row(d.parent.name + mark, title, story.db.get_meta("plan_status") or "-",
                  str(len(story.db.approved_eps())), f"{cost:.3f}")
    console.print(t)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
