"""Human-in-the-loop terminal UI: plan approval, the per-episode review menu, memory views."""
from __future__ import annotations

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .checker import CheckResult
from .directives import BeatChange, handle_feedback
from .memory import active_directives, characters_view, directive_line, fact_line, facts_view, threads_view
from .pipeline import ReviewDecision
from .planner import PlanError, export_plan_yaml, generate_plan, import_plan_yaml
from .story import Story

console = Console()


# ---------------------------------------------------------------------------
# Plan display + approval
# ---------------------------------------------------------------------------
def show_plan_overview(story: Story) -> None:
    db = story.db
    b = story.bible
    console.print(Panel(f"[bold]{b.title}[/]\n{b.premise}\n\n[dim]{b.genre} · {b.tone} · {b.pov} · {b.tense}[/]",
                        title="Story bible"))
    t = Table("Character", "Role", "Arc", title="Main characters", show_lines=False)
    for c in b.characters:
        t.add_row(c.name, c.role, c.arc)
    console.print(t)
    t = Table("Act", "Title", "Summary", "Turning point", show_lines=True)
    for a in db.query("SELECT * FROM acts ORDER BY act_no"):
        t.add_row(str(a["act_no"]), a["title"], a["summary"], a["turning_point"])
    console.print(t)
    t = Table("Arc", "Eps", "Title", "Goal", "Threads opened (→ payoff arc)", show_lines=True)
    for r in db.query("SELECT * FROM arcs ORDER BY arc_no"):
        threads = "; ".join(f"{p['description']} (→{p['payoff_arc']})" for p in db.query(
            "SELECT * FROM plan_threads WHERE opened_arc=?", (r["arc_no"],)))
        t.add_row(str(r["arc_no"]), f"{r['start_ep']}-{r['end_ep']}", r["title"], r["goal"], threads)
    console.print(t)


def show_arc_beats(story: Story, arc_no: int) -> None:
    arc = story.db.one("SELECT * FROM arcs WHERE arc_no=?", (arc_no,))
    if arc is None:
        return
    t = Table("Ep", "Beat", title=f"Arc {arc_no}: {arc['title']}  (eps {arc['start_ep']}-{arc['end_ep']})")
    for r in story.db.query("SELECT ep, beat FROM beats WHERE ep BETWEEN ? AND ? ORDER BY ep",
                            (arc["start_ep"], arc["end_ep"])):
        t.add_row(str(r["ep"]), r["beat"])
    console.print(t)


def plan_approval(story: Story) -> bool:
    """Page through the plan; approve, edit YAML in $EDITOR, or give feedback (re-plan)."""
    page = 0
    n_arcs = story.settings.n_arcs
    while True:
        console.rule(f"Plan review — page {page}/{n_arcs} (0 = overview)")
        if page == 0:
            show_plan_overview(story)
        else:
            show_arc_beats(story, page)
        for w in story.db.get_meta("plan_warnings") or []:
            console.print(f"[yellow]⚠ {w}[/]")
        choice = Prompt.ask("[a]pprove  [n]ext  [b]ack  [g]o to arc  [e]dit YAML  [f]eedback → re-plan  [q]uit",
                            choices=["a", "n", "b", "g", "e", "f", "q"], default="n")
        if choice == "a":
            story.db.set_meta("plan_status", "approved")
            export_plan_yaml(story)
            story.tracer.decision(0, "plan", "human approved plan")
            console.print("[green]Plan approved.[/]")
            return True
        if choice == "n":
            page = min(n_arcs, page + 1)
        elif choice == "b":
            page = max(0, page - 1)
        elif choice == "g":
            page = int(Prompt.ask("Arc number", default="1"))
        elif choice == "e":
            path = export_plan_yaml(story)
            edited = click.edit(path.read_text(encoding="utf-8"), extension=".yaml", require_save=True)
            if edited is None:
                console.print("[dim]No changes.[/]")
                continue
            try:
                warnings = import_plan_yaml(story, edited)
                story.db.set_meta("plan_warnings", warnings)
                story.tracer.decision(0, "plan", "human edited plan YAML (validated)")
                console.print("[green]Edited plan validated and saved.[/]")
            except PlanError as exc:
                console.print(f"[red]Edit rejected: {exc}[/]")
        elif choice == "f":
            fb = Prompt.ask("Feedback on the plan (e.g. 'make act 3 darker', 'add a rival rider')")
            replan_with_feedback(story, fb)
            page = 0
        elif choice == "q":
            console.print("Plan saved as pending. Resume with: story resume " + story.id)
            return False


def replan_with_feedback(story: Story, feedback: str) -> None:
    feedback_list = (story.db.get_meta("plan_feedback") or []) + [feedback]
    story.db.set_meta("plan_feedback", feedback_list)
    story.tracer.decision(0, "plan", f"plan feedback: {feedback}")
    with console.status("Re-planning with feedback…") as status:
        warnings = generate_plan(story, feedback=feedback_list, progress=lambda m: status.update(m))
    story.db.set_meta("plan_warnings", warnings)


# ---------------------------------------------------------------------------
# Beat diff approval (directive re-planning)
# ---------------------------------------------------------------------------
def show_beat_diff(changes: list[BeatChange], rationale: str) -> None:
    t = Table("Ep", "Before", "After", title="Proposed beat changes", show_lines=True)
    for c in changes:
        t.add_row(str(c.ep), f"[red]{c.old}[/]", f"[green]{c.new}[/]")
    console.print(t)
    if rationale:
        console.print(f"[dim]Rationale: {rationale}[/]")


def approve_beat_diff(changes: list[BeatChange], rationale: str) -> bool:
    show_beat_diff(changes, rationale)
    return Confirm.ask("Apply these beat changes?", default=True)


def auto_approve_beat_diff(changes: list[BeatChange], rationale: str) -> bool:
    show_beat_diff(changes, rationale)
    console.print("[dim](--yes: applying automatically)[/]")
    return True


def feedback_flow(story: Story, text: str, at_ep: int, auto_yes: bool = False):
    console.print("[dim]Classifying feedback and re-planning affected beats…[/]")
    outcome = handle_feedback(story, text, at_ep, auto_approve_beat_diff if auto_yes else approve_beat_diff)
    c = outcome.classification
    console.print(Panel(
        f"Stored as [bold]D{outcome.directive_id}[/] — kind: {c.kind}, scope: {c.scope}"
        f"{' ' + str(c.scope_target) if c.scope_target else ''}"
        f"{' until ep ' + str(c.until_episode) if c.until_episode else ''}\n{c.normalized_text}\n"
        + "".join(f"\nScheduled fate: {f.name} → {f.status} by ep {f.by_episode}" for f in c.fates)
        + (f"\n{len(outcome.changes)} beats re-planned ({'applied' if outcome.applied else 'not applied'})"
           if outcome.changes else "\nNo beat changes needed."),
        title="Directive saved"))
    return outcome


# ---------------------------------------------------------------------------
# Memory views
# ---------------------------------------------------------------------------
def show_characters(story: Story, before_ep: int) -> None:
    t = Table("Name", "Role", "Status", "Last seen", "Relationships", title="Characters")
    for c in characters_view(story.db, before_ep).values():
        rels = "; ".join(f"{o}: {typ}{' (' + st + ')' if st else ''}" for o, (typ, st, _) in c.relationships.items())
        style = "red" if c.status != "alive" else ""
        t.add_row(c.name, c.role, f"[{style}]{c.status}[/]" if style else c.status,
                  str(c.last_seen_ep or "-"), rels)
    console.print(t)


def show_threads(story: Story, before_ep: int) -> None:
    t = Table("ID", "Status", "Opened", "Last touched", "Payoff arc", "Thread", title="Open threads")
    for th in threads_view(story.db, before_ep):
        t.add_row(f"T{th.id}", th.status, str(th.opened_ep), str(th.last_touched_ep),
                  str(th.expected_payoff_arc or "-"), th.description)
    console.print(t)


def show_facts(story: Story, before_ep: int, subject: str | None = None, limit: int = 60) -> None:
    t = Table("Fact (with provenance)", title="Fact ledger (current, newest first)")
    n = 0
    for f in facts_view(story.db, before_ep):
        if subject and subject.lower() not in f["subject"].lower():
            continue
        t.add_row(fact_line(f))
        n += 1
        if n >= limit:
            break
    console.print(t)


def show_directives(story: Story, ep: int) -> None:
    t = Table("ID", "Kind", "Scope", "Since ep", "Active now", "Directive", title="Directives")
    active = {d["id"] for d in active_directives(story.db, ep)}
    for d in story.db.query("SELECT * FROM directives ORDER BY id"):
        t.add_row(f"D{d['id']}", d["kind"], d["scope"] + (f" {d['scope_target']}" if d["scope_target"] else "")
                  + (f" ≤{d['until_ep']}" if d["until_ep"] else ""), str(d["created_at_ep"]),
                  "yes" if d["id"] in active else "no", d["text"])
    console.print(t)
    fates = story.db.query("SELECT * FROM fates ORDER BY by_ep")
    if fates:
        ft = Table("Character", "Becomes", "By ep", "Directive", "Fulfilled in", title="Scheduled fates")
        for f in fates:
            ft.add_row(f["name"], f["status"], str(f["by_ep"]), f"D{f['directive_id']}", str(f["fulfilled_ep"] or "-"))
        console.print(ft)


def show_memory(story: Story, ep: int) -> None:
    show_characters(story, ep)
    show_threads(story, ep)
    show_directives(story, ep)
    show_facts(story, ep, limit=30)


# ---------------------------------------------------------------------------
# Episode review
# ---------------------------------------------------------------------------
def show_checks(checks: CheckResult | None) -> None:
    if checks is None:
        console.print("[dim]No checks recorded.[/]")
        return
    color = "green" if checks.passed else "red"
    head = (f"[{color}]{'PASSED' if checks.passed else 'FAILED'}[/] · {checks.word_count} words · "
            f"rubric mean {checks.rubric_mean} · max similarity {checks.max_similarity}"
            f"{' (ep ' + str(checks.most_similar_ep) + ')' if checks.most_similar_ep else ''}")
    console.print(head)
    if checks.rubric:
        r = checks.rubric
        console.print("  " + " · ".join(f"{k} {r[k]}" for k in
                                        ("hook_strength", "momentum", "voice", "beat_adherence", "directive_adherence")))
        for d in r.get("directive_scores", []):
            console.print(f"  D{d['directive_id']}: {d['score']}/5 {d.get('note', '')}")
        if r.get("notes"):
            console.print(f"  [dim]Judge: {r['notes']}[/]")
    for i in checks.issues:
        console.print(f"  [{'red' if i.severity == 'error' else 'yellow'}]{i.severity} [{i.kind}][/] {i.message}")


def show_episode(story: Story, ep: int, checks: CheckResult | None = None, full: bool = True) -> None:
    row = story.db.get_episode(ep)
    if row is None:
        console.print(f"Episode {ep} does not exist.")
        return
    console.print(Panel(row["text"] if full else (row["summary"] or ""),
                        title=f"Episode {ep}: {row['title']}  [{row['status']}]",
                        subtitle=f"Beat: {story.db.beat(ep)[:100]}"))
    dirs = json.loads(row["directives"] or "[]")
    console.print(f"Directives in context: {', '.join('D' + str(d) for d in dirs) or 'none'} · "
                  f"revisions: {row['revisions']} · cost so far: ${_ep_cost(story, ep):.4f}")
    show_checks(checks if checks is not None else CheckResult.from_json(row["checks"]))


def _ep_cost(story: Story, ep: int) -> float:
    return float(story.db.scalar("SELECT COALESCE(SUM(cost_usd),0) FROM traces WHERE episode=?", (ep,)))


class InteractiveReviewer:
    """The per-episode review menu."""

    def review(self, story: Story, ep: int, checks: CheckResult | None) -> ReviewDecision:
        while True:
            console.rule(f"Review episode {ep}")
            show_episode(story, ep, checks)
            choice = Prompt.ask(
                "[a]pprove  [e]dit  [r]eject+regenerate  [f]eedback/directive  [s]how memory  [p]ause",
                choices=["a", "e", "r", "f", "s", "p"], default="a")
            if choice == "a":
                story.tracer.decision(ep, "review", "human approved")
                return ReviewDecision("approve")
            if choice == "e":
                text = story.db.get_episode(ep)["text"]
                edited = click.edit(text, extension=".md", require_save=True)
                if edited and edited.strip() != text.strip():
                    return ReviewDecision("edit", edited.strip())
                console.print("[dim]No changes.[/]")
            elif choice == "r":
                reason = Prompt.ask("Why reject? (used to regenerate this episode)")
                return ReviewDecision("reject", reason)
            elif choice == "f":
                text = Prompt.ask("Feedback (e.g. 'slow down the romance', 'kill off Meera by ep 11')")
                feedback_flow(story, text, ep)
                if Confirm.ask("Regenerate this episode with the new directive?", default=True):
                    return ReviewDecision("regenerate", f"feedback: {text}")
            elif choice == "s":
                show_memory(story, ep)
            elif choice == "p":
                return ReviewDecision("pause")
