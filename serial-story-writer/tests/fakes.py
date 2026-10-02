"""A deterministic fake LLM provider. Used ONLY in tests."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

from serial_writer.llm import RawCompletion

WORDS = ("lantern harbor ledger monsoon rooftop ferry ticket orchard bruise whistle copper tram ledger "
         "spice market cinema saffron rickshaw locket compass pigeon rain tunnel kiln mirror violin "
         "kite festival bridge courtroom tailor clinic garage parcel signal radio quarry").split()


def _ep(user: str, pattern: str) -> int:
    m = re.search(pattern, user)
    return int(m.group(1)) if m else 1


def _words_for(ep: int, n: int) -> list[str]:
    out = []
    for i in range(n):
        h = int(hashlib.md5(f"{ep}-{i}".encode()).hexdigest(), 16)
        out.append(WORDS[h % len(WORDS)])
    return out


def bible(_: str) -> str:
    return json.dumps({
        "title": "The Monsoon Ledger", "premise": "A courier finds a ledger.", "genre": "mystery",
        "tone": "wry, grounded", "pov": "close third, rotating", "tense": "past",
        "setting": "Mumbai, 1990s",
        "style_rules": ["short sentences in action", "smell before sight"],
        "world_rules": ["no mobile phones", "the ledger cannot be copied"],
        "banned_phrases": ["dark and stormy"],
        "characters": [
            {"name": "Ravi Kumar", "role": "courier", "traits": "restless", "voice_notes": "slang",
             "arc": "from runner to witness", "relationships": [{"other": "Meera", "type": "friend", "state": "close"}]},
            {"name": "Meera Shah", "role": "clerk", "traits": "precise", "voice_notes": "formal",
             "arc": "from loyal to defiant", "relationships": [{"other": "Ravi Kumar", "type": "friend", "state": "close"}]},
            {"name": "Kabir Rao", "role": "inspector", "traits": "tired", "voice_notes": "clipped",
             "arc": "from cynic to ally", "relationships": []},
        ],
    })


def acts(_: str) -> str:
    return json.dumps({"acts": [
        {"act_no": i, "title": f"Act {i}", "summary": f"Things escalate {i}.", "turning_point": f"TP {i}",
         "character_arcs": ["Ravi: changes", "Meera: changes", "Kabir: changes"]} for i in range(1, 6)]})


def arcs(_: str) -> str:
    return json.dumps({"arcs": [
        {"arc_no": i, "act_no": (i + 1) // 2, "title": f"Arc {i}", "goal": f"Goal {i}",
         "threads_opened": [{"description": f"Mystery {i}", "payoff_arc": min(10, i + 2)}]} for i in range(1, 11)]})


def beats(user: str) -> str:
    start = _ep(user, r"episodes (\d+) to")
    return json.dumps({"beats": [{"ep": start + i, "beat": f"Ravi and Meera chase clue {start + i}."}
                                 for i in range(20)]})


def scene_plan(user: str) -> str:
    ep = _ep(user, r"Plan episode (\d+)")
    hooks = ["cliffhanger", "revelation", "decision", "threat", "mystery"]
    return json.dumps({
        "title": f"Clue {ep}", "in_story_day": ep, "time_label": f"Day {ep}, night",
        "characters": ["Ravi Kumar", "Meera Shah"], "scenes": [f"Ravi wants clue {ep}"],
        "goal": f"find clue {ep}", "conflict": "Kabir blocks", "hook_type": hooks[ep % len(hooks)],
        "hook_idea": "a knock at the door",
    })


def prose(user: str) -> str:
    ep = _ep(user, r"(?:Write|Revise) episode (\d+)")
    body = " ".join(_words_for(ep, 480))
    return f"# Clue {ep}\n\nRavi Kumar ran. Meera Shah waited. {body}"


def consistency(_: str) -> str:
    return json.dumps({"contradictions": []})


def rubric(user: str) -> str:
    ep = _ep(user, r"Score this draft of episode (\d+)")
    ids = [int(x) for x in re.findall(r"\[D(\d+)\]", user)]
    return json.dumps({
        "hook_strength": 4, "momentum": 4, "voice": 4, "beat_adherence": 4, "directive_adherence": 4,
        "directive_scores": [{"directive_id": i, "score": 4, "note": "followed"} for i in ids],
        "summary": " ".join(_words_for(ep * 7919, 40)), "notes": "none",
    })


def extraction(user: str) -> str:
    ep = _ep(user, r"Episode (\d+) has been approved")
    edited = "EDITED" in user
    tag = f"clue{ep}{'b' if edited else ''}"
    return json.dumps({
        "summary": f"Episode {ep}: " + " ".join(_words_for(ep * 104729, 40)),
        "hook_type": "mystery", "in_story_day": ep, "time_label": f"Day {ep}",
        "characters_present": ["Ravi Kumar", "Meera Shah"],
        "facts": [{"subject": tag, "predicate": "is hidden in", "object": f"locker {ep}"}],
        "threads_opened": [{"description": f"Who hid {tag}?", "expected_payoff_arc": 2}],
        "threads_touched": [], "threads_resolved": [],
    })


def classify(user: str) -> str:
    ep = _ep(user, r"at episode (\d+)")
    said = re.search(r'^"(.*)"$', user, re.MULTILINE)
    if said and "kill" in said.group(1).lower():
        return json.dumps({"kind": "character_fate", "scope": "character", "scope_target": "Meera Shah",
                           "needs_replan": True, "normalized_text": "Meera Shah dies on-page.",
                           "fates": [{"name": "Meera", "status": "dead", "by_episode": ep + 2}]})
    return json.dumps({"kind": "pacing", "scope": "global", "needs_replan": False,
                       "normalized_text": "Slow down the romance between Ravi and Meera."})


def replan(user: str) -> str:
    start = _ep(user, r"episodes (\d+)-")
    return json.dumps({"changed_beats": [{"ep": start + 1, "beat": "Meera is killed in the tram depot."}],
                       "rationale": "directive"})


DEFAULTS: dict[str, Callable[[str], str]] = {
    "plan:bible": bible, "plan:acts": acts, "plan:arcs": arcs, "plan:beats": beats,
    "refine_beat": scene_plan, "draft": prose, "revise": prose, "check:consistency": consistency,
    "check:rubric": rubric, "extract": extraction, "summary": lambda u: "A summary of events.",
    "directive:classify": classify, "directive:replan": replan,
}


class FakeProvider:
    name = "fake"
    retryable = (TimeoutError,)

    def __init__(self, overrides: dict[str, Callable[[str], str]] | None = None, fail_on: set[str] | None = None):
        self.overrides = overrides or {}
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def complete(self, *, system, user, model, max_tokens, json_mode, effort, step) -> RawCompletion:
        base = step.split(":retry")[0]
        self.calls.append(base)
        if base in self.fail_on:
            raise RuntimeError(f"simulated crash at {base}")
        handler = self.overrides.get(base)
        if handler is None:
            handler = next(h for prefix, h in DEFAULTS.items() if base.startswith(prefix))
        text = handler(user)
        return RawCompletion(text=text, input_tokens=len(user) // 4, output_tokens=len(text) // 4)

    def embed(self, texts):
        return None
