"""Pydantic models for every structured LLM output.

Models are deliberately lenient about *shape* (defaults for optional lists,
normalised enums) and strict about *meaning* (scores clamped to 1-5, required
core fields). A parse failure triggers exactly one corrective retry in `llm.py`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

HOOK_TYPES = [
    "cliffhanger", "revelation", "decision", "threat", "mystery",
    "emotional_turn", "reversal", "arrival",
]


def _norm_token(v: str) -> str:
    return (v or "").strip().lower().replace("-", "_").replace(" ", "_")


# --- Planning ---------------------------------------------------------------
class RelationshipSeed(BaseModel):
    other: str
    type: str
    state: str = ""


class CharacterSeed(BaseModel):
    name: str
    role: str = ""
    traits: str = ""
    voice_notes: str = ""
    arc: str = ""
    relationships: list[RelationshipSeed] = Field(default_factory=list)


class StoryBible(BaseModel):
    title: str
    premise: str
    genre: str
    tone: str
    pov: str
    tense: str
    setting: str
    style_rules: list[str] = Field(default_factory=list)
    world_rules: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)
    characters: list[CharacterSeed] = Field(min_length=3)


class Act(BaseModel):
    act_no: int
    title: str
    summary: str
    turning_point: str
    character_arcs: list[str] = Field(default_factory=list)


class ActPlan(BaseModel):
    acts: list[Act]


class PlannedThread(BaseModel):
    description: str
    payoff_arc: int


class Arc(BaseModel):
    arc_no: int
    act_no: int
    title: str
    goal: str
    threads_opened: list[PlannedThread] = Field(default_factory=list)


class ArcPlan(BaseModel):
    arcs: list[Arc]


class Beat(BaseModel):
    ep: int
    beat: str


class BeatList(BaseModel):
    beats: list[Beat]


# --- Episode pipeline ----------------------------------------------------------
class ScenePlan(BaseModel):
    title: str
    in_story_day: int = Field(description="Story day number (Day 1 = first day of the story)")
    time_label: str = ""
    is_flashback: bool = False
    characters: list[str] = Field(description="Characters physically present / acting in this episode")
    new_characters: list[str] = Field(default_factory=list, description="Characters appearing for the first time")
    scenes: list[str]
    goal: str
    conflict: str
    threads_to_touch: list[str] = Field(default_factory=list)
    hook_type: str
    hook_idea: str

    @field_validator("hook_type")
    @classmethod
    def _hook(cls, v: str) -> str:
        return _norm_token(v)


class Contradiction(BaseModel):
    claim: str = Field(description="What the draft says")
    conflicts_with: str = Field(description="The canon it contradicts")
    evidence: str = Field(default="", description="Quote from the draft")
    severity: str = "medium"

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        v = _norm_token(v)
        return v if v in ("high", "medium", "low") else "medium"


class ConsistencyReport(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


class DirectiveScore(BaseModel):
    directive_id: int
    score: int
    note: str = ""

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(1, min(5, int(v)))


class RubricScore(BaseModel):
    hook_strength: int
    momentum: int
    voice: int
    beat_adherence: int
    directive_adherence: int
    directive_scores: list[DirectiveScore] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence plain summary of what happens")
    notes: str = Field(default="", description="Most important concrete weaknesses, if any")

    @field_validator("hook_strength", "momentum", "voice", "beat_adherence", "directive_adherence")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(1, min(5, int(v)))

    def dims(self) -> dict[str, int]:
        return {
            "hook_strength": self.hook_strength,
            "momentum": self.momentum,
            "voice": self.voice,
            "beat_adherence": self.beat_adherence,
            "directive_adherence": self.directive_adherence,
        }

    def mean(self) -> float:
        d = self.dims()
        return sum(d.values()) / len(d)


# --- Extraction (post-approval memory update) ------------------------------------
class NewCharacter(BaseModel):
    name: str
    role: str = ""
    traits: str = ""
    voice_notes: str = ""


class StatusChange(BaseModel):
    name: str
    status: str
    detail: str = ""

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        return _norm_token(v)


class RelationshipChange(BaseModel):
    a: str
    b: str
    type: str
    state: str = ""


class FactItem(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9


class ThreadOpened(BaseModel):
    description: str
    expected_payoff_arc: int | None = None


class Extraction(BaseModel):
    summary: str
    hook_type: str = ""
    in_story_day: int
    time_label: str = ""
    characters_present: list[str] = Field(default_factory=list)
    new_characters: list[NewCharacter] = Field(default_factory=list)
    status_changes: list[StatusChange] = Field(default_factory=list)
    relationship_changes: list[RelationshipChange] = Field(default_factory=list)
    facts: list[FactItem] = Field(default_factory=list)
    threads_opened: list[ThreadOpened] = Field(default_factory=list)
    threads_touched: list[int] = Field(default_factory=list)
    threads_resolved: list[int] = Field(default_factory=list)

    @field_validator("hook_type")
    @classmethod
    def _hook(cls, v: str) -> str:
        return _norm_token(v)


# --- Directives -------------------------------------------------------------------
class FateEffect(BaseModel):
    name: str
    status: str
    by_episode: int | None = None

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        return _norm_token(v)


class DirectiveClassification(BaseModel):
    kind: str = Field(description="style | pacing | plot | character_fate")
    scope: str = Field(description="global | character | until_episode | arc")
    scope_target: str | None = None
    until_episode: int | None = None
    needs_replan: bool
    normalized_text: str = Field(description="The directive rewritten as a clear, standing instruction to the writer")
    fates: list[FateEffect] = Field(default_factory=list)

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        v = _norm_token(v)
        return v if v in ("style", "pacing", "plot", "character_fate") else "plot"

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: str) -> str:
        v = _norm_token(v)
        return v if v in ("global", "character", "until_episode", "arc") else "global"


class Replan(BaseModel):
    changed_beats: list[Beat] = Field(default_factory=list)
    rationale: str = ""
