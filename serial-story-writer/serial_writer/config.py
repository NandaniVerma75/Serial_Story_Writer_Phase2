"""Configuration: provider/model selection, price table, and every limit the system enforces.

All knobs live here (overridable through environment variables / `.env`) so that
cost caps, revision limits and budgets are visible in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"

load_dotenv(PROJECT_ROOT / ".env")

# USD per 1M tokens: (input, output). Verify against the provider's pricing page
# before relying on absolute numbers; the relative numbers drive the estimates.
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    # Fake model used by tests
    "fake": (0.0, 0.0),
}
# Prompt-cache pricing multipliers relative to the input price.
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25

DEFAULT_MODELS = {
    "anthropic": ("claude-opus-5-5", "claude-haiku-4-5"),
    "openai": ("gpt-4.1", "gpt-4.1-mini"),
    "fake": ("fake", "fake"),
}

# Generic LLM tics banned in every story (merged with the bible's own list).
BANNED_TICS = [
    "a sense of", "little did they know", "the air was thick with", "a testament to",
    "couldn't help but", "sent shivers down", "in that moment", "it was as if",
    "a wave of", "the weight of", "palpable", "tapestry", "delve", "a mix of",
    "for what felt like an eternity", "a breath she didn't know", "a breath he didn't know",
    "something shifted", "the silence was deafening", "every fiber of",
    "eyes widened", "heart pounded in", "a flicker of", "steeled herself", "steeled himself",
]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    """Runtime settings. Construct with :func:`load_settings`."""

    provider: str = "anthropic"
    model_writer: str = "claude-opus-5-5"
    model_cheap: str = "claude-haiku-4-5"
    effort_writer: str = "medium"
    effort_cheap: str = "low"
    embeddings: str = "auto"  # auto | openai | local
    anthropic_fallbacks: bool = True
    stories_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "stories")

    # Story shape
    total_episodes: int = 200
    n_acts: int = 5
    n_arcs: int = 10

    # Context assembly
    context_budget_tokens: int = 12000

    # Episode pipeline limits
    words_min: int = 400
    words_max: int = 700
    max_revisions: int = 2
    episode_cost_cap_usd: float = 0.25
    run_budget_usd: float = 10.0
    rubric_pass_mean: float = 3.5
    rubric_min_dim: int = 3
    repetition_threshold: float = 0.88
    hook_window: int = 3
    untouched_thread_eps: int = 15
    skip_judge_if_hard_pass: bool = False
    reconcile_window: int = 20
    replan_window: int = 20

    # LLM call behaviour
    llm_max_retries: int = 3
    max_tokens_writer: int = 16000
    max_tokens_cheap: int = 8000

    # Fallback estimates for `story estimate` before anything has been measured
    default_episode_cost_usd: float = 0.15
    default_episode_seconds: float = 120.0


def load_settings(**overrides) -> Settings:
    """Read settings from the environment, then apply keyword overrides."""
    provider = _env("LLM_PROVIDER", "anthropic").lower()
    writer_default, cheap_default = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    s = Settings(
        provider=provider,
        model_writer=_env("LLM_MODEL_WRITER", writer_default),
        model_cheap=_env("LLM_MODEL_CHEAP", cheap_default),
        effort_writer=_env("LLM_EFFORT_WRITER", "medium"),
        effort_cheap=_env("LLM_EFFORT_CHEAP", "low"),
        embeddings=_env("EMBEDDINGS", "auto").lower(),
        anthropic_fallbacks=_env("ANTHROPIC_FALLBACKS", "1") not in ("0", "false", "no"),
        stories_dir=Path(_env("STORIES_DIR", str(PROJECT_ROOT / "stories"))),
        context_budget_tokens=int(_env("CONTEXT_BUDGET_TOKENS", "12000")),
        max_revisions=int(_env("MAX_REVISIONS", "2")),
        episode_cost_cap_usd=float(_env("EPISODE_COST_CAP_USD", "0.25")),
        run_budget_usd=float(_env("RUN_BUDGET_USD", "10.0")),
        rubric_pass_mean=float(_env("RUBRIC_PASS_MEAN", "3.5")),
        repetition_threshold=float(_env("REPETITION_THRESHOLD", "0.88")),
        skip_judge_if_hard_pass=_env("SKIP_JUDGE_IF_HARD_PASS", "0") in ("1", "true", "yes"),
    )
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def price_for(model: str) -> tuple[float, float]:
    """Return (input, output) USD per 1M tokens; unknown models fall back to a prefix match."""
    if model in PRICES:
        return PRICES[model]
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            return PRICES[name]
    return (0.0, 0.0)
