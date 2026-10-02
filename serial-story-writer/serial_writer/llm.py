"""Thin provider wrapper. Every LLM call in the system goes through :class:`LLM`.

Responsibilities: provider selection (Anthropic / OpenAI), model tiers
("writer" for drafting/planning, "cheap" for checks/extraction/summaries),
retries with exponential backoff, JSON output validated by Pydantic (one
corrective retry), token + cost accounting, latency timing, budget enforcement,
and a trace row per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from .config import CACHE_READ_MULT, CACHE_WRITE_MULT, Settings, price_for
from .tracing import Tracer

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A call failed permanently (after retries) or returned unusable output."""


class BudgetExceeded(LLMError):
    """The whole-run budget cap was hit. The run pauses; state is already saved."""


@dataclass
class RawCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    retries: int


class Provider(Protocol):
    name: str
    retryable: tuple[type[BaseException], ...]

    def complete(self, *, system: str, user: str, model: str, max_tokens: int,
                 json_mode: bool, effort: str | None, step: str) -> RawCompletion: ...

    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, fallbacks: bool = True):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0, timeout=600)
        self.fallbacks = fallbacks
        self.retryable = (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )

    @staticmethod
    def _supports_effort(model: str) -> bool:
        return not model.startswith("claude-haiku")

    @staticmethod
    def _supports_fallbacks(model: str) -> bool:
        return model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5"))

    def complete(self, *, system, user, model, max_tokens, json_mode, effort, step) -> RawCompletion:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            # The system prompt carries the story bible: stable per story, so cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if effort and self._supports_effort(model):
            kwargs["output_config"] = {"effort": effort}
        if self.fallbacks and self._supports_fallbacks(model):
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as exc:
            if "fallback" in str(exc).lower() and self.fallbacks:
                # Account/platform without server-side fallbacks: switch it off for the session.
                self.fallbacks = False
                kwargs.pop("extra_headers", None)
                kwargs.pop("extra_body", None)
                resp = self.client.messages.create(**kwargs)
            else:
                raise
        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused at step {step}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        return RawCompletion(
            text=text,
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    def embed(self, texts):  # Anthropic has no embeddings endpoint.
        return None


class OpenAIProvider:
    name = "openai"

    def __init__(self):
        import openai

        self.client = openai.OpenAI(max_retries=0, timeout=600)
        self.retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )

    def complete(self, *, system, user, model, max_tokens, json_mode, effort, step) -> RawCompletion:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return RawCompletion(
            text=resp.choices[0].message.content or "",
            input_tokens=(u.prompt_tokens or 0) - cached,
            output_tokens=u.completion_tokens or 0,
            cache_read_tokens=cached,
        )

    def embed(self, texts):
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]


def make_provider(settings: Settings) -> Provider:
    if settings.provider == "anthropic":
        return AnthropicProvider(fallbacks=settings.anthropic_fallbacks)
    if settings.provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM_PROVIDER {settings.provider!r} (use anthropic or openai)")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_STOP = set(
    "a an the and or but of to in on at for with by from as is was were be been it its he she they "
    "his her their them him i you we our this that these those not no so then than there here into "
    "out up down over under again just only very had has have do did does will would could should".split()
)


def local_embed(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic hashed bag of unigrams + bigrams (no API needed).

    Weaker than a learned embedding for paraphrase, but good at catching the
    failure we care about: an episode whose events repeat an earlier one.
    """
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP and len(w) > 2]
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    vec = np.zeros(dim, dtype=np.float32)
    for g in grams:
        h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _strip_titles(obj):
    if isinstance(obj, dict):
        return {k: _strip_titles(v) for k, v in obj.items() if k != "title" or not isinstance(v, str)}
    if isinstance(obj, list):
        return [_strip_titles(v) for v in obj]
    return obj


def schema_instructions(schema: type[BaseModel]) -> str:
    js = json.dumps(_strip_titles(schema.model_json_schema()), separators=(",", ":"))
    return (
        "Respond with ONE JSON object only - no prose before or after, no code fences. "
        f"It must validate against this JSON Schema:\n{js}"
    )


def parse_json_text(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    return json.loads(t[start : end + 1])


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------
class LLM:
    """Single entry point for model calls, with accounting and tracing."""

    def __init__(self, settings: Settings, tracer: Tracer, provider: Provider | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.tracer = tracer
        self.provider = provider or make_provider(settings)
        self._sleep = sleep
        self.run_cost = 0.0

    # -- accounting ----------------------------------------------------------
    @staticmethod
    def cost(model: str, raw: RawCompletion) -> float:
        pin, pout = price_for(model)
        return (
            raw.input_tokens * pin
            + raw.cache_read_tokens * pin * CACHE_READ_MULT
            + raw.cache_write_tokens * pin * CACHE_WRITE_MULT
            + raw.output_tokens * pout
        ) / 1_000_000

    def _model(self, tier: str) -> tuple[str, str, int]:
        s = self.settings
        if tier == "writer":
            return s.model_writer, s.effort_writer, s.max_tokens_writer
        return s.model_cheap, s.effort_cheap, s.max_tokens_cheap

    # -- calls ---------------------------------------------------------------
    def complete(self, *, step: str, episode: int, system: str, user: str, tier: str = "writer",
                 json_mode: bool = False, max_tokens: int | None = None, decision: str = "") -> LLMResult:
        if self.run_cost >= self.settings.run_budget_usd:
            self.tracer.decision(episode, step, f"run budget cap hit (${self.run_cost:.2f})")
            raise BudgetExceeded(
                f"Run budget ${self.settings.run_budget_usd:.2f} reached (spent ${self.run_cost:.2f})."
            )
        model, effort, default_max = self._model(tier)
        retries = 0
        t0 = time.perf_counter()
        while True:
            try:
                raw = self.provider.complete(system=system, user=user, model=model,
                                             max_tokens=max_tokens or default_max,
                                             json_mode=json_mode, effort=effort, step=step)
                break
            except self.provider.retryable as exc:
                if retries >= self.settings.llm_max_retries:
                    self.tracer.log(episode=episode, step=step, model=model, retry_count=retries,
                                    decision=f"failed: {type(exc).__name__}")
                    raise LLMError(f"{step}: {exc}") from exc
                retries += 1
                self._sleep(min(30.0, 2 ** retries + random.random()))
        latency = int((time.perf_counter() - t0) * 1000)
        cost = self.cost(model, raw)
        self.run_cost += cost
        self.tracer.log(
            episode=episode, step=step, model=model, input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens, cache_read_tokens=raw.cache_read_tokens,
            cache_write_tokens=raw.cache_write_tokens, cost_usd=cost, latency_ms=latency,
            retry_count=retries, decision=decision,
        )
        return LLMResult(raw.text, model, raw.input_tokens, raw.output_tokens, cost, latency, retries)

    def complete_json(self, schema: type[T], *, step: str, episode: int, system: str, user: str,
                      tier: str = "cheap", max_tokens: int | None = None,
                      validate: Callable[[T], None] | None = None) -> T:
        """Call the model, parse + validate into `schema`; retry once with the error on failure.

        `validate` may raise ValueError for semantic checks (e.g. "exactly 20 beats").
        """
        prompt = f"{user}\n\n{schema_instructions(schema)}"
        res = self.complete(step=step, episode=episode, system=system, user=prompt, tier=tier,
                            json_mode=True, max_tokens=max_tokens)
        try:
            obj = schema.model_validate(parse_json_text(res.text))
            if validate:
                validate(obj)
            return obj
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)[:1500]
        self.tracer.decision(episode, step, f"parse/validate failed, retrying once: {error[:200]}")
        retry_prompt = (
            f"{prompt}\n\nYour previous reply was rejected:\n{error}\n\n"
            f"Previous reply (for reference):\n{res.text[:4000]}\n\nReturn the corrected JSON object only."
        )
        res = self.complete(step=f"{step}:retry", episode=episode, system=system, user=retry_prompt,
                            tier=tier, json_mode=True, max_tokens=max_tokens)
        try:
            obj = schema.model_validate(parse_json_text(res.text))
            if validate:
                validate(obj)
            return obj
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(f"{step}: invalid structured output after retry: {str(exc)[:300]}") from exc

    # -- embeddings ------------------------------------------------------------
    def embed(self, text: str, *, episode: int, step: str = "embed") -> np.ndarray:
        mode = self.settings.embeddings
        use_openai = mode == "openai" or (mode == "auto" and bool(os.getenv("OPENAI_API_KEY"))
                                          and self.settings.provider in ("openai", "anthropic"))
        if use_openai:
            try:
                provider = self.provider if isinstance(self.provider, OpenAIProvider) else OpenAIProvider()
                t0 = time.perf_counter()
                vec = provider.embed([text])[0]
                tokens = len(text) // 4
                cost = tokens * price_for("text-embedding-3-small")[0] / 1_000_000
                self.run_cost += cost
                self.tracer.log(episode=episode, step=step, model="text-embedding-3-small",
                                input_tokens=tokens, cost_usd=cost,
                                latency_ms=int((time.perf_counter() - t0) * 1000))
                return np.asarray(vec, dtype=np.float32)
            except Exception as exc:  # fall back rather than fail an episode on embeddings
                self.tracer.decision(episode, step, f"openai embedding failed, using local: {exc}")
        return local_embed(text)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for context budgeting."""
    return len(text) // 4 + 1
