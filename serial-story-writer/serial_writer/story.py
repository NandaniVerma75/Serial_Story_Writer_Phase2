"""A `Story` bundles a story folder: its database, tracer, LLM wrapper and file exports."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .config import BANNED_TICS, PROMPTS_DIR, Settings, load_settings
from .db import StoryDB
from .llm import LLM, Provider
from .models import StoryBible
from .tracing import Tracer


def load_prompt(name: str, **values) -> str:
    """Render `prompts/<name>.md` with `$placeholders` (missing ones are left untouched)."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return Template(text).safe_substitute({k: str(v) for k, v in values.items()})


def slugify(text: str, n: int = 4) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:n]
    return "-".join(words) or "story"


@dataclass
class Story:
    id: str
    dir: Path
    db: StoryDB
    settings: Settings
    tracer: Tracer
    llm: LLM

    # -- construction --------------------------------------------------------
    @classmethod
    def open(cls, story_id: str, settings: Settings | None = None, provider: Provider | None = None,
             create: bool = False) -> "Story":
        settings = settings or load_settings()
        d = settings.stories_dir / story_id
        if not create and not (d / "story.db").exists():
            raise FileNotFoundError(f"No story {story_id!r} in {settings.stories_dir}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "episodes").mkdir(exist_ok=True)
        db = StoryDB(d / "story.db")
        tracer = Tracer(db, d / "trace.jsonl")
        llm = LLM(settings, tracer, provider=provider)
        return cls(story_id, d, db, settings, tracer, llm)

    @classmethod
    def new_id(cls, premise: str) -> str:
        return f"{slugify(premise)}-{uuid.uuid4().hex[:4]}"

    # -- bible -----------------------------------------------------------------
    @property
    def bible(self) -> StoryBible:
        data = self.db.get_meta("bible")
        if data is None:
            raise RuntimeError("Story has no bible yet")
        return StoryBible.model_validate(data)

    def banned_phrases(self) -> list[str]:
        seen, out = set(), []
        for p in [*BANNED_TICS, *self.bible.banned_phrases]:
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
        return out

    def render_bible(self) -> str:
        b = self.bible
        lines = [
            f"Title: {b.title}", f"Premise: {b.premise}", f"Genre: {b.genre}", f"Tone: {b.tone}",
            f"POV: {b.pov}", f"Tense: {b.tense}", f"Setting: {b.setting}",
            "Style rules:", *[f"- {r}" for r in b.style_rules],
            "World rules (never break these):", *[f"- {r}" for r in b.world_rules],
            "Banned phrases/tics: " + "; ".join(self.banned_phrases()),
        ]
        return "\n".join(lines)

    def writer_system(self) -> str:
        """Stable system prompt for the writer model (bible inside → cacheable prefix)."""
        return load_prompt("writer_system", bible=self.render_bible(),
                           total=self.settings.total_episodes)

    def checker_system(self) -> str:
        return load_prompt("checker_system", bible=self.render_bible())

    # -- exports ----------------------------------------------------------------
    def episode_path(self, ep: int) -> Path:
        return self.dir / "episodes" / f"ep_{ep:03d}.md"

    def write_episode_md(self, ep: int) -> None:
        row = self.db.get_episode(ep)
        if row is None:
            return
        beat = self.db.beat(ep)
        body = f"# Episode {ep}: {row['title'] or ''}\n\n_Beat: {beat}_\n\n{row['text']}\n"
        self.episode_path(ep).write_text(body, encoding="utf-8")
