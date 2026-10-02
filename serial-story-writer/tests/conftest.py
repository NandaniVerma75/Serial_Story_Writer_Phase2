from __future__ import annotations

import pytest

from serial_writer.config import load_settings
from serial_writer.pipeline import AutoReviewer, Pipeline
from serial_writer.planner import generate_plan
from serial_writer.story import Story
from tests.fakes import FakeProvider


def make_story(tmp_path, provider: FakeProvider | None = None, story_id: str = "t", **overrides) -> Story:
    settings = load_settings(provider="fake", model_writer="fake", model_cheap="fake", embeddings="local",
                             stories_dir=tmp_path, **overrides)
    return Story.open(story_id, settings, provider=provider or FakeProvider(), create=True)


def reopen(story: Story, provider: FakeProvider) -> Story:
    """Simulate a fresh process on the same story folder."""
    story.db.close()
    return Story.open(story.id, story.settings, provider=provider)


@pytest.fixture
def planned_story(tmp_path):
    story = make_story(tmp_path)
    generate_plan(story, premise="A courier finds a ledger.")
    story.db.set_meta("plan_status", "approved")
    return story


def write_n(story: Story, n: int) -> None:
    pipe = Pipeline(story, AutoReviewer(on_fail="approve"))
    for _ in range(n):
        assert pipe.run_episode(pipe.next_episode()) == "approved"
