"""Episodic task memory models for trajectory caching."""

from __future__ import annotations

from pydantic import Field

from src.models.common import StrictModel
from src.models.perception import PageHint
from src.models.policy import ActionType


class EpisodeStep(StrictModel):
    """One compressed step in a reusable episode trajectory."""

    step_index: int = Field(ge=1)
    page_hint: PageHint
    action_type: ActionType
    target_description: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=500)
    key: str | None = Field(default=None, max_length=100)
    subgoal: str = Field(min_length=1, max_length=200)


class Episode(StrictModel):
    """A compressed, reusable action trajectory from a successful run."""

    episode_id: str = Field(min_length=1)
    normalized_intent: str = Field(min_length=1, max_length=500)
    benchmark: str = Field(min_length=1, max_length=100)
    source_run_id: str = Field(min_length=1)
    steps: list[EpisodeStep] = Field(min_length=1)
    success_count: int = Field(default=1, ge=1)
    created_at: str = Field(min_length=1)


class EpisodeReplayState(StrictModel):
    """Tracks progress through an active episode replay."""

    episode_id: str = Field(min_length=1)
    current_step_index: int = Field(default=0, ge=0)
    active: bool = True
    deviations: int = Field(default=0, ge=0)
    max_deviations: int = Field(default=2, ge=1)
