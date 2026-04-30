"""Compact advisory memory models for local benchmark learning."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StrictModel
from src.models.perception import PageHint, UIElement
from src.models.policy import ActionType


class MemoryOutcome(StrEnum):
    """Outcome labels for compact memory records."""

    SUCCESS = "success"
    FAILURE = "failure"
    GUARDRAIL = "guardrail"


class MemoryRecord(StrictModel):
    """One compact local memory record."""

    key: str = Field(min_length=1, max_length=100)
    benchmark: str = Field(min_length=1, max_length=100)
    hint: str = Field(min_length=1, max_length=300)
    outcome: MemoryOutcome
    page_hint: PageHint | None = None
    subgoal: str | None = Field(default=None, min_length=1, max_length=200)
    action_type: ActionType | None = None
    target_element_id: str | None = Field(default=None, min_length=1, max_length=200)
    failure_category: FailureCategory | None = None
    stage: LoopStage | None = None
    success: bool = False
    count: int = Field(default=1, ge=1)
    # Advisory confidence weight [0, 1]. Halved each time this hint is used but the
    # subsequent verification fails. Hints whose effective bucket weight drops below
    # 0.1 are pruned from get_hints() results.
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryHint(StrictModel):
    """Advisory hint returned to rule evaluation and LLM policy prompts."""

    key: str = Field(min_length=1, max_length=100)
    hint: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=32)
    count: int = Field(ge=1)


class SpatialCache:
    """Rolling cache of perceived UI elements from the last N frames.

    Tracks which elements were visible per frame so the perception service can
    identify GhostElements: elements present at T-1 but absent at T-0 despite a
    stable screen (visual_velocity < 2%). These are likely occluded rather than gone.
    """

    def __init__(self, max_frames: int = 3) -> None:
        self._frames: deque[list[UIElement]] = deque(maxlen=max_frames)

    def push(self, elements: list[UIElement]) -> None:
        """Record the current frame's element list into the rolling window."""
        self._frames.append(list(elements))

    def prev_frame(self) -> list[UIElement]:
        """Return elements from the immediately preceding frame (T-1), or empty list."""
        if not self._frames:
            return []
        return list(self._frames[-1])

    def clear(self) -> None:
        """Discard all cached frames. Called at the start of each new run."""
        self._frames.clear()
