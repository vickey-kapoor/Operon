"""Compact advisory memory models for local benchmark learning."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StrictModel
from src.models.perception import GhostElement, PageHint, UIElement
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


class RollingElementBuffer:
    """Rolling N-frame buffer of perceived UI elements with ghost-element TTL tracking.

    Each frame's element list is stored in a deque capped at max_frames.  When the
    perception service detects that a previously-visible element is absent from the
    current frame (despite a stable screen), it becomes a GhostElement — assumed
    occluded rather than gone.

    Ghost TTL: each ghost survives for exactly _GHOST_TTL_FRAMES additional frames
    after first detection.  If the element does not reappear within that window it is
    purged so the policy engine never acts on coordinates from a window that closed
    several seconds ago.
    """

    _GHOST_TTL_FRAMES: int = 2

    def __init__(self, max_frames: int = 3) -> None:
        self._frames: deque[list[UIElement]] = deque(maxlen=max_frames)
        # element_id → (GhostElement, remaining_ttl)
        self._active_ghosts: dict[str, tuple[GhostElement, int]] = {}

    def push(self, elements: list[UIElement]) -> None:
        """Record the current frame's element list into the rolling window."""
        self._frames.append(list(elements))

    def prev_frame(self) -> list[UIElement]:
        """Return elements from the immediately preceding frame (T-1), or empty list."""
        if not self._frames:
            return []
        return list(self._frames[-1])

    def update_ghosts(
        self,
        new_ghosts: list[GhostElement],
        current_elements: list[UIElement],
    ) -> list[GhostElement]:
        """Age existing ghosts, register new ones, and return only surviving entries.

        Per-ghost lifecycle:
        - First detected absent: TTL set to _GHOST_TTL_FRAMES (2).
        - Each subsequent frame still absent: TTL decremented by 1.
        - Reappears in current_elements: removed immediately.
        - TTL reaches 0: purged — prevents stale clicks on closed windows.

        Returns the list of GhostElements still within their TTL budget.
        """
        current_ids = {e.element_id for e in current_elements}

        # Age surviving ghosts; drop reappeared or expired entries.
        surviving: dict[str, tuple[GhostElement, int]] = {}
        for eid, (ghost, ttl) in self._active_ghosts.items():
            if eid in current_ids:
                continue  # element reappeared — remove from ghost set
            new_ttl = ttl - 1
            if new_ttl > 0:
                surviving[eid] = (ghost, new_ttl)
            # ttl == 0: expired — do not carry forward

        # Register freshly detected ghosts at full TTL (or refresh if already tracked).
        for g in new_ghosts:
            surviving[g.element_id] = (g, self._GHOST_TTL_FRAMES)

        self._active_ghosts = surviving
        return [ghost for ghost, _ in surviving.values()]

    def clear(self) -> None:
        """Discard all cached frames and active ghosts. Called at run start."""
        self._frames.clear()
        self._active_ghosts.clear()


# Backwards-compatible alias so any external code using the old name still works.
SpatialCache = RollingElementBuffer
