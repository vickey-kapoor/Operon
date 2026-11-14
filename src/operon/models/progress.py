"""Progress tracking schemas for deterministic multi-step loop stability."""

from __future__ import annotations

from pydantic import Field

from operon.models.common import FailureCategory, StopReason, StrictModel


class ProgressState(StrictModel):
    """Serializable run-level progress state used for loop suppression."""

    completed_targets: list[str] = Field(default_factory=list)
    completed_subgoals: list[str] = Field(default_factory=list)
    target_value_history: dict[str, str] = Field(default_factory=dict)
    recent_actions: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)
    repeated_target_count: dict[str, int] = Field(default_factory=dict)
    repeated_action_count: dict[str, int] = Field(default_factory=dict)
    last_meaningful_progress_step: int | None = Field(default=None, ge=1)
    no_progress_streak: int = Field(default=0, ge=0)
    loop_detected: bool = False
    # Screen change ratio from the most recently completed step (0.0–1.0).
    # Used by _block_redundant_action to detect a Stalled State when < 0.01 (1%).
    last_screen_change_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    latest_page_signature: str | None = Field(default=None, min_length=1)
    previous_page_signature: str | None = Field(default=None, min_length=1)
    latest_subgoal_signature: str | None = Field(default=None, min_length=1)
    target_completion_page_signatures: dict[str, str] = Field(default_factory=dict)
    subgoal_completion_page_signatures: dict[str, str] = Field(default_factory=dict)
    # RFC 0003 (OPERON_PROGRESS_REVISIT): per action signature, the after-screenshot
    # paths of its recent successful runs. Unlike the repeat counters it is not
    # cleared on progress, so a repeat can be compared with what it produced before.
    action_outcomes: dict[str, list[str]] = Field(default_factory=dict)


class ProgressTrace(StrictModel):
    """Per-step progress and loop-suppression trace."""

    step_index: int = Field(ge=1)
    page_signature: str = Field(min_length=1)
    action_signature: str = Field(min_length=1)
    target_signature: str | None = Field(default=None, min_length=1)
    subgoal_signature: str = Field(min_length=1)
    failure_signature: str | None = Field(default=None, min_length=1)
    blocked_as_redundant: bool = False
    redundancy_reason: FailureCategory | None = None
    loop_pattern_detected: str | None = Field(default=None, min_length=1)
    progress_made: bool = False
    # RFC 0003: this step would have counted as progress, but the action reached a
    # screen it had already produced, so it did not. False when today's rule already
    # said no progress (the revisit check only runs to overturn a "progress").
    revisited_screen: bool = False
    screen_change_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    no_progress_streak: int = Field(default=0, ge=0)
    final_failure_category: FailureCategory | None = None
    final_stop_reason: StopReason | None = None
