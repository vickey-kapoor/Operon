"""Progress tracking schemas for deterministic multi-step loop stability."""

from __future__ import annotations

from pydantic import Field

from src.models.common import FailureCategory, StopReason, StrictModel


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
    latest_page_signature: str | None = Field(default=None, min_length=1)
    target_completion_page_signatures: dict[str, str] = Field(default_factory=dict)
    subgoal_completion_page_signatures: dict[str, str] = Field(default_factory=dict)


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
    no_progress_streak: int = Field(default=0, ge=0)
    final_failure_category: FailureCategory | None = None
    final_stop_reason: StopReason | None = None
