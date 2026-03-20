"""Schemas for persisted in-process agent state."""

from __future__ import annotations

from pydantic import Field

from src.models.common import RunStatus, StopReason, StrictModel
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception
from src.models.verification import VerificationResult


class AgentState(StrictModel):
    """Local typed state stored for a single Gmail draft run."""

    run_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    status: RunStatus
    current_subgoal: str | None = Field(default=None, min_length=1)
    step_count: int = Field(default=0, ge=0)
    observation_history: list[ScreenPerception] = Field(default_factory=list)
    action_history: list[ExecutedAction] = Field(default_factory=list)
    verification_history: list[VerificationResult] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    stop_reason: StopReason | None = None
