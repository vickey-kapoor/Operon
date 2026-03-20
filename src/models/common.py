"""Shared strict model primitives and API-facing DTOs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that forbids undeclared fields across all schemas."""

    model_config = ConfigDict(extra="forbid")


class RunStatus(StrEnum):
    """Lifecycle states for a local agent run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LoopStage(StrEnum):
    """Named stages of the core control loop."""

    ORCHESTRATE = "orchestrate"
    CAPTURE = "capture"
    PERCEIVE = "perceive"
    UPDATE_STATE = "update_state"
    CHOOSE_ACTION = "choose_action"
    EXECUTE = "execute"
    VERIFY = "verify"
    RECOVER = "recover"


class FailureCategory(StrEnum):
    """Explicit failure categories used across execution, verification, and recovery."""

    EXECUTION_TARGET_NOT_FOUND = "execution_target_not_found"
    EXECUTION_TARGET_NOT_EDITABLE = "execution_target_not_editable"
    EXECUTION_ERROR = "execution_error"
    EXPECTED_OUTCOME_NOT_MET = "expected_outcome_not_met"
    UNCERTAIN_SCREEN_STATE = "uncertain_screen_state"
    RETRY_LIMIT_REACHED = "retry_limit_reached"
    MAX_STEP_LIMIT_REACHED = "max_step_limit_reached"


class StopReason(StrEnum):
    """Terminal reasons for ending a run."""

    STOP_BEFORE_SEND = "stop_before_send"
    RETRY_LIMIT_REACHED = "retry_limit_reached"
    MAX_STEP_LIMIT_REACHED = "max_step_limit_reached"


class RunTaskRequest(StrictModel):
    intent: str = Field(min_length=1, max_length=500)


class StepRequest(StrictModel):
    run_id: str = Field(min_length=1)


class RunResponse(StrictModel):
    run_id: str = Field(min_length=1)
    status: RunStatus
    intent: str = Field(min_length=1)
    step_count: int = Field(ge=0)


class HealthResponse(StrictModel):
    status: str = Field(min_length=1)
