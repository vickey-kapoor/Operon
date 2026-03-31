"""Schemas for post-execution verification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StopReason, StrictModel


class VerificationStatus(StrEnum):
    """Deterministic verification outcomes for the MVP loop."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


class VerificationFailureType(StrEnum):
    """Failure classes used to drive deterministic recovery decisions."""

    ACTION_FAILED = "action_failed"
    BENCHMARK_PRECONDITION_FAILED = "benchmark_precondition_failed"
    EXPECTED_OUTCOME_NOT_MET = "expected_outcome_not_met"
    STOP_BOUNDARY_REACHED = "stop_boundary_reached"
    UNCERTAIN_SCREEN_STATE = "uncertain_screen_state"


class VerificationResult(StrictModel):
    """Outcome of checking whether the expected UI change occurred."""

    status: VerificationStatus
    expected_outcome_met: bool
    stop_condition_met: bool
    reason: str = Field(min_length=1)
    failure_type: VerificationFailureType | None = None
    recovery_hint: str | None = Field(default=None, min_length=1)
    failure_category: FailureCategory | None = None
    failure_stage: LoopStage | None = None
    stop_reason: StopReason | None = None
    video_verified: bool = False
    video_detail: str | None = Field(default=None, min_length=1)
