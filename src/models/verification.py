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
    PENDING = "pending"  # page is mid-transition (loading); wait 2–8s before re-verifying
    PROGRESSING_STABLE = "progressing_stable"  # UI reacted (Gemini-confirmed); advance immediately
    STABLE_WAIT = "stable_wait"  # screen actively changing post-action; wait 200ms and re-verify once


class VerificationFailureType(StrEnum):
    """Failure classes used to drive deterministic recovery decisions."""

    ACTION_FAILED = "action_failed"
    BENCHMARK_PRECONDITION_FAILED = "benchmark_precondition_failed"
    EXPECTED_OUTCOME_NOT_MET = "expected_outcome_not_met"
    PAGE_LOADING = "page_loading"  # transient: page navigating or redirecting
    STOP_BOUNDARY_REACHED = "stop_boundary_reached"
    UNCERTAIN_SCREEN_STATE = "uncertain_screen_state"


class VerificationResult(StrictModel):
    """Outcome of checking whether the expected UI change occurred."""

    status: VerificationStatus
    expected_outcome_met: bool
    stop_condition_met: bool
    reason: str = Field(min_length=1)
    failure_type: VerificationFailureType | None = None
    failure_category: FailureCategory | None = None
    failure_stage: LoopStage | None = None
    stop_reason: StopReason | None = None
    recovery_hint: str | None = Field(default=None, min_length=1)
    video_verified: bool = False
    video_detail: str | None = Field(default=None, min_length=1)
    critic_model_used: bool = False
    critic_fallback_reason: str | None = Field(default=None, min_length=1)
    patience_retries: int = 0  # number of PENDING re-verify attempts before settling
