"""Schemas for recovery decisions after failed or uncertain steps."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StopReason, StrictModel


class RecoveryStrategy(StrEnum):
    """Supported recovery policies for the MVP loop."""

    RETRY_SAME_STEP = "retry_same_step"
    RETRY_DIFFERENT_TACTIC = "retry_different_tactic"
    WAIT_AND_RETRY = "wait_and_retry"
    BACKOFF = "backoff"
    CONTEXT_RESET = "context_reset"
    SESSION_RESET = "session_reset"
    STOP = "stop"
    ADVANCE = "advance"


class RecoveryDecision(StrictModel):
    """Decision describing how the loop should recover next."""

    strategy: RecoveryStrategy
    message: str = Field(min_length=1)
    retry_after_ms: int | None = Field(default=None, ge=1, le=30000)
    failure_category: FailureCategory | None = None
    failure_stage: LoopStage | None = None
    terminal: bool = False
    recoverable: bool = True
    stop_reason: StopReason | None = None
