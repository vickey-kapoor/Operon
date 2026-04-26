"""Schemas for per-step JSONL logging records."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StopReason, StrictModel
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.progress import ProgressState
from src.models.recovery import RecoveryDecision
from src.models.usage import ModelUsage
from src.models.verification import VerificationResult


class ModelDebugArtifacts(StrictModel):
    """Artifact references for one model-backed stage within a step."""

    prompt_artifact_path: str = Field(min_length=1)
    raw_response_artifact_path: str = Field(min_length=1)
    parsed_artifact_path: str = Field(min_length=1)
    retry_log_artifact_path: str | None = None
    selector_trace_artifact_path: str | None = None
    diagnostics_artifact_path: str | None = None
    usage_artifact_path: str | None = None
    usage: ModelUsage | None = None


class FailureRecord(StrictModel):
    """Explicit failure taxonomy record for one non-success step."""

    category: FailureCategory
    stage: LoopStage
    retry_count: int = Field(ge=0)
    terminal: bool
    recoverable: bool
    reason: str = Field(min_length=1)
    stop_reason: StopReason | None = None


class StepLog(StrictModel):
    """Structured log record for one completed control-loop step."""

    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    before_artifact_path: str = Field(min_length=1)
    after_artifact_path: str = Field(min_length=1)
    perception_debug: ModelDebugArtifacts
    policy_debug: ModelDebugArtifacts
    verification_debug: ModelDebugArtifacts | None = None
    perception: ScreenPerception
    policy_decision: PolicyDecision
    executed_action: ExecutedAction
    verification_result: VerificationResult
    recovery_decision: RecoveryDecision
    progress_state: ProgressState | None = None
    progress_trace_artifact_path: str | None = Field(default=None, min_length=1)
    failure: FailureRecord | None = None


class PreStepFailureLog(StrictModel):
    """Structured log record for a failure before a control-loop step completed."""

    record_type: Literal["pre_step_failure"] = "pre_step_failure"
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    before_artifact_path: str = Field(min_length=1)
    perception_debug: ModelDebugArtifacts
    failure: FailureRecord
    error_message: str = Field(min_length=1)


RunLogEntry = StepLog | PreStepFailureLog
