"""Schemas for per-step JSONL logging records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from operon.models.common import (
    FailureCategory,
    LoopStage,
    RunFailureCause,
    RunStatus,
    StopReason,
    StrictModel,
)
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.progress import ProgressState
from operon.models.recovery import RecoveryDecision
from operon.models.usage import ModelUsage
from operon.models.verification import VerificationResult


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
    # None when nothing ran in the step (a pause before the action), so no after screenshot exists.
    after_artifact_path: str | None = Field(default=None, min_length=1)
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
    decision_source: str | None = None
    visual_variance: float | None = None
    # Actions this step ran before ``executed_action`` that failed their check and
    # were retried in the same step (full harness), oldest first.
    earlier_attempts: list[ExecutedAction] = Field(default_factory=list)
    # Wall-clock UTC start and end of the step, the step's duration, and the part of it
    # spent waiting on the model (perceive + policy stages). Absent in older run.jsonl files.
    started_at: datetime | None = None
    finished_at: datetime | None = None
    step_latency_ms: float | None = Field(default=None, ge=0)
    model_latency_ms: float | None = Field(default=None, ge=0)


class PreStepFailureLog(StrictModel):
    """Structured log record for a failure before a control-loop step completed."""

    record_type: Literal["pre_step_failure"] = "pre_step_failure"
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    before_artifact_path: str = Field(min_length=1)
    # None when the perception backend failed before it reported any files of its own.
    perception_debug: ModelDebugArtifacts | None = None
    failure: FailureRecord
    error_message: str = Field(min_length=1)
    # Wall-clock UTC time the failure was logged. Absent in older run.jsonl files.
    logged_at: datetime | None = None


class RunEndLog(StrictModel):
    """KAN-175: terminal run.jsonl row for a run that ended outside a completed step.

    Written when the step budget runs out or a step crashes, so the step log always
    says how a failed run ended. Readers that only want steps skip it by ``record_type``.
    """

    record_type: Literal["run_end"] = "run_end"
    run_id: str = Field(min_length=1)
    step_index: int = Field(ge=0, description="The run's step count when it ended (a step that crashed partway counts).")
    status: RunStatus
    stop_reason: StopReason | None = None
    failure_cause: RunFailureCause | None = None
    failure: FailureRecord | None = None
    message: str = Field(min_length=1)


RunLogEntry = StepLog | PreStepFailureLog | RunEndLog
