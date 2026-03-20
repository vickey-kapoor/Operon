"""Schemas for per-step JSONL logging records."""

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StopReason, StrictModel
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.recovery import RecoveryDecision
from src.models.verification import VerificationResult


class ModelDebugArtifacts(StrictModel):
    """Artifact references for one model-backed stage within a step."""

    prompt_artifact_path: str = Field(min_length=1)
    raw_response_artifact_path: str = Field(min_length=1)
    parsed_artifact_path: str = Field(min_length=1)


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
    perception: ScreenPerception
    policy_decision: PolicyDecision
    executed_action: ExecutedAction
    verification_result: VerificationResult
    recovery_decision: RecoveryDecision
    failure: FailureRecord | None = None
