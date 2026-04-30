"""Schemas for action execution results and trace artifacts."""

from __future__ import annotations

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StrictModel
from src.models.policy import AgentAction
from src.models.selector import OriginalTargetSignature, SelectorTrace, TargetIntent


class ExecutionTargetSnapshot(StrictModel):
    """Observed DOM state for the action target before or after execution."""

    target_element_id: str | None = Field(default=None, min_length=1)
    dom_id: str | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)
    tag_name: str | None = Field(default=None, min_length=1)
    input_type: str | None = Field(default=None, min_length=1)
    x: float | None = Field(default=None, ge=0)
    y: float | None = Field(default=None, ge=0)
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    is_visible: bool
    is_interactable: bool
    is_focused: bool
    value: str | None = None
    checked: bool | None = None
    selected_value: str | None = None
    page_signature: str | None = Field(default=None, min_length=1)
    page_url: str | None = Field(default=None, min_length=1)


class ExecutionAttemptTrace(StrictModel):
    """Single execution attempt details for one action."""

    attempt_index: int = Field(ge=1)
    selected_target_before_action: ExecutionTargetSnapshot | None = None
    selected_target_after_action: ExecutionTargetSnapshot | None = None
    revalidation_result: str = Field(min_length=1)
    focus_verification_result: str | None = Field(default=None, min_length=1)
    verification_result: str = Field(min_length=1)
    no_progress_detected: bool = False
    failure_category: FailureCategory | None = None


class ExecutionReresolutionTrace(StrictModel):
    """Deterministic trace for one intent-based target re-resolution attempt."""

    trigger_reason: FailureCategory
    original_target_element_id: str | None = Field(default=None, min_length=1)
    original_intent: TargetIntent
    original_target_signature: OriginalTargetSignature
    original_page_signature: str | None = Field(default=None, min_length=1)
    selector_trace: SelectorTrace
    reused_original_element_id: bool = False
    final_target_element_id: str | None = Field(default=None, min_length=1)
    succeeded: bool = False
    detail: str = Field(min_length=1)


class ExecutionTrace(StrictModel):
    """Execution hardening trace spanning one or two deterministic attempts."""

    attempts: list[ExecutionAttemptTrace] = Field(default_factory=list)
    target_reresolved: bool = False
    retry_attempted: bool = False
    retry_reason: FailureCategory | None = None
    reresolution_trace: ExecutionReresolutionTrace | None = None
    final_outcome: str = Field(min_length=1)
    final_failure_category: FailureCategory | None = None


class AnchorSnapInfo(StrictModel):
    """Diagnostic record written when coordinate anchoring corrects perception jitter."""

    element_id: str = Field(min_length=1)
    original_x: int
    original_y: int
    anchored_x: int
    anchored_y: int
    drift_px: float


class ExecutedAction(StrictModel):
    """Recorded outcome of executing a single agent action."""

    action: AgentAction
    success: bool
    detail: str = Field(min_length=1)
    artifact_path: str | None = Field(default=None, min_length=1)
    execution_trace: ExecutionTrace | None = None
    execution_trace_artifact_path: str | None = Field(default=None, min_length=1)
    recording_path: str | None = Field(default=None, min_length=1)
    failure_category: FailureCategory | None = None
    failure_stage: LoopStage | None = None
    anchor_snap: AnchorSnapInfo | None = None
    visual_variance: float | None = None
