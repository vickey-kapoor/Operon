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
    WAITING_FOR_USER = "waiting_for_user"
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

    BENCHMARK_PRECONDITION_FAILED = "benchmark_precondition_failed"
    PRE_STEP_PERCEPTION_FAILED = "pre_step_perception_failed"
    PERCEPTION_LOW_QUALITY = "perception_low_quality"
    AMBIGUOUS_TARGET_CANDIDATES = "ambiguous_target_candidates"
    SELECTOR_SCORE_BELOW_THRESHOLD = "selector_score_below_threshold"
    SELECTOR_NO_CANDIDATES_AFTER_FILTERING = "selector_no_candidates_after_filtering"
    TARGET_UNLABELED_INSUFFICIENT_GROUNDING = "target_unlabeled_insufficient_grounding"
    TARGET_ACTION_INCOMPATIBLE = "target_action_incompatible"
    SELECTOR_RECOVERY_USED = "selector_recovery_used"
    SELECTOR_RECOVERY_FAILED = "selector_recovery_failed"
    TARGET_RERESOLUTION_FAILED = "target_reresolution_failed"
    TARGET_RERESOLUTION_AMBIGUOUS = "target_reresolution_ambiguous"
    STALE_TARGET_BEFORE_ACTION = "stale_target_before_action"
    TARGET_SHIFTED_BEFORE_ACTION = "target_shifted_before_action"
    TARGET_LOST_BEFORE_ACTION = "target_lost_before_action"
    FOCUS_VERIFICATION_FAILED = "focus_verification_failed"
    CLICK_BEFORE_TYPE_FAILED = "click_before_type_failed"
    TYPE_VERIFICATION_FAILED = "type_verification_failed"
    CLICK_NO_EFFECT = "click_no_effect"
    CHECKBOX_VERIFICATION_FAILED = "checkbox_verification_failed"
    SELECT_VERIFICATION_FAILED = "select_verification_failed"
    EXECUTION_NO_PROGRESS = "execution_no_progress"
    REPEATED_ACTION_WITHOUT_PROGRESS = "repeated_action_without_progress"
    REPEATED_TARGET_WITHOUT_PROGRESS = "repeated_target_without_progress"
    REPEATED_FAILURE_LOOP = "repeated_failure_loop"
    NO_MEANINGFUL_PROGRESS_ACROSS_STEPS = "no_meaningful_progress_across_steps"
    SUBGOAL_ALREADY_COMPLETED = "subgoal_already_completed"
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
    FORM_SUBMITTED_SUCCESS = "form_submitted_success"
    TASK_COMPLETED = "task_completed"
    BENCHMARK_PRECONDITION_FAILED = "benchmark_precondition_failed"
    PRE_STEP_PERCEPTION_FAILED = "pre_step_perception_failed"
    PERCEPTION_LOW_QUALITY = "perception_low_quality"
    AMBIGUOUS_TARGET_CANDIDATES = "ambiguous_target_candidates"
    SELECTOR_SCORE_BELOW_THRESHOLD = "selector_score_below_threshold"
    SELECTOR_NO_CANDIDATES_AFTER_FILTERING = "selector_no_candidates_after_filtering"
    TARGET_UNLABELED_INSUFFICIENT_GROUNDING = "target_unlabeled_insufficient_grounding"
    TARGET_ACTION_INCOMPATIBLE = "target_action_incompatible"
    SELECTOR_RECOVERY_USED = "selector_recovery_used"
    SELECTOR_RECOVERY_FAILED = "selector_recovery_failed"
    TARGET_RERESOLUTION_FAILED = "target_reresolution_failed"
    TARGET_RERESOLUTION_AMBIGUOUS = "target_reresolution_ambiguous"
    STALE_TARGET_BEFORE_ACTION = "stale_target_before_action"
    TARGET_SHIFTED_BEFORE_ACTION = "target_shifted_before_action"
    TARGET_LOST_BEFORE_ACTION = "target_lost_before_action"
    FOCUS_VERIFICATION_FAILED = "focus_verification_failed"
    CLICK_BEFORE_TYPE_FAILED = "click_before_type_failed"
    TYPE_VERIFICATION_FAILED = "type_verification_failed"
    CLICK_NO_EFFECT = "click_no_effect"
    CHECKBOX_VERIFICATION_FAILED = "checkbox_verification_failed"
    SELECT_VERIFICATION_FAILED = "select_verification_failed"
    EXECUTION_NO_PROGRESS = "execution_no_progress"
    REPEATED_ACTION_WITHOUT_PROGRESS = "repeated_action_without_progress"
    REPEATED_TARGET_WITHOUT_PROGRESS = "repeated_target_without_progress"
    REPEATED_FAILURE_LOOP = "repeated_failure_loop"
    NO_MEANINGFUL_PROGRESS_ACROSS_STEPS = "no_meaningful_progress_across_steps"
    SUBGOAL_ALREADY_COMPLETED = "subgoal_already_completed"
    RETRY_LIMIT_REACHED = "retry_limit_reached"
    MAX_STEP_LIMIT_REACHED = "max_step_limit_reached"
    WAITING_FOR_USER = "waiting_for_user"


class ResumeRequest(StrictModel):
    """Resume a run that is paused waiting for user input."""
    run_id: str = Field(min_length=1)


class RunTaskRequest(StrictModel):
    intent: str = Field(min_length=1, max_length=500)
    start_url: str | None = Field(default=None, min_length=1)


class StepRequest(StrictModel):
    run_id: str = Field(min_length=1)


class RunResponse(StrictModel):
    run_id: str = Field(min_length=1)
    status: RunStatus
    intent: str = Field(min_length=1)
    step_count: int = Field(ge=0)


class HealthResponse(StrictModel):
    status: str = Field(min_length=1)
