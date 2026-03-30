"""Verifier service interface for post-execution checks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.clients.gemini import GeminiClient
from src.models.common import FailureCategory, LoopStage, StopReason
from src.models.execution import ExecutedAction
from src.models.perception import PageHint
from src.models.policy import ActionType, PolicyDecision
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)


class VerifierService(ABC):
    """Typed interface for verification and stop-before-send checks."""

    @abstractmethod
    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        """Verify whether the executed action achieved the expected outcome."""


class DeterministicVerifierService(VerifierService):
    """Deterministic verifier for general-purpose agent tasks."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        """Evaluate typed inputs using deterministic rules."""
        action = decision.action
        latest_perception = state.observation_history[-1] if state.observation_history else None

        if latest_perception is not None and self._task_success_visible(latest_perception):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason="Task success state is visible.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                recovery_hint="stop",
                stop_reason=StopReason.FORM_SUBMITTED_SUCCESS,
            )

        if action.action_type is ActionType.STOP:
            if decision.active_subgoal == "stop for benchmark setup":
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    expected_outcome_met=False,
                    stop_condition_met=True,
                    reason="Benchmark precondition failed: authenticated Gmail start state was required.",
                    failure_type=VerificationFailureType.BENCHMARK_PRECONDITION_FAILED,
                    recovery_hint="stop",
                    failure_category=FailureCategory.BENCHMARK_PRECONDITION_FAILED,
                    failure_stage=LoopStage.CHOOSE_ACTION,
                    stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED,
                )
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason="Intentional task stop boundary met.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                recovery_hint="stop",
                stop_reason=self._stop_reason_for_intentional_stop(state, decision),
            )

        # Self-terminating actions: when the chosen action IS the goal
        if executed_action.success and self._is_goal_completing_action(state, action):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason=f"Goal-completing action {action.action_type.value} succeeded.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                recovery_hint="stop",
                stop_reason=StopReason.TASK_COMPLETED,
            )

        if not executed_action.success:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Executed action reported failure.",
                failure_type=VerificationFailureType.ACTION_FAILED,
                recovery_hint="retry_same_step",
                failure_category=executed_action.failure_category or FailureCategory.EXECUTION_ERROR,
                failure_stage=executed_action.failure_stage or LoopStage.EXECUTE,
            )

        if decision.confidence < 0.5:
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Policy confidence is too low to confirm the expected outcome.",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                recovery_hint="wait_and_retry",
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )

        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="Executed action succeeded and the expected outcome is treated as met.",
            recovery_hint="advance",
        )

    @staticmethod
    def _stop_reason_for_intentional_stop(state: AgentState, decision: PolicyDecision) -> StopReason:
        latest_perception = state.observation_history[-1] if state.observation_history else None
        if decision.active_subgoal == "verify_success" or (
            latest_perception is not None and latest_perception.page_hint is PageHint.FORM_SUCCESS
        ):
            return StopReason.FORM_SUBMITTED_SUCCESS
        if decision.active_subgoal == "stop for benchmark setup":
            return StopReason.BENCHMARK_PRECONDITION_FAILED
        return StopReason.TASK_COMPLETED

    @staticmethod
    def _is_goal_completing_action(state: AgentState, action: AgentAction) -> bool:
        """Check if this action type directly fulfills the task intent."""
        intent = state.intent.lower()
        if action.action_type is ActionType.READ_CLIPBOARD and ("clipboard" in intent or "read" in intent):
            return True
        if action.action_type is ActionType.HOVER and "hover" in intent:
            return True
        if action.action_type is ActionType.RIGHT_CLICK and ("right-click" in intent or "right click" in intent):
            return True
        if action.action_type is ActionType.DOUBLE_CLICK and "double-click" in intent:
            return True
        if action.action_type is ActionType.SCREENSHOT_REGION and ("screenshot" in intent or "capture" in intent):
            return True
        if action.action_type is ActionType.DRAG and "drag" in intent:
            return True
        if action.action_type is ActionType.WRITE_CLIPBOARD and ("clipboard" in intent or "copy" in intent):
            return True
        # Hotkey that matches the last phrase of the intent (e.g. "copy it with Ctrl+C")
        if action.action_type is ActionType.HOTKEY and action.key:
            key_lower = action.key.lower().replace("+", "")
            if key_lower in intent.replace("+", "").replace("-", ""):
                # Only if this is the LAST action in the intent
                last_phrase = intent.split(",")[-1].strip().lower() if "," in intent else intent
                if key_lower in last_phrase.replace("+", "").replace("-", ""):
                    return True
        return False

    @staticmethod
    def _task_success_visible(perception) -> bool:
        if perception.page_hint is PageHint.FORM_SUCCESS:
            return True
        success_tokens = ("thank you", "success", "submitted")
        if any(token in perception.summary.lower() for token in success_tokens):
            return True
        return any(any(token in element.primary_name.lower() for token in success_tokens) for element in perception.visible_elements)
