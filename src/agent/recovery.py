"""Recovery manager interface for failed or uncertain steps."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.common import FailureCategory, LoopStage, StopReason
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)


class RecoveryManager(ABC):
    @abstractmethod
    async def recover(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> RecoveryDecision:
        """Return the next recovery step after verification completes."""


class RuleBasedRecoveryManager(RecoveryManager):
    """Deterministic recovery rules driven by verification facts and repetition signals."""

    MAX_RECOVERY_ATTEMPTS = 5

    async def recover(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> RecoveryDecision:
        retry_key = self._retry_key(decision, executed_action, verification)
        retries = state.retry_counts.get(retry_key, 0)

        if verification.stop_condition_met:
            state.retry_counts[retry_key] = retries
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message="Verifier signaled a terminal stop condition.",
                failure_category=verification.failure_category,
                failure_stage=verification.failure_stage,
                terminal=True,
                recoverable=verification.stop_reason is StopReason.STOP_BEFORE_SEND,
                stop_reason=verification.stop_reason or StopReason.STOP_BEFORE_SEND,
            )

        if verification.failure_category is FailureCategory.EXECUTION_NO_PROGRESS:
            state.retry_counts[retry_key] = retries + 1
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message="Repeated no-progress execution detected; stop instead of looping indefinitely.",
                failure_category=FailureCategory.EXECUTION_NO_PROGRESS,
                failure_stage=verification.failure_stage or LoopStage.EXECUTE,
                terminal=True,
                recoverable=False,
                stop_reason=StopReason.EXECUTION_NO_PROGRESS,
            )

        if verification.status is VerificationStatus.SUCCESS and verification.expected_outcome_met:
            state.retry_counts[retry_key] = 0
            return RecoveryDecision(
                strategy=RecoveryStrategy.ADVANCE,
                message="Expected outcome met; advance to the next subgoal.",
                terminal=False,
                recoverable=True,
            )

        if (
            verification.failure_type is VerificationFailureType.ACTION_FAILED
            and decision.action.action_type is ActionType.TYPE
            and executed_action.failure_category in {
                FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
            }
        ):
            state.retry_counts[retry_key] = retries + 1
            target = decision.action.target_element_id or "input"
            state.current_subgoal = f"focus {target}"
            return RecoveryDecision(
                strategy=RecoveryStrategy.WAIT_AND_RETRY,
                message="Type failed; focus the target first before typing again.",
                retry_after_ms=500,
                failure_category=executed_action.failure_category,
                failure_stage=LoopStage.EXECUTE,
                terminal=False,
                recoverable=True,
            )

        return self._ladder_decision(
            state=state,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            retry_key=retry_key,
            retries=retries,
        )

    def _ladder_decision(
        self,
        *,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
        retry_key: str,
        retries: int,
    ) -> RecoveryDecision:
        state.retry_counts[retry_key] = retries + 1
        attempt = retries + 1
        failure_category = (
            verification.failure_category
            or executed_action.failure_category
            or FailureCategory.EXPECTED_OUTCOME_NOT_MET
        )
        failure_stage = verification.failure_stage or executed_action.failure_stage or LoopStage.VERIFY
        subgoal = decision.active_subgoal or "current step"

        if attempt >= self.MAX_RECOVERY_ATTEMPTS:
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message="Recovery ladder exhausted after repeated failures on the same subgoal and target.",
                failure_category=FailureCategory.RETRY_LIMIT_REACHED,
                failure_stage=LoopStage.RECOVER,
                terminal=True,
                recoverable=False,
                stop_reason=StopReason.RETRY_LIMIT_REACHED,
            )

        if attempt == 1:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY_SAME_STEP,
                message="First failure on this subgoal and target; retry once before escalating.",
                failure_category=failure_category,
                failure_stage=failure_stage,
                terminal=False,
                recoverable=True,
            )

        if attempt == 2:
            state.current_subgoal = f"Try a different tactic for: {subgoal}"
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY_DIFFERENT_TACTIC,
                message="Repeated failure detected; choose a different tactic instead of repeating the same action.",
                failure_category=failure_category,
                failure_stage=LoopStage.RECOVER,
                terminal=False,
                recoverable=True,
            )

        if attempt == 3:
            state.current_subgoal = f"Reset the local page context, then continue: {subgoal}"
            return RecoveryDecision(
                strategy=RecoveryStrategy.CONTEXT_RESET,
                message="Repeated failure persists; reset the local context before continuing.",
                retry_after_ms=1000,
                failure_category=failure_category,
                failure_stage=LoopStage.RECOVER,
                terminal=False,
                recoverable=True,
            )

        state.current_subgoal = f"Restart the session context, then continue: {subgoal}"
        return RecoveryDecision(
            strategy=RecoveryStrategy.SESSION_RESET,
            message="Context reset was insufficient; restart the session context before continuing.",
            retry_after_ms=1500,
            failure_category=failure_category,
            failure_stage=LoopStage.RECOVER,
            terminal=False,
            recoverable=True,
        )

    @staticmethod
    def _retry_key(
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> str:
        action = executed_action.action
        target_text = None
        if action.target_context is not None and action.target_context.intent.target_text is not None:
            target_text = action.target_context.intent.target_text.strip().lower()
        target = (
            f"text:{target_text}"
            if target_text
            else f"id:{action.target_element_id}"
            if action.target_element_id is not None
            else f"selector:{action.selector}"
            if action.selector is not None
            else f"xy:{action.x}:{action.y}"
            if action.x is not None and action.y is not None
            else "no_target"
        )
        suffix = (
            verification.failure_category.value
            if verification.failure_category is not None
            else verification.failure_type.value
            if verification.failure_type is not None
            else verification.status.value
        )
        return f"{decision.active_subgoal}:{target}:{suffix}"
