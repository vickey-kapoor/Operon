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
    """Deterministic recovery rules for the Gmail draft MVP."""

    async def recover(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> RecoveryDecision:
        retry_key = self._retry_key(decision, verification)
        retries = state.retry_counts.get(retry_key, 0)

        if verification.stop_condition_met:
            state.retry_counts[retry_key] = retries
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message="Verifier signaled the stop-before-send boundary.",
                terminal=True,
                recoverable=False,
                stop_reason=StopReason.STOP_BEFORE_SEND,
            )

        if verification.status is VerificationStatus.SUCCESS and verification.expected_outcome_met:
            state.retry_counts[retry_key] = 0
            return RecoveryDecision(
                strategy=RecoveryStrategy.ADVANCE,
                message="Expected outcome met; advance to the next subgoal.",
                terminal=False,
                recoverable=True,
            )

        if verification.status is VerificationStatus.UNCERTAIN:
            state.retry_counts[retry_key] = retries + 1
            if retries >= 1:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.BACKOFF,
                    message="Repeated uncertain verification; back off before retrying.",
                    retry_after_ms=2000,
                    failure_category=verification.failure_category or FailureCategory.UNCERTAIN_SCREEN_STATE,
                    failure_stage=verification.failure_stage or LoopStage.VERIFY,
                    terminal=False,
                    recoverable=True,
                )
            return RecoveryDecision(
                strategy=RecoveryStrategy.WAIT_AND_RETRY,
                message="Verification uncertain; wait briefly and retry the same step.",
                retry_after_ms=1000,
                failure_category=verification.failure_category or FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=verification.failure_stage or LoopStage.VERIFY,
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

        if verification.failure_type is VerificationFailureType.ACTION_FAILED:
            state.retry_counts[retry_key] = retries + 1
            if retries >= 2:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.STOP,
                    message="Retry limit reached after repeated action failures.",
                    failure_category=FailureCategory.RETRY_LIMIT_REACHED,
                    failure_stage=LoopStage.RECOVER,
                    terminal=True,
                    recoverable=False,
                    stop_reason=StopReason.RETRY_LIMIT_REACHED,
                )
            return RecoveryDecision(
                strategy=RecoveryStrategy.RETRY_SAME_STEP,
                message="Action failed; retry the same step.",
                failure_category=verification.failure_category or FailureCategory.EXECUTION_ERROR,
                failure_stage=verification.failure_stage or LoopStage.EXECUTE,
                terminal=False,
                recoverable=True,
            )

        state.retry_counts[retry_key] = retries + 1
        if retries >= 1:
            return RecoveryDecision(
                strategy=RecoveryStrategy.BACKOFF,
                message="Expected outcome still not met; back off before retrying.",
                retry_after_ms=2000,
                failure_category=verification.failure_category or FailureCategory.EXPECTED_OUTCOME_NOT_MET,
                failure_stage=verification.failure_stage or LoopStage.VERIFY,
                terminal=False,
                recoverable=True,
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.WAIT_AND_RETRY,
            message="Expected outcome not met; wait briefly and retry.",
            retry_after_ms=1000,
            failure_category=verification.failure_category or FailureCategory.EXPECTED_OUTCOME_NOT_MET,
            failure_stage=verification.failure_stage or LoopStage.VERIFY,
            terminal=False,
            recoverable=True,
        )

    @staticmethod
    def _retry_key(decision: PolicyDecision, verification: VerificationResult) -> str:
        suffix = verification.failure_type.value if verification.failure_type else verification.status.value
        return f"{decision.active_subgoal}:{suffix}"
