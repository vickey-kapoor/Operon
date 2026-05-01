"""Recovery manager interface for failed or uncertain steps."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.agent.subgoal_utils import wrap_subgoal
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

logger = logging.getLogger(__name__)

# Stop reasons that claim task success — only valid when verification status is SUCCESS.
_SUCCESS_STOP_REASONS: frozenset[StopReason] = frozenset({
    StopReason.FORM_SUBMITTED_SUCCESS,
    StopReason.TASK_COMPLETED,
    StopReason.STOP_BEFORE_SEND,
})


def validate_benchmark_integrity(
    recovery: RecoveryDecision,
    verification: VerificationResult,
) -> RecoveryDecision:
    """Benchmark Integrity Check: validate a recovery decision before the loop commits to it.

    Blocks two classes of integrity violation:

    1. Unverified success claim — a recovery decision that marks the run as
       successfully complete (terminal + success stop_reason) when the verifier
       did not confirm SUCCESS. This would allow a recovery "patch" to bypass the
       Verify step and declare victory without visual evidence.

    2. Guardrail bypass — a non-terminal ADVANCE decision when the verifier
       explicitly signaled STOP (stop_condition_met=True). Advancing past a stop
       boundary ignores a safety signal.

    When a violation is detected, the recovery is replaced with a safe hard-stop
    that surfaces the integrity violation in the run log.
    """
    # Rule 1: success stop_reason requires verification SUCCESS
    if (
        recovery.terminal
        and recovery.stop_reason in _SUCCESS_STOP_REASONS
        and verification.status is not VerificationStatus.SUCCESS
    ):
        logger.warning(
            "BenchmarkIntegrity: recovery claims %r (terminal success) but "
            "verification status is %r — rejecting and hard-stopping.",
            recovery.stop_reason.value if recovery.stop_reason else "None",
            verification.status.value,
        )
        return RecoveryDecision(
            strategy=RecoveryStrategy.STOP,
            message=(
                "Benchmark integrity check failed: recovery attempted to mark success "
                "without visual confirmation from the verifier. "
                f"verification_status={verification.status.value}. Hard stop applied."
            ),
            failure_category=FailureCategory.EXPECTED_OUTCOME_NOT_MET,
            failure_stage=LoopStage.RECOVER,
            terminal=True,
            recoverable=False,
            stop_reason=StopReason.RETRY_LIMIT_REACHED,
        )

    # Rule 2: cannot advance past a verifier stop boundary
    if (
        not recovery.terminal
        and recovery.strategy is RecoveryStrategy.ADVANCE
        and verification.stop_condition_met
        and verification.status is not VerificationStatus.SUCCESS
    ):
        logger.warning(
            "BenchmarkIntegrity: recovery strategy is ADVANCE but verifier set "
            "stop_condition_met=True with status=%r — rejecting advance.",
            verification.status.value,
        )
        return RecoveryDecision(
            strategy=RecoveryStrategy.STOP,
            message=(
                "Benchmark integrity check failed: cannot advance past a verifier stop boundary "
                f"(stop_condition_met=True, status={verification.status.value}). Hard stop applied."
            ),
            failure_category=verification.failure_category or FailureCategory.EXPECTED_OUTCOME_NOT_MET,
            failure_stage=verification.failure_stage or LoopStage.RECOVER,
            terminal=True,
            recoverable=False,
            stop_reason=verification.stop_reason or StopReason.RETRY_LIMIT_REACHED,
        )

    return recovery


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

        if verification.status is VerificationStatus.PROGRESSING_STABLE:
            # UI reacted to the action (ripple, focus ring, loading state) but the
            # page has not fully changed. This is forward motion — advance to let the
            # next perception decide whether the task goal is now visible.
            state.retry_counts[retry_key] = 0
            return RecoveryDecision(
                strategy=RecoveryStrategy.ADVANCE,
                message="UI reacted to the action; advancing — interaction confirmed, awaiting full page update.",
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
            target = decision.action.target_element_id or "input"
            state.current_subgoal = f"focus {target}"
            # Pass retries unchanged so _ladder_decision counts this as attempt 1
            # (not a second increment on the same failure).
            return self._ladder_decision(
                state=state,
                decision=decision,
                executed_action=executed_action,
                verification=verification,
                retry_key=retry_key,
                retries=retries,
                override_strategy=RecoveryStrategy.WAIT_AND_RETRY,
                override_message="Type failed; focus the target first before typing again.",
                override_retry_after_ms=500,
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
        override_strategy: RecoveryStrategy | None = None,
        override_message: str | None = None,
        override_retry_after_ms: int | None = None,
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
                strategy=override_strategy or RecoveryStrategy.RETRY_SAME_STEP,
                message=override_message or "First failure on this subgoal and target; retry once before escalating.",
                retry_after_ms=override_retry_after_ms,
                failure_category=failure_category,
                failure_stage=failure_stage,
                terminal=False,
                recoverable=True,
            )

        if attempt == 2:
            state.current_subgoal = wrap_subgoal("Try a different tactic for: ", subgoal)
            return RecoveryDecision(
                strategy=override_strategy or RecoveryStrategy.RETRY_DIFFERENT_TACTIC,
                message=override_message or "Repeated failure detected; choose a different tactic instead of repeating the same action.",
                retry_after_ms=override_retry_after_ms,
                failure_category=failure_category,
                failure_stage=LoopStage.RECOVER,
                terminal=False,
                recoverable=True,
            )

        if attempt == 3:
            state.current_subgoal = wrap_subgoal("Reset the local page context, then continue: ", subgoal)
            return RecoveryDecision(
                strategy=override_strategy or RecoveryStrategy.CONTEXT_RESET,
                message=override_message or "Repeated failure persists; reset the local context before continuing.",
                retry_after_ms=override_retry_after_ms if override_retry_after_ms is not None else 1000,
                failure_category=failure_category,
                failure_stage=LoopStage.RECOVER,
                terminal=False,
                recoverable=True,
            )

        state.current_subgoal = wrap_subgoal("Restart the session context, then continue: ", subgoal)
        return RecoveryDecision(
            strategy=override_strategy or RecoveryStrategy.SESSION_RESET,
            message=override_message or "Context reset was insufficient; restart the session context before continuing.",
            retry_after_ms=override_retry_after_ms if override_retry_after_ms is not None else 1500,
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
