"""Recovery manager interface for failed or uncertain steps."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from operon.models.common import FailureCategory, LoopStage, StopReason
from operon.models.execution import ExecutedAction
from operon.models.policy import PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import (
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
