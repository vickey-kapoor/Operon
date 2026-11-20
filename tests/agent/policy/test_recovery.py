"""validate_benchmark_integrity: the loop's guard on every recovery decision."""

from __future__ import annotations

from operon.agent.policy.recovery import validate_benchmark_integrity
from operon.models.common import StopReason
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.verification import VerificationResult, VerificationStatus


def _verification(status: VerificationStatus, *, stop: bool = False) -> VerificationResult:
    return VerificationResult(
        status=status,
        expected_outcome_met=status is VerificationStatus.SUCCESS,
        stop_condition_met=stop,
        reason="r",
    )


def _stop(reason: StopReason) -> RecoveryDecision:
    return RecoveryDecision(strategy=RecoveryStrategy.STOP, message="m", terminal=True, recoverable=False, stop_reason=reason)


def test_a_success_claim_without_a_verified_success_is_hard_stopped() -> None:
    result = validate_benchmark_integrity(
        _stop(StopReason.TASK_COMPLETED), _verification(VerificationStatus.UNCERTAIN, stop=True)
    )

    assert result.terminal and result.stop_reason is StopReason.RETRY_LIMIT_REACHED
    assert "integrity" in result.message


def test_a_verified_success_claim_passes_unchanged() -> None:
    recovery = _stop(StopReason.TASK_COMPLETED)

    assert validate_benchmark_integrity(recovery, _verification(VerificationStatus.SUCCESS, stop=True)) is recovery


def test_advancing_past_a_verifier_stop_boundary_is_hard_stopped() -> None:
    advance = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="go on")

    result = validate_benchmark_integrity(advance, _verification(VerificationStatus.FAILURE, stop=True))

    assert result.strategy is RecoveryStrategy.STOP and result.terminal


def test_a_plain_advance_passes_unchanged() -> None:
    advance = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="go on")

    assert validate_benchmark_integrity(advance, _verification(VerificationStatus.SUCCESS)) is advance
