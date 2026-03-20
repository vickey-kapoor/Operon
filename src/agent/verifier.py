"""Verifier service interface for post-execution checks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.clients.gemini import GeminiClient
from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction
from src.models.policy import ActionType
from src.models.policy import PolicyDecision
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
    """Minimal typed verifier for the Gmail draft MVP without model calls."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        """Evaluate typed inputs using deterministic Gmail-draft rules."""
        action = decision.action

        if action.action_type is ActionType.STOP:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason="Stop-before-send boundary met. Run should stop before sending.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                recovery_hint="stop",
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
