"""Verifier service interface for post-execution checks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from operon.models.execution import ExecutedAction
from operon.models.policy import PolicyDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationResult


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
