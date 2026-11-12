"""Thin harness profile (``OPERON_HARNESS_PROFILE=thin``) for model-driven backends.

The model decides every action and the executor runs exactly that. Nothing here
replaces, retries or resets an action behind the model's back:

- ``ThinPolicy`` passes the backend's decision through, with no rule engine or
  post-model guards in front of it.
- ``OwnDoneVerifier`` lets a failed action stand as the model's tool result and
  takes the model's "done" as the end of the run; graders decide success.
- ``ThinRecovery`` always hands the next step back to the model; the run's step
  budget and the model's own "done" are the only stops.

Safety gates (trust gate, file roots, coordinate-thrash pause, max steps) stay in
the loop and apply to every profile.
"""

from __future__ import annotations

import logging
import os

from operon.models.common import StopReason
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus

from .recovery import RecoveryManager
from .service import PolicyService
from .verifier import VerifierService

logger = logging.getLogger(__name__)

THIN = "thin"
FULL = "full"

# Gemini computer use (browser and desktop) runs full by default and thin on request,
# so the benchmark can run the same model in both harnesses (KAN-67).
GEMINI_COMPUTER_USE = "computer_use"


def harness_profile(backend: str) -> str:
    """The harness profile for a backend. Gemini computer use runs full unless
    ``OPERON_HARNESS_PROFILE=thin`` asks for the thin harness; the json backend always
    runs full, since its decisions still need the rules and the critic."""
    requested = (os.getenv("OPERON_HARNESS_PROFILE") or "").strip().lower()
    if backend == GEMINI_COMPUTER_USE:
        return THIN if requested == THIN else FULL
    if requested == THIN:
        logger.warning("OPERON_HARNESS_PROFILE=thin needs backend=computer_use (got %r); using full", backend)
    return FULL


class ThinPolicy(PolicyService):
    """The backend's decision, unchanged."""

    def __init__(self, delegate: PolicyService) -> None:
        self.delegate = delegate

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        return await self.delegate.choose_action(state, perception)

    def supports_resampling(self, run_id: str = "") -> bool:
        return False

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        getter = getattr(self.delegate, "latest_debug_artifacts", None)
        return getter() if getter is not None else None


class OwnDoneVerifier(VerifierService):
    """The model's own "done" ends the run, unchecked.

    No second model judges the claim. The benchmark's graders (app state, request logs)
    decide success afterwards, which makes a false success a clean metric. Every other
    step passes through: a failed action is reported to the model as its tool result by
    the backend, not retried here."""

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return None

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        *,
        allow_stable_wait: bool = True,
    ) -> VerificationResult:
        if decision.action.action_type is ActionType.STOP:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason=f"Model says done (not checked; graders decide): {decision.rationale}",
                stop_reason=StopReason.TASK_COMPLETED,
                unchecked=True,
            )
        if not executed_action.success:
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason=f"Action failed; reported to the model: {executed_action.detail or 'no detail'}",
                failure_category=executed_action.failure_category,
                failure_stage=executed_action.failure_stage,
            )
        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="Action executed; the model judges its effect from the next screenshot.",
            unchecked=True,
        )


class ThinRecovery(RecoveryManager):
    """The model chooses the next step; only its own "done" (or the step budget) ends the run."""

    async def recover(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> RecoveryDecision:
        if verification.stop_condition_met:
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message=verification.reason,
                terminal=True,
                recoverable=False,
                stop_reason=verification.stop_reason,
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.ADVANCE,
            message="The model sees the result and chooses the next step.",
        )
