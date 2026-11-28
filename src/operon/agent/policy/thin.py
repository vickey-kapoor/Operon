"""The harness: policy, verifier and recovery layers for every backend.

The model decides every action and the executor runs exactly that. Nothing here
replaces, retries or resets an action behind the model's back (KAN-183 retired the
rule-based "full" harness; this is the only harness):

- ``ThinPolicy`` passes the backend's decision through, with no rule engine or
  post-model guards in front of it.
- ``OwnDoneVerifier`` lets a failed action stand as the model's tool result and
  takes the model's "done" as the end of the run; graders decide success.
- ``ThinRecovery`` hands the next step back to the model. The run ends on the model's own
  "done", on the step budget, or when the model has replied with words only and no action
  ``WORDS_ONLY_LIMIT`` times in a row (KAN-204). That last stop changes nothing the model
  does: it only ends a run that is going nowhere.

With ``OPERON_DONE_CHECK`` on (experimental, KAN-208) the loop asks one independent check of a
"done" (``agent/done_check.py``). A rejection is only a hint on the next step: the model's
action is never replaced, and after two checks the model's done stands.

Safety gates (trust gate, file roots, coordinate-thrash pause, max steps, the word-only
stop) stay in the loop.
"""

from __future__ import annotations

from operon.models.common import FailureCategory, LoopStage, StopReason
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

THIN = "thin"

# The backends the loop runs: Gemini computer use, and the model backend
# (OPERON_*_BACKEND=model), which runs the registry model named by OPERON_MODEL_PROFILE
# (Claude, GPT; src/operon/agent/backends/model_adapter/models.yaml).
GEMINI_COMPUTER_USE = "computer_use"
MODEL_BACKEND = "model"

# KAN-204: replies in a row with words only (a question, a refusal) and no action that
# end the run. Like max steps, this is a stop, not a rule about what the model does.
WORDS_ONLY_LIMIT = 3
_WORDS_SHOWN = 350


class ThinPolicy(PolicyService):
    """The backend's decision, unchanged."""

    def __init__(self, delegate: PolicyService) -> None:
        self.delegate = delegate

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        return await self.delegate.choose_action(state, perception)

    @property
    def model_label(self) -> str | None:
        label = getattr(self.delegate, "model_label", None)
        return label if isinstance(label, str) and label else None

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        getter = getattr(self.delegate, "latest_debug_artifacts", None)
        return getter() if getter is not None else None


class OwnDoneVerifier(VerifierService):
    """The model's own "done" ends the run, unchecked.

    No second model judges the claim. The benchmark's graders (app state, request logs)
    decide success afterwards, which makes a false success a clean metric. Every other
    step passes through: a failed action is reported to the model as its tool result by
    the backend, not retried here. A ``fail`` (rationale starting with ``gave_up_prefix``)
    ends the run as not done. (With ``OPERON_DONE_CHECK`` on, the loop may ask one independent
    check of a "done" after this verdict; see ``agent/done_check.py``.)"""

    def __init__(self, *, gave_up_prefix: str = "Model gave up: ") -> None:
        self.gave_up_prefix = gave_up_prefix

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return None

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        if decision.action.action_type is ActionType.STOP:
            if (decision.rationale or "").startswith(self.gave_up_prefix):
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    expected_outcome_met=False,
                    stop_condition_met=True,
                    reason=decision.rationale,
                    stop_reason=StopReason.COMPLETION_UNVERIFIED,
                    # KAN-175: a structured cause for the step log (a label, not a rule).
                    failure_category=FailureCategory.MODEL_GAVE_UP,
                    failure_stage=LoopStage.CHOOSE_ACTION,
                )
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
    """The model chooses the next step. Its own "done", the step budget, or ``WORDS_ONLY_LIMIT``
    replies in a row with words only and no action end the run."""

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
                # None on a normal "done"; MODEL_GAVE_UP when the model gave up (KAN-175).
                failure_category=verification.failure_category,
                failure_stage=verification.failure_stage,
            )
        if decision.words_only is not None and state.words_only_streak >= WORDS_ONLY_LIMIT:
            words = decision.words_only.strip()
            shown = words if len(words) <= _WORDS_SHOWN else words[:_WORDS_SHOWN].rstrip() + "..."
            return RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message=(
                    f"The model replied {WORDS_ONLY_LIMIT} times in a row without taking any action, "
                    f'so the run stopped. Its last reply: "{shown or "(no text)"}"'
                ),
                terminal=True,
                recoverable=False,
                stop_reason=StopReason.MODEL_ONLY_TALKED,
                failure_category=FailureCategory.MODEL_ONLY_TALKED,
                failure_stage=LoopStage.CHOOSE_ACTION,
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.ADVANCE,
            message="The model sees the result and chooses the next step.",
        )
