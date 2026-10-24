"""Thin harness profile (``OPERON_HARNESS_PROFILE=thin``) for model-driven backends.

The model decides every action and the executor runs exactly that. Nothing here
replaces, retries or resets an action behind the model's back:

- ``ThinPolicy`` passes the backend's decision through, with no rule engine or
  post-model guards in front of it.
- ``FinalCheckVerifier`` lets a failed action stand as the model's tool result and
  checks success once, against the task, when the model says it is done.
- ``ThinRecovery`` always hands the next step back to the model; the run's step
  budget is the only stop besides a confirmed success.

Safety gates (trust gate, file roots, coordinate-thrash pause, max steps) stay in
the loop and apply to every profile.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from operon.agent._agent_utils import collect_latest_usage
from operon.agent.backends.browser_claude_computer_use import encode_screenshot
from operon.clients.anthropic import AnthropicClientError
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

_FINAL_CHECK_PROMPT = """You are checking whether a computer-use agent finished its task.

Task: {intent}
The agent says it is done: {claim}

The image is the screen the agent saw when it said so. Judge only from the task and
this screen. Reply with one JSON object and nothing else:
{{"complete": true or false, "reason": "one sentence naming what on the screen shows it"}}"""


def final_check_model(claude_model: str | None) -> str:
    """Model for the completion check: ``OPERON_FINAL_CHECK_MODEL``, else the acting model.

    The check is one image and a short answer per claimed finish, so a cheaper
    model (``claude-sonnet-5-5``) can cut its cost without touching the actions.
    """
    return (os.getenv("OPERON_FINAL_CHECK_MODEL") or "").strip() or claude_model or "claude-opus-5-5"


def harness_profile(backend: str) -> str:
    """The harness profile for a backend. Claude computer use runs thin unless
    ``OPERON_HARNESS_PROFILE=full`` asks for the old scaffolding; every other
    backend runs full, since its decisions still need the rules and the critic."""
    requested = (os.getenv("OPERON_HARNESS_PROFILE") or "").strip().lower()
    if backend != "claude_computer_use":
        if requested == THIN:
            logger.warning("OPERON_HARNESS_PROFILE=thin needs backend=claude_computer_use (got %r); using full", backend)
        return FULL
    return FULL if requested == FULL else THIN


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


class FinalCheckVerifier(VerifierService):
    """One success check, when the model says the task is done.

    Every other step passes through: a failed action is reported to the model as
    its tool result by the backend, not retried here. A rejected claim goes back
    to the model as a hint through ``on_rejected`` and the run continues.
    """

    def __init__(
        self, client, *, on_rejected: Callable[[str, str], None] | None = None, effort: str | None = "low"
    ) -> None:
        self.client = client
        self.effort = effort
        self.on_rejected = on_rejected
        self._last_debug_artifacts: ModelDebugArtifacts | None = None

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        *,
        allow_stable_wait: bool = True,
    ) -> VerificationResult:
        self._last_debug_artifacts = None
        if decision.action.action_type is ActionType.STOP:
            return await self._check_completion(state, decision, executed_action)
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
        )

    async def _check_completion(
        self, state: AgentState, decision: PolicyDecision, executed_action: ExecutedAction
    ) -> VerificationResult:
        candidates = [
            executed_action.artifact_path,
            state.observation_history[-1].capture_artifact_path if state.observation_history else None,
        ]
        screenshot = next((path for path in candidates if path and Path(path).exists()), None)
        if screenshot is None:
            return self._unchecked(state, decision, "no final screenshot to check against")
        prompt = _FINAL_CHECK_PROMPT.format(intent=state.intent, claim=decision.rationale)
        image, media_type, _, _ = encode_screenshot(Path(screenshot))
        try:
            raw = await self.client.generate_judgement(prompt, image_data=image, media_type=media_type, effort=self.effort)
        except AnthropicClientError as exc:
            # A check that cannot run must not keep the model going in circles: take
            # its word, say so in the result, and leave grading to the benchmark judge.
            return self._unchecked(state, decision, str(exc))
        self._write_artifacts(Path(screenshot).parent, prompt, raw)
        complete, reason = _parse_verdict(raw)
        if complete:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason=f"Completion check: {reason}",
                stop_reason=StopReason.TASK_COMPLETED,
                critic_model_used=True,
            )
        return self._rejected(state, reason)

    @staticmethod
    def _unchecked(state: AgentState, decision: PolicyDecision, why: str) -> VerificationResult:
        """The check could not run. Rejecting would send the model round in circles
        until the step budget runs out, so take its word, say so in the result, and
        leave grading to the benchmark judge."""
        logger.warning("final check unavailable for run %s, accepting the model's claim: %s", state.run_id[:8], why)
        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=True,
            reason=f"Model says done (completion check unavailable): {decision.rationale}",
            stop_reason=StopReason.TASK_COMPLETED,
            critic_fallback_reason=why[:500] or "unavailable",
        )

    def _rejected(self, state: AgentState, reason: str) -> VerificationResult:
        logger.warning("final check rejected the claim for run %s: %s", state.run_id[:8], reason)
        if self.on_rejected is not None:
            self.on_rejected(
                state.run_id,
                f"A completion check found the task not done yet: {reason} Keep going, or say why it is done.",
            )
        return VerificationResult(
            status=VerificationStatus.UNCERTAIN,
            expected_outcome_met=False,
            stop_condition_met=False,
            reason=f"Completion check rejected the claim: {reason}",
            critic_model_used=True,
        )

    def _write_artifacts(self, step_dir: Path, prompt: str, raw: str) -> None:
        prompt_path = step_dir / "final_check_prompt.txt"
        raw_path = step_dir / "final_check_raw.txt"
        try:
            prompt_path.write_text(prompt, encoding="utf-8")
            raw_path.write_text(raw, encoding="utf-8")
        except OSError:
            return
        usage_path = step_dir / "final_check_usage.json"
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(raw_path),
            usage_artifact_path=str(usage_path),
            usage=collect_latest_usage(self.client, usage_path),
        )


def _parse_verdict(raw: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            reason = str(payload.get("reason") or "no reason given").strip()
            return payload.get("complete") is True, reason
        except (json.JSONDecodeError, AttributeError):
            pass
    return False, f"Unreadable completion check reply: {(raw or '').strip()[:200] or 'empty'}"


class ThinRecovery(RecoveryManager):
    """The model chooses the next step; only a confirmed success ends the run."""

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
