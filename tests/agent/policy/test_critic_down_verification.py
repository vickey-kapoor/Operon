"""Verification and recovery when the critic is unavailable, and the retry re-gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from operon.agent.actions.retry_hardening import RetryResolution
from operon.agent.policy.recovery import RuleBasedRecoveryManager
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.models.common import FailureCategory, StopReason
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from tests.agent.policy.test_verifier import StubVerificationClient, _prompt_path

CLICK = AgentAction(action_type=ActionType.CLICK, x=10, y=10)


def _service(tmp_path: Path) -> DeterministicVerifierService:
    return DeterministicVerifierService(
        gemini_client=StubVerificationClient(raises=True), prompt_path=_prompt_path(tmp_path)
    )


def _frames(tmp_path: Path, *, changed: bool) -> str:
    Image.new("RGB", (100, 100), "white").save(tmp_path / "before.png")
    after = Image.new("RGB", (100, 100), "white")
    if changed:
        after.paste(Image.new("RGB", (60, 60), "black"), (10, 10))
    after.save(tmp_path / "after.png")
    return str(tmp_path / "after.png")


def _decision(action: AgentAction = CLICK) -> PolicyDecision:
    return PolicyDecision(action=action, rationale="r", confidence=0.9, active_subgoal="s")


async def test_critic_down_and_unchanged_screen_is_uncertain(tmp_path: Path) -> None:
    state = AgentState(run_id="r", intent="Do it", status="running")
    executed = ExecutedAction(action=CLICK, success=True, detail="ok", artifact_path=_frames(tmp_path, changed=False))

    result = await _service(tmp_path).verify(state, _decision(), executed)

    assert result.status is VerificationStatus.UNCERTAIN
    assert "screen did not change" in result.reason
    assert result.critic_fallback_reason == "critic_unavailable_or_unusable"


async def test_critic_down_and_changed_screen_is_success(tmp_path: Path) -> None:
    state = AgentState(run_id="r", intent="Do it", status="running")
    executed = ExecutedAction(action=CLICK, success=True, detail="ok", artifact_path=_frames(tmp_path, changed=True))

    result = await _service(tmp_path).verify(state, _decision(), executed, allow_stable_wait=False)

    assert result.status is VerificationStatus.SUCCESS


async def test_recovery_escalates_on_repeated_dead_clicks_with_critic_down(tmp_path: Path) -> None:
    state = AgentState(run_id="r", intent="Do it", status="running")
    service, manager = _service(tmp_path), RuleBasedRecoveryManager()
    strategies = []
    for _ in range(3):
        executed = ExecutedAction(action=CLICK, success=True, detail="ok", artifact_path=_frames(tmp_path, changed=False))
        verification = await service.verify(state, _decision(), executed)
        recovery = await manager.recover(state, _decision(), executed, verification)
        strategies.append(recovery.strategy)

    assert strategies[0] is RecoveryStrategy.RETRY_SAME_STEP
    assert strategies[1] is RecoveryStrategy.RETRY_DIFFERENT_TACTIC
    assert strategies[2] is RecoveryStrategy.CONTEXT_RESET


async def test_stop_after_only_unverified_actions_is_rejected(tmp_path: Path) -> None:
    stop = AgentAction(action_type=ActionType.STOP)
    state = AgentState(run_id="r", intent="Do it", status="running")
    state.action_history.append(ExecutedAction(action=CLICK, success=True, detail="ok"))
    state.verification_history.append(
        VerificationResult(
            status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="u"
        )
    )
    executed = ExecutedAction(action=stop, success=True, detail="stop")

    result = await _service(tmp_path).verify(state, _decision(stop), executed)

    assert result.status is VerificationStatus.FAILURE
    assert "verified" in result.reason


async def test_stop_rejected_when_a_recent_step_failed_even_if_last_did_not(tmp_path: Path) -> None:
    stop = AgentAction(action_type=ActionType.STOP)
    state = AgentState(run_id="r", intent="Do it", status="running")
    state.action_history.append(ExecutedAction(action=CLICK, success=True, detail="ok"))
    for status in (VerificationStatus.SUCCESS, VerificationStatus.FAILURE, VerificationStatus.UNCERTAIN):
        state.verification_history.append(
            VerificationResult(status=status, expected_outcome_met=False, stop_condition_met=False, reason="x")
        )

    result = await _service(tmp_path).verify(state, _decision(stop), ExecutedAction(action=stop, success=True, detail="s"))

    assert result.status is VerificationStatus.FAILURE
    assert result.stop_reason is not StopReason.TASK_COMPLETED


async def test_stop_accepted_after_a_verified_success(tmp_path: Path) -> None:
    stop = AgentAction(action_type=ActionType.STOP)
    state = AgentState(run_id="r", intent="Do it", status="running")
    state.action_history.append(ExecutedAction(action=CLICK, success=True, detail="ok"))
    state.verification_history.append(
        VerificationResult(
            status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="ok"
        )
    )

    result = await _service(tmp_path).verify(state, _decision(stop), ExecutedAction(action=stop, success=True, detail="s"))

    assert result.status is VerificationStatus.SUCCESS


def _stale(action: AgentAction) -> ExecutedAction:
    return ExecutedAction(
        action=action, success=False, detail="stale", failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION
    )


async def test_regrounded_retry_onto_a_confirm_target_is_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.agent.test_trust_gate_resume import PURCHASE, _loop

    safe = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box")
    loop, execute, state = _loop(monkeypatch, [safe])
    calls: list[AgentAction] = []

    async def run(action: AgentAction) -> ExecutedAction:
        calls.append(action)
        return _stale(action)

    execute.side_effect = run
    loop._resolve_retry_action = lambda **_: RetryResolution(action=PURCHASE, trace=None)

    _, executed = await loop._execute_with_hardening(
        record=state, state=state, perception=None, decision_action=safe, step_index=1
    )

    assert calls == [safe]  # only the first attempt ran
    assert executed.success is False
    assert "Retry not executed" in executed.detail


async def test_regrounded_retry_that_stays_allowed_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.agent.test_trust_gate_resume import _loop

    safe = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box")
    other = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box2")
    loop, execute, state = _loop(monkeypatch, [safe])
    calls: list[AgentAction] = []

    async def run(action: AgentAction) -> ExecutedAction:
        calls.append(action)
        if len(calls) == 1:
            return _stale(action)
        return ExecutedAction(action=action, success=True, detail="ok")

    execute.side_effect = run
    loop._resolve_retry_action = lambda **_: RetryResolution(action=other, trace=None)

    _, executed = await loop._execute_with_hardening(
        record=state, state=state, perception=None, decision_action=safe, step_index=1
    )

    assert calls == [safe, other]
    assert executed.success is True
