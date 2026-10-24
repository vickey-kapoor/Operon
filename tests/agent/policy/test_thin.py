"""Thin harness profile: the model decides, the executor runs it once, success is checked at the end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import (
    FULL,
    THIN,
    FinalCheckVerifier,
    ThinPolicy,
    ThinRecovery,
    harness_profile,
)
from operon.clients.anthropic import AnthropicClientError
from operon.models.common import FailureCategory, RunStatus, StopReason
from operon.models.execution import ExecutedAction
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationStatus


def _state(tmp_path: Path) -> AgentState:
    screen = tmp_path / "before.png"
    screen.write_bytes(b"png")
    perception = ScreenPerception(
        summary="Inbox", page_hint=PageHint.UNKNOWN, capture_artifact_path=str(screen), elements_reported=False,
    )
    return AgentState(
        run_id="run-thin", intent="Archive the newest email", status=RunStatus.RUNNING,
        observation_history=[perception],
    )


def _decision(action: AgentAction, rationale: str = "Clicking archive.") -> PolicyDecision:
    return PolicyDecision(action=action, rationale=rationale, confidence=0.7, active_subgoal="archive")


def _executed(action: AgentAction, *, success: bool = True, detail: str = "ok") -> ExecutedAction:
    return ExecutedAction(action=action, success=success, detail=detail)


@pytest.mark.parametrize(
    ("requested", "backend", "expected"),
    [
        ("thin", "claude_computer_use", THIN),
        ("THIN", "claude_computer_use", THIN),
        ("thin", "computer_use", FULL),
        ("thin", "json", FULL),
        (None, "claude_computer_use", FULL),
    ],
)
def test_thin_profile_only_applies_to_claude(monkeypatch, requested, backend, expected) -> None:
    if requested is None:
        monkeypatch.delenv("OPERON_HARNESS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("OPERON_HARNESS_PROFILE", requested)
    assert harness_profile(backend) == expected


@pytest.mark.asyncio
async def test_thin_policy_returns_the_backend_decision_unchanged(tmp_path) -> None:
    decision = _decision(AgentAction(action_type=ActionType.CLICK, x=10, y=20))
    delegate = Mock()
    delegate.choose_action = AsyncMock(return_value=decision)
    state = _state(tmp_path)

    assert await ThinPolicy(delegate).choose_action(state, state.observation_history[-1]) is decision
    assert ThinPolicy(delegate).supports_resampling() is False


@pytest.mark.asyncio
async def test_an_ordinary_step_is_not_judged_by_a_model(tmp_path) -> None:
    client = Mock(generate_verification=AsyncMock())
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)

    result = await FinalCheckVerifier(client).verify(_state(tmp_path), _decision(action), _executed(action))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is False
    client.generate_verification.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_action_is_left_for_the_model_not_retried(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    executed = _executed(action, success=False, detail="Click did nothing").model_copy(
        update={"failure_category": FailureCategory.EXECUTION_ERROR}
    )

    result = await FinalCheckVerifier(Mock()).verify(_state(tmp_path), _decision(action), executed)

    # UNCERTAIN, not FAILURE: the loop's in-step retry only runs on FAILURE.
    assert result.status is VerificationStatus.UNCERTAIN
    assert "Click did nothing" in result.reason


@pytest.mark.asyncio
async def test_done_is_confirmed_once_against_the_task(tmp_path) -> None:
    client = Mock(generate_verification=AsyncMock(return_value='{"complete": true, "reason": "Inbox shows no email."}'))
    stop = AgentAction(action_type=ActionType.STOP)
    state = _state(tmp_path)

    result = await FinalCheckVerifier(client).verify(state, _decision(stop, "Archived it."), _executed(stop))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is True
    assert result.stop_reason is StopReason.TASK_COMPLETED
    prompt, screenshot = client.generate_verification.await_args.args
    assert "Archive the newest email" in prompt and "Archived it." in prompt
    assert screenshot == state.observation_history[-1].capture_artifact_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        AsyncMock(return_value='{"complete": false, "reason": "The email is still in the inbox."}'),
        AsyncMock(return_value="I think so?"),
        AsyncMock(side_effect=AnthropicClientError("Anthropic HTTP error 529")),
    ],
)
async def test_an_unconfirmed_done_goes_back_to_the_model(tmp_path, reply) -> None:
    hints: list[tuple[str, str]] = []
    verifier = FinalCheckVerifier(Mock(generate_verification=reply), on_rejected=lambda run, hint: hints.append((run, hint)))
    stop = AgentAction(action_type=ActionType.STOP)

    result = await verifier.verify(_state(tmp_path), _decision(stop, "Done."), _executed(stop))

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.stop_condition_met is False
    assert len(hints) == 1 and hints[0][0] == "run-thin"
    assert "not done yet" in hints[0][1]


@pytest.mark.asyncio
async def test_thin_recovery_never_resets_the_screen(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = await FinalCheckVerifier(Mock()).verify(
        _state(tmp_path), _decision(action), _executed(action, success=False, detail="x")
    )

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(action), _executed(action, success=False), failed)

    assert recovery.strategy is RecoveryStrategy.ADVANCE


@pytest.mark.asyncio
async def test_thin_recovery_stops_on_a_confirmed_success(tmp_path) -> None:
    client = Mock(generate_verification=AsyncMock(return_value='{"complete": true, "reason": "Done."}'))
    stop = AgentAction(action_type=ActionType.STOP)
    confirmed = await FinalCheckVerifier(client).verify(_state(tmp_path), _decision(stop), _executed(stop))

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(stop), _executed(stop), confirmed)

    assert recovery.strategy is RecoveryStrategy.STOP
    assert recovery.stop_reason is StopReason.TASK_COMPLETED


def _loop(profile: str) -> AgentLoop:
    return AgentLoop(
        capture_service=Mock(capture=AsyncMock()),
        perception_service=Mock(perceive=AsyncMock()),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
        harness_profile=profile,
    )


@pytest.mark.asyncio
async def test_thin_loop_runs_a_failed_action_once_without_a_second_look(tmp_path) -> None:
    loop = _loop(THIN)
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = _executed(action, success=False, detail="target moved").model_copy(
        update={"failure_category": FailureCategory.CLICK_NO_EFFECT}
    )
    loop.executor_adapter = Mock(execute=AsyncMock(return_value=failed))
    loop._persist_execution_trace = lambda run_id, step, executed: executed
    state = _state(tmp_path)

    _, executed = await loop._execute_with_hardening(
        record=state, state=state, perception=state.observation_history[-1],
        decision_action=action, step_index=1, use_unified_executor=True,
    )

    assert executed is failed
    loop.executor_adapter.execute.assert_awaited_once()
    loop.capture_service.capture.assert_not_awaited()
    loop.perception_service.perceive.assert_not_awaited()


def test_the_profile_reaches_the_loop() -> None:
    assert _loop(THIN).thin_harness is True
    assert _loop(FULL).thin_harness is False
