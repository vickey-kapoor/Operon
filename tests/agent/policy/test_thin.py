"""The harness: the model decides, the executor runs it once, success is checked at the end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import (
    OwnDoneVerifier,
    ThinPolicy,
    ThinRecovery,
)
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


@pytest.mark.asyncio
async def test_thin_policy_returns_the_backend_decision_unchanged(tmp_path) -> None:
    decision = _decision(AgentAction(action_type=ActionType.CLICK, x=10, y=20))
    delegate = Mock()
    delegate.choose_action = AsyncMock(return_value=decision)
    state = _state(tmp_path)

    assert await ThinPolicy(delegate).choose_action(state, state.observation_history[-1]) is decision


@pytest.mark.asyncio
async def test_an_ordinary_step_passes_through(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)

    result = await OwnDoneVerifier().verify(_state(tmp_path), _decision(action), _executed(action))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is False
    assert result.unchecked  # the step list must not call it checked


@pytest.mark.asyncio
async def test_a_failed_action_is_left_for_the_model_not_retried(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    executed = _executed(action, success=False, detail="Click did nothing").model_copy(
        update={"failure_category": FailureCategory.EXECUTION_ERROR}
    )

    result = await OwnDoneVerifier().verify(_state(tmp_path), _decision(action), executed)

    # UNCERTAIN, not FAILURE: the loop's in-step retry only runs on FAILURE.
    assert result.status is VerificationStatus.UNCERTAIN
    assert "Click did nothing" in result.reason


@pytest.mark.asyncio
async def test_the_models_own_done_ends_the_run_unchecked(tmp_path) -> None:
    stop = AgentAction(action_type=ActionType.STOP)

    result = await OwnDoneVerifier().verify(_state(tmp_path), _decision(stop, "Archived it."), _executed(stop))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is True
    assert result.stop_reason is StopReason.TASK_COMPLETED
    assert "Archived it." in result.reason
    assert result.unchecked


@pytest.mark.asyncio
async def test_thin_recovery_never_resets_the_screen(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = await OwnDoneVerifier().verify(
        _state(tmp_path), _decision(action), _executed(action, success=False, detail="x")
    )

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(action), _executed(action, success=False), failed)

    assert recovery.strategy is RecoveryStrategy.ADVANCE


@pytest.mark.asyncio
async def test_thin_recovery_stops_on_the_models_done(tmp_path) -> None:
    stop = AgentAction(action_type=ActionType.STOP)
    confirmed = await OwnDoneVerifier().verify(_state(tmp_path), _decision(stop), _executed(stop))

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(stop), _executed(stop), confirmed)

    assert recovery.strategy is RecoveryStrategy.STOP
    assert recovery.stop_reason is StopReason.TASK_COMPLETED


def _loop() -> AgentLoop:
    return AgentLoop(
        capture_service=Mock(capture=AsyncMock()),
        perception_service=Mock(perceive=AsyncMock()),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )


@pytest.mark.asyncio
async def test_the_loop_runs_a_failed_action_once_without_a_second_look(tmp_path) -> None:
    loop = _loop()
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = _executed(action, success=False, detail="target moved").model_copy(
        update={"failure_category": FailureCategory.CLICK_NO_EFFECT}
    )
    loop.executor_adapter = Mock(execute=AsyncMock(return_value=failed))
    loop._persist_execution_trace = lambda run_id, step, executed: executed
    state = _state(tmp_path)

    _, executed = await loop._execute_action(
        record=state, state=state, decision_action=action, step_index=1, use_unified_executor=True,
    )

    assert executed is failed
    loop.executor_adapter.execute.assert_awaited_once()
    loop.capture_service.capture.assert_not_awaited()
    loop.perception_service.perceive.assert_not_awaited()


def test_the_loop_has_no_harness_switch() -> None:
    import inspect

    assert "harness_profile" not in inspect.signature(AgentLoop.__init__).parameters
    assert not hasattr(_loop(), "thin_harness")
