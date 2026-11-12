"""Thin harness profile: the model decides, the executor runs it once, success is checked at the end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import (
    FULL,
    THIN,
    OwnDoneVerifier,
    ThinPolicy,
    ThinRecovery,
    harness_profile,
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


@pytest.mark.parametrize(
    ("requested", "backend", "expected"),
    [
        (None, "computer_use", FULL),
        ("thin", "computer_use", THIN),  # Gemini computer use runs thin on request (KAN-67)
        ("full", "computer_use", FULL),
        ("thin", "json", FULL),
    ],
)
def test_gemini_computer_use_runs_full_unless_thin_is_asked_for(monkeypatch, requested, backend, expected) -> None:
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


@pytest.mark.parametrize(("profile", "hinted"), [(THIN, False), (FULL, True)])
def test_the_stagnation_hint_is_full_harness_only_and_scoped_to_its_run(tmp_path, profile, hinted) -> None:
    """Under the thin harness it named a stop action the model may not have, and its
    unscoped bucket could surface in another run's turn."""
    loop = _loop(profile)
    state = _state(tmp_path)
    state.progress_state.no_progress_streak = 3

    loop._inject_stagnation_hint(state)

    add = loop.perception_service.add_advisory_hints
    if hinted:
        add.assert_called_once()
        assert add.call_args.kwargs["run_id"] == "run-thin"
    else:
        add.assert_not_called()


def test_the_profile_reaches_the_loop() -> None:
    assert _loop(THIN).thin_harness is True
    assert _loop(FULL).thin_harness is False
