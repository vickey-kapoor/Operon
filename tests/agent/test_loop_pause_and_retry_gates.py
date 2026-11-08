"""Manual pause, correction delivery, retry gating and per-step run selection in the loop."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from operon.agent import control
from operon.models.common import RunStatus, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.verification import VerificationResult, VerificationStatus
from tests.agent.test_trust_gate_resume import PURCHASE, _loop

SAFE = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box")


async def test_pause_stops_a_running_run(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE])

    await loop.pause_run("run-1")
    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert state.status is RunStatus.WAITING_FOR_USER
    assert response.status is RunStatus.WAITING_FOR_USER
    execute.assert_not_called()


async def test_a_step_in_flight_does_not_overwrite_the_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE, SAFE])

    async def pause_mid_step(action: AgentAction) -> ExecutedAction:
        await loop.pause_run("run-1")
        return ExecutedAction(action=action, success=True, detail="ok")

    execute.side_effect = pause_mid_step
    await loop.step_run(StepRequest(run_id="run-1"))
    assert state.status is RunStatus.WAITING_FOR_USER

    await loop.resume_run("run-1")
    execute.side_effect = lambda action: ExecutedAction(action=action, success=True, detail="ok")
    await loop.step_run(StepRequest(run_id="run-1"))
    assert state.status is RunStatus.RUNNING  # resuming clears the pause


async def test_a_correction_reaches_the_backend_before_it_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, _, _ = _loop(monkeypatch, [SAFE])
    order: list[str] = []
    loop.perception_service.add_advisory_hints = lambda hints, source="", run_id="": order.append(f"hint:{hints[0]}")
    perceive = loop.perception_service.perceive

    async def tracked_perceive(*args, **kwargs):
        order.append("perceive")
        return await perceive(*args, **kwargs)

    loop.perception_service.perceive = tracked_perceive
    control.set_override_hint("run-1", "use the second field")

    await loop.step_run(StepRequest(run_id="run-1"))

    assert order[0].startswith("hint:") and "use the second field" in order[0]
    assert "perceive" in order[1:]


async def test_a_retry_decision_passes_the_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE, PURCHASE])
    loop.verifier_service.verify = AsyncMock(
        return_value=VerificationResult(
            status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="no"
        )
    )
    loop.recovery_manager.recover = AsyncMock(
        return_value=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="go on")
    )

    response = await loop.step_run(StepRequest(run_id="run-1"))

    # The gated retry never ran and the run pauses for a human; the attempt that
    # did run is still recorded as the step's action.
    assert response.status is RunStatus.WAITING_FOR_USER
    assert [c.args[0].text for c in execute.call_args_list] == ["hello"]
    assert any(getattr(entry.action, "text", None) == "hello" for entry in state.action_history)


async def test_a_retry_asking_for_help_pauses_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Please solve the CAPTCHA")
    loop, execute, _ = _loop(monkeypatch, [SAFE, ask])
    loop.verifier_service.verify = AsyncMock(
        return_value=VerificationResult(
            status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="no"
        )
    )
    loop.recovery_manager.recover = AsyncMock(
        return_value=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="go on")
    )

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert response.status is RunStatus.WAITING_FOR_USER
    assert [c.args[0].text for c in execute.call_args_list] == ["hello"]


async def test_each_step_selects_its_run_on_the_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, _, _ = _loop(monkeypatch, [SAFE])
    selected: list[str] = []
    loop.executor.set_current_run_id = selected.append
    capture = loop.capture_service.capture

    async def tracked_capture(*args, **kwargs):
        selected.append("capture")
        return await capture(*args, **kwargs)

    loop.capture_service.capture = tracked_capture

    await loop.step_run(StepRequest(run_id="run-1"))

    assert selected[:2] == ["run-1", "capture"]
