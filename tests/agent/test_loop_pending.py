"""SESSION_RESET recovery (loop.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.perception import ScreenPerception
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from tests.agent.test_loop import _loop, _settle_inputs


def _result(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(
        status=status, expected_outcome_met=False, stop_condition_met=False, reason=status.value
    )


def _pending_loop(tmp_path: Path, statuses, *, perceive=None):
    """Loop wired for _wait_for_page_load; the verifier replays ``statuses`` in order."""
    _after, fresh, state, decision, executed, _ = _settle_inputs(tmp_path)
    loop = _loop()
    loop.capture_service = SimpleNamespace(
        capture=AsyncMock(return_value=CaptureFrame(artifact_path=str(fresh), width=480, height=270))
    )
    loaded = ScreenPerception(
        summary="loaded", page_hint="unknown", capture_artifact_path=str(fresh), visible_elements=[]
    )
    loop._perceive_with_liveness_retry = perceive or AsyncMock(return_value=loaded)
    loop._infer_focused_element = lambda _record, perception: perception
    loop._sync_progress_state_with_perception = Mock()
    loop.verifier_service = SimpleNamespace(
        verify=AsyncMock(side_effect=[_result(s) for s in statuses])
    )
    kwargs = {
        "record": state,
        "state": state,
        "decision": decision,
        "executed_action": executed,
        "before_artifact_path": str(tmp_path / "run-1" / "step_1" / "before.png"),
    }
    return loop, state, kwargs


def _session_state() -> AgentState:
    return AgentState(
        run_id="run-1", intent="x", status=RunStatus.RUNNING, start_url="https://example.com/start"
    )


@pytest.mark.asyncio
async def test_session_reset_strategy_calls_executor_with_start_url() -> None:
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), context_reset=AsyncMock())
    recovery = RecoveryDecision(strategy=RecoveryStrategy.SESSION_RESET, message="restart")

    await loop._apply_recovery_actions(_session_state(), recovery)

    loop.executor.session_reset.assert_awaited_once_with("https://example.com/start")
    loop.executor.context_reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_strategies_do_not_session_reset() -> None:
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), context_reset=AsyncMock())

    await loop._apply_recovery_actions(
        _session_state(), RecoveryDecision(strategy=RecoveryStrategy.CONTEXT_RESET, message="ctx")
    )
    await loop._apply_recovery_actions(
        _session_state(), RecoveryDecision(strategy=RecoveryStrategy.RETRY_SAME_STEP, message="retry")
    )

    loop.executor.session_reset.assert_not_awaited()
    loop.executor.context_reset.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_reset_failure_is_recorded_not_raised() -> None:
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(side_effect=RuntimeError("browser gone")))
    recovery = RecoveryDecision(strategy=RecoveryStrategy.SESSION_RESET, message="restart")

    await loop._apply_recovery_actions(_session_state(), recovery)  # must not raise

    assert loop._recovered_errors["run-1"]["session_reset"] == 1
