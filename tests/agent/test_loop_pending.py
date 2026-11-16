"""PENDING verification back-off, timeout downgrade and SESSION_RESET recovery (loop.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch as _patch

import pytest

from operon.agent.perception import PerceptionLowQualityError
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory, RunStatus
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


@pytest.mark.asyncio
async def test_pending_resolves_to_success_after_n_polls(tmp_path: Path) -> None:
    pending, success = VerificationStatus.PENDING, VerificationStatus.SUCCESS
    loop, state, kwargs = _pending_loop(tmp_path, [pending, pending, success])

    with _patch("operon.agent.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await loop._wait_for_page_load(**kwargs)

    assert result.status is success
    assert result.patience_retries == 3
    assert [c.args[0] for c in sleep.await_args_list] == [2.0, 4.0, 8.0]
    assert loop.verifier_service.verify.await_count == 3
    assert len(state.observation_history) == 3


@pytest.mark.asyncio
async def test_pending_stops_polling_as_soon_as_it_resolves(tmp_path: Path) -> None:
    loop, _state, kwargs = _pending_loop(tmp_path, [VerificationStatus.SUCCESS])

    with _patch("operon.agent.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await loop._wait_for_page_load(**kwargs)

    assert result.patience_retries == 1
    assert [c.args[0] for c in sleep.await_args_list] == [2.0]


@pytest.mark.asyncio
async def test_pending_until_budget_exhausted_downgrades_to_uncertain(tmp_path: Path) -> None:
    pending = VerificationStatus.PENDING
    loop, _state, kwargs = _pending_loop(tmp_path, [pending, pending, pending])

    with _patch("operon.agent.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await loop._wait_for_page_load(**kwargs)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.expected_outcome_met is False
    assert result.stop_condition_met is False
    assert result.failure_category is FailureCategory.UNCERTAIN_SCREEN_STATE
    assert result.patience_retries == 3
    assert [c.args[0] for c in sleep.await_args_list] == [2.0, 4.0, 8.0]
    assert loop.verifier_service.verify.await_count == 3


@pytest.mark.asyncio
async def test_pending_blank_perception_keeps_waiting_without_verifying(tmp_path: Path) -> None:
    perceive = AsyncMock(side_effect=PerceptionLowQualityError("blank"))
    loop, state, kwargs = _pending_loop(tmp_path, [], perceive=perceive)

    with _patch("operon.agent.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await loop._wait_for_page_load(**kwargs)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.patience_retries == 3
    assert sleep.await_count == 3
    loop.verifier_service.verify.assert_not_awaited()
    assert state.observation_history == []


@pytest.mark.asyncio
async def test_pending_reverify_forbids_stable_wait(tmp_path: Path) -> None:
    """Each poll asks the verifier not to answer STABLE_WAIT, which would escape the loop."""
    modes: list[bool] = []

    async def verify(_state, _decision, _executed, allow_stable_wait: bool = True):
        modes.append(allow_stable_wait)
        # A verifier that honors the flag: it would say STABLE_WAIT if allowed.
        status = VerificationStatus.STABLE_WAIT if allow_stable_wait else VerificationStatus.SUCCESS
        return _result(status)

    loop, _state, kwargs = _pending_loop(tmp_path, [])
    loop.verifier_service = SimpleNamespace(verify=verify)

    with _patch("operon.agent.loop.asyncio.sleep", new_callable=AsyncMock):
        result = await loop._wait_for_page_load(**kwargs)

    assert modes == [False]
    assert result.status is VerificationStatus.SUCCESS


@pytest.mark.asyncio
async def test_verify_attempt_routes_pending_through_page_load_wait(tmp_path: Path) -> None:
    loop, state, kwargs = _pending_loop(tmp_path, [])
    resolved = _result(VerificationStatus.SUCCESS)
    loop.verifier_service = SimpleNamespace(verify=AsyncMock(return_value=_result(VerificationStatus.PENDING)))
    loop._wait_for_page_load = AsyncMock(return_value=resolved)
    loop._wait_for_ui_stable = AsyncMock()
    loop._resolve_model_debug_artifacts = Mock(return_value=None)
    ctx = SimpleNamespace(
        last_attempt=kwargs["executed_action"],
        state=state,
        decision=kwargs["decision"],
        record=state,
        before_artifact_path=kwargs["before_artifact_path"],
        run_id="run-1",
        step_index=1,
        record_stage=lambda *_a: 0.0,
    )

    await loop._verify_attempt(ctx)

    assert ctx.verification is resolved
    loop._wait_for_page_load.assert_awaited_once()
    loop._wait_for_ui_stable.assert_not_awaited()


# -- SESSION_RESET (4th consecutive failure on a subgoal) ---------------------------


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
