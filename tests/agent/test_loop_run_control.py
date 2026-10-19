"""Per-run step serialization and the trust-gated bootstrap warmup."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.trust_gate import TrustGate
from operon.models.capture import CaptureFrame
from operon.models.common import RunResponse, RunStatus, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from tests.fixtures.paths import unique_artifact_dir


def _bare_loop() -> AgentLoop:
    return AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )


async def _record_steps(loop: AgentLoop, timeline: list[str]) -> None:
    async def fake_step(request: StepRequest) -> RunResponse:
        timeline.append(f"start:{request.run_id}")
        await asyncio.sleep(0.01)
        timeline.append(f"end:{request.run_id}")
        return RunResponse(run_id=request.run_id, status=RunStatus.RUNNING, intent="x", step_count=1)

    loop._step_run_locked = fake_step  # type: ignore[method-assign]


async def test_concurrent_steps_on_one_run_do_not_interleave() -> None:
    loop, timeline = _bare_loop(), []
    await _record_steps(loop, timeline)

    await asyncio.gather(loop.step_run(StepRequest(run_id="a")), loop.step_run(StepRequest(run_id="a")))

    assert timeline == ["start:a", "end:a", "start:a", "end:a"]


async def test_steps_on_different_runs_still_overlap() -> None:
    loop, timeline = _bare_loop(), []
    await _record_steps(loop, timeline)

    await asyncio.gather(loop.step_run(StepRequest(run_id="a")), loop.step_run(StepRequest(run_id="b")))

    assert timeline[:2] == ["start:a", "start:b"]


async def test_run_locks_are_released_after_steps() -> None:
    loop, timeline = _bare_loop(), []
    await _record_steps(loop, timeline)

    await loop.step_run(StepRequest(run_id="a"))

    assert "a" not in loop._run_locks  # weak map: no lock outlives its step


# ── Bootstrap warmup ─────────────────────────────────────────────────────────


def _debug() -> ModelDebugArtifacts:
    return ModelDebugArtifacts(prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="d")


def _first_step_loop(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentLoop, AsyncMock]:
    run_root = unique_artifact_dir("test-warmup")
    state = AgentState(run_id="run-1", intent="Find the docs", status=RunStatus.RUNNING)
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=10)
    executed = ExecutedAction(action=action, success=True, detail="waited")
    run_store = SimpleNamespace(
        root_dir=str(run_root),
        get_run=AsyncMock(return_value=state),
        update_state=AsyncMock(return_value=state),
        set_status=AsyncMock(return_value=state),
        before_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "before.png"),
        after_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "after.png"),
        run_log_path=lambda run_id: str(run_root / run_id / "run.jsonl"),
    )
    execute = AsyncMock(return_value=executed)
    loop = AgentLoop(
        capture_service=SimpleNamespace(
            capture=AsyncMock(return_value=CaptureFrame(artifact_path="f.png", width=10, height=10, mime_type="image/png"))
        ),
        perception_service=SimpleNamespace(
            perceive=AsyncMock(
                return_value=ScreenPerception(summary="s", page_hint="unknown", capture_artifact_path="f.png", visible_elements=[])
            ),
            latest_debug_artifacts=_debug,
        ),
        run_store=run_store,
        policy_service=SimpleNamespace(
            choose_action=AsyncMock(return_value=PolicyDecision(action=action, rationale="w", confidence=0.9, active_subgoal="w")),
            latest_debug_artifacts=_debug,
        ),
        executor=SimpleNamespace(execute=execute),
        verifier_service=SimpleNamespace(
            verify=AsyncMock(
                return_value=VerificationResult(
                    status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="c"
                )
            )
        ),
        recovery_manager=SimpleNamespace(
            recover=AsyncMock(return_value=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="m", retry_after_ms=10))
        ),
    )
    # Exercise the real (non-safe-mode) first step, minus its 2.5s hydrate sleep.
    monkeypatch.setattr(AgentLoop, "_test_safe_mode_enabled", staticmethod(lambda: False))
    real_sleep = asyncio.sleep
    monkeypatch.setattr("operon.agent.loop.asyncio.sleep", lambda s: real_sleep(0))
    return loop, execute


def _navigations(execute: AsyncMock) -> list[str]:
    return [c.args[0].url for c in execute.call_args_list if c.args[0].action_type is ActionType.NAVIGATE]


async def test_warmup_navigation_respects_trust_gate_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute = _first_step_loop(monkeypatch)
    loop._trust_gate = TrustGate(enabled=True, deny=(), confirm=(), allow_domains=("example.com",))

    await loop.step_run(StepRequest(run_id="run-1"))

    assert "https://www.google.com" not in _navigations(execute)


async def test_warmup_navigation_runs_when_gate_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute = _first_step_loop(monkeypatch)

    await loop.step_run(StepRequest(run_id="run-1"))

    assert "https://www.google.com" in _navigations(execute)


# ── Pre-execution gates ──────────────────────────────────────────────────────


def _decision(action: AgentAction, confidence: float = 0.9) -> PolicyDecision:
    return PolicyDecision(action=action, rationale="r", confidence=confidence, active_subgoal="s")


@pytest.fixture
def gate_loop(tmp_path, monkeypatch: pytest.MonkeyPatch) -> AgentLoop:
    from operon.agent import control

    monkeypatch.setattr(control, "_confidence_threshold", 0.0)
    monkeypatch.delenv("OPERON_ALLOWED_FILE_ROOTS", raising=False)
    loop = _bare_loop()
    loop.run_store = SimpleNamespace(root_dir=str(tmp_path))
    loop._trust_gate = TrustGate()  # disabled
    return loop


def test_gates_allow_an_ordinary_action(gate_loop: AgentLoop) -> None:
    decision = _decision(AgentAction(action_type=ActionType.WAIT, wait_ms=10))
    assert gate_loop._pre_execution_gate_text(decision, intent="x", run_id="r") is None


def test_confidence_gate_blocks_below_threshold_but_not_stop(gate_loop: AgentLoop, monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.agent import control

    monkeypatch.setattr(control, "_confidence_threshold", 0.8)
    low = _decision(AgentAction(action_type=ActionType.WAIT, wait_ms=10), confidence=0.5)
    stop = _decision(AgentAction(action_type=ActionType.STOP), confidence=0.5)
    assert "below your threshold 80%" in gate_loop._pre_execution_gate_text(low, intent="x", run_id="r")
    assert gate_loop._pre_execution_gate_text(stop, intent="x", run_id="r") is None


def test_trust_gate_deny_and_confirm(gate_loop: AgentLoop) -> None:
    gate_loop._trust_gate = TrustGate(enabled=True, deny=("delete",), confirm=("purchase",))
    deny = _decision(AgentAction(action_type=ActionType.TYPE, text="delete account", target_element_id="e"))
    confirm = _decision(AgentAction(action_type=ActionType.TYPE, text="purchase now", target_element_id="e"))
    assert gate_loop._pre_execution_gate_text(deny, intent="x", run_id="r").startswith("Blocked by policy")
    assert gate_loop._pre_execution_gate_text(confirm, intent="x", run_id="r").startswith("High-risk action")


def test_file_guard_blocks_unnamed_path(gate_loop: AgentLoop) -> None:
    unnamed = _decision(AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/etc/hosts"))
    named = _decision(AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/tmp/a.txt"))
    assert "/etc/hosts" in gate_loop._pre_execution_gate_text(unnamed, intent="save it", run_id="r")
    assert gate_loop._pre_execution_gate_text(named, intent="save it to /tmp/a.txt", run_id="r") is None


def test_first_failing_gate_wins(gate_loop: AgentLoop, monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.agent import control

    monkeypatch.setattr(control, "_confidence_threshold", 0.95)
    decision = _decision(AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/etc/hosts"), confidence=0.5)
    assert "below your threshold" in gate_loop._pre_execution_gate_text(decision, intent="x", run_id="r")
