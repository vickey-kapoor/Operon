"""Resuming a gate pause approves the paused action once, instead of re-pausing on it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.trust_gate import TrustGate
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from tests.fixtures.paths import unique_artifact_dir

PURCHASE = AgentAction(action_type=ActionType.TYPE, text="purchase now", target_element_id="buy")


def _debug() -> ModelDebugArtifacts:
    return ModelDebugArtifacts(prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="d")


def _loop(monkeypatch: pytest.MonkeyPatch, actions: list[AgentAction]) -> tuple[AgentLoop, AsyncMock, AgentState]:
    from operon.agent import control, hitl

    monkeypatch.setattr(hitl, "notify_desktop", lambda **_: None)
    monkeypatch.setattr(hitl, "post_hitl_webhook", AsyncMock())
    monkeypatch.setattr(hitl, "start_escalation_timer", AsyncMock())
    monkeypatch.setattr(control, "_confidence_threshold", 0.0)

    run_root = unique_artifact_dir("test-gate-resume")
    state = AgentState(run_id="run-1", intent="Buy the item", status=RunStatus.RUNNING)

    async def set_status(_run_id: str, status: RunStatus) -> AgentState:
        state.status = status
        return state

    async def update_state(_run_id: str, _perception) -> AgentState:
        return state

    run_store = SimpleNamespace(
        root_dir=str(run_root),
        get_run=AsyncMock(return_value=state),
        update_state=update_state,
        save_state=AsyncMock(),
        set_status=set_status,
        before_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "before.png"),
        after_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "after.png"),
        run_log_path=lambda run_id: str(run_root / run_id / "run.jsonl"),
    )
    decisions = [PolicyDecision(action=a, rationale="r", confidence=0.9, active_subgoal="s") for a in actions]
    execute = AsyncMock(side_effect=lambda action: ExecutedAction(action=action, success=True, detail="ok"))
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
        policy_service=SimpleNamespace(choose_action=AsyncMock(side_effect=decisions), latest_debug_artifacts=_debug),
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
    loop._trust_gate = TrustGate(enabled=True, confirm=("purchase",))
    return loop, execute, state


def _executed_types(execute: AsyncMock) -> list[str]:
    return [c.args[0].text for c in execute.call_args_list if c.args[0].action_type is ActionType.TYPE]


async def test_resume_approves_the_paused_action_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [PURCHASE, PURCHASE])

    first = await loop.step_run(StepRequest(run_id="run-1"))
    assert first.status is RunStatus.WAITING_FOR_USER
    assert _executed_types(execute) == []

    await loop.resume_run("run-1")
    await loop.step_run(StepRequest(run_id="run-1"))

    assert _executed_types(execute) == ["purchase now"]
    assert state.status is not RunStatus.WAITING_FOR_USER


async def test_approval_is_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [PURCHASE, PURCHASE, PURCHASE])

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    await loop.step_run(StepRequest(run_id="run-1"))  # approved: runs
    third = await loop.step_run(StepRequest(run_id="run-1"))  # a fresh repeat is gated again

    assert _executed_types(execute) == ["purchase now"]
    assert third.status is RunStatus.WAITING_FOR_USER


async def test_resuming_with_a_correction_does_not_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, _ = _loop(monkeypatch, [PURCHASE, PURCHASE])

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1", approve_gated_action=False)
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert _executed_types(execute) == []
    assert second.status is RunStatus.WAITING_FOR_USER


async def test_approval_does_not_carry_to_a_different_action(monkeypatch: pytest.MonkeyPatch) -> None:
    other = AgentAction(action_type=ActionType.TYPE, text="purchase the upgrade too", target_element_id="buy")
    loop, execute, _ = _loop(monkeypatch, [PURCHASE, other])

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert _executed_types(execute) == []
    assert second.status is RunStatus.WAITING_FOR_USER


async def test_each_gate_needs_its_own_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.agent import control

    loop, execute, _ = _loop(monkeypatch, [PURCHASE, PURCHASE, PURCHASE])
    monkeypatch.setattr(control, "_confidence_threshold", 0.95)  # decisions carry 0.9

    first = await loop.step_run(StepRequest(run_id="run-1"))  # confidence gate
    await loop.resume_run("run-1")
    second = await loop.step_run(StepRequest(run_id="run-1"))  # confidence approved; trust gate still asks
    await loop.resume_run("run-1")
    await loop.step_run(StepRequest(run_id="run-1"))  # both approved: runs

    assert first.status is RunStatus.WAITING_FOR_USER
    assert second.status is RunStatus.WAITING_FOR_USER
    assert _executed_types(execute) == ["purchase now"]


async def test_file_guard_cannot_be_approved_by_resuming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_ALLOWED_FILE_ROOTS", raising=False)
    read = AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/etc/hosts")
    loop, execute, _ = _loop(monkeypatch, [read, read])

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert second.status is RunStatus.WAITING_FOR_USER
    assert not [c for c in execute.call_args_list if c.args[0].action_type is ActionType.READ_TEXT]


async def test_resume_after_coordinate_thrash_runs_the_click(monkeypatch: pytest.MonkeyPatch) -> None:
    click = AgentAction(action_type=ActionType.CLICK, x=50, y=60)
    loop, execute, state = _loop(monkeypatch, [click, click, click])
    state.target_failure_counts["click:50:60"] = 3

    first = await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    await loop.step_run(StepRequest(run_id="run-1"))  # approved: the click runs

    def clicks() -> int:
        return sum(1 for c in execute.call_args_list if c.args[0].action_type is ActionType.CLICK)

    assert first.status is RunStatus.WAITING_FOR_USER
    assert clicks() >= 1
    ran = clicks()
    third = await loop.step_run(StepRequest(run_id="run-1"))  # still thrashing: asks again
    assert third.status is RunStatus.WAITING_FOR_USER
    assert clicks() == ran
