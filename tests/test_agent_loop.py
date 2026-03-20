"""Agent loop orchestration tests using async mocks."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agent.loop import AgentLoop
from src.models.capture import CaptureFrame
from src.models.common import RunStatus, RunTaskRequest, StepRequest
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.asyncio
async def test_agent_loop_calls_stages_in_required_order() -> None:
    call_order: list[str] = []

    state = AgentState(run_id="run-1", intent="Create draft", status=RunStatus.PENDING)
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Inbox visible",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait for page.", confidence=0.1, active_subgoal="wait for inbox")
    executed = ExecutedAction(action=action, success=True, detail="waited")
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Continue",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="Try again", retry_after_ms=1000)
    final_state = state.model_copy(update={"status": RunStatus.RUNNING, "step_count": 1})

    capture_service = SimpleNamespace(
        capture=AsyncMock(side_effect=lambda run_state: call_order.append("capture") or frame)
    )
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=lambda captured, run_state: call_order.append("perceive") or perception),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-1/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-1/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-1/step_1/perception_parsed.json",
        ),
    )
    run_store = SimpleNamespace(
        get_run=AsyncMock(return_value=state),
        set_status=AsyncMock(return_value=final_state),
        update_state=AsyncMock(side_effect=lambda run_id, seen: call_order.append("update_state") or final_state),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(side_effect=lambda run_state, seen: call_order.append("choose_action") or decision),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-1/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-1/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-1/step_1/policy_decision.json",
        ),
    )
    browser_executor = SimpleNamespace(
        execute=AsyncMock(side_effect=lambda chosen: call_order.append("execute") or executed)
    )
    verifier_service = SimpleNamespace(
        verify=AsyncMock(side_effect=lambda run_state, chosen, done: call_order.append("verify") or verification)
    )
    recovery_manager = SimpleNamespace(
        recover=AsyncMock(
            side_effect=lambda run_state, chosen, done, checked: call_order.append("recover") or recovery
        )
    )
    run_root = _local_test_dir("test-runs-order")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        browser_executor=browser_executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert call_order == [
        "capture",
        "perceive",
        "update_state",
        "choose_action",
        "execute",
        "verify",
        "recover",
    ]
    assert response.run_id == "run-1"
    payload = json.loads((run_root / "run-1" / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["step_id"] == "step_1"
    assert payload["perception_debug"]["prompt_artifact_path"].endswith("perception_prompt.txt")


@pytest.mark.asyncio
async def test_agent_loop_delegates_stage_inputs_and_outputs() -> None:
    initial_state = AgentState(run_id="run-2", intent="Create draft", status=RunStatus.PENDING)
    updated_state = initial_state.model_copy(update={"step_count": 1, "status": RunStatus.RUNNING})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Compose button visible",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[],
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.8, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=True,
        reason="Draft ready",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.STOP, message="unused")
    final_state = updated_state.model_copy(update={"status": RunStatus.SUCCEEDED})

    capture_service = Mock()
    capture_service.capture = AsyncMock(return_value=frame)
    perception_service = Mock()
    perception_service.perceive = AsyncMock(return_value=perception)
    perception_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-2/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-2/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-2/step_1/perception_parsed.json",
        )
    )
    run_store = Mock()
    run_store.get_run = AsyncMock(return_value=initial_state)
    run_store.set_status = AsyncMock(return_value=final_state)
    run_store.update_state = AsyncMock(return_value=updated_state)
    policy_service = Mock()
    policy_service.choose_action = AsyncMock(return_value=decision)
    policy_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-2/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-2/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-2/step_1/policy_decision.json",
        )
    )
    browser_executor = Mock()
    browser_executor.execute = AsyncMock(return_value=executed)
    verifier_service = Mock()
    verifier_service.verify = AsyncMock(return_value=verification)
    recovery_manager = Mock()
    recovery_manager.recover = AsyncMock(return_value=recovery)
    run_root = _local_test_dir("test-runs-delegate")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        browser_executor=browser_executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id="run-2"))

    capture_service.capture.assert_awaited_once_with(initial_state)
    remapped_executed = executed.model_copy(
        update={"artifact_path": str(run_root / "run-2" / "step_1" / "after.png")}
    )
    perception_service.perceive.assert_awaited_once_with(frame, initial_state)
    run_store.update_state.assert_awaited_once_with("run-2", perception)
    policy_service.choose_action.assert_awaited_once_with(updated_state, perception)
    browser_executor.execute.assert_awaited_once_with(action)
    verifier_service.verify.assert_awaited_once_with(updated_state, decision, remapped_executed)
    recovery_manager.recover.assert_awaited_once_with(updated_state, decision, remapped_executed, verification)
    assert response.status is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_agent_loop_start_run_delegates_to_store_only() -> None:
    created = AgentState(run_id="run-3", intent="Create draft", status=RunStatus.PENDING)
    run_store = Mock()
    run_store.create_run = Mock(return_value=created)

    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        browser_executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    response = await loop.start_run(RunTaskRequest(intent="Create a Gmail draft and stop before send."))

    run_store.create_run.assert_called_once()
    assert response.run_id == "run-3"
    assert response.status is RunStatus.PENDING
