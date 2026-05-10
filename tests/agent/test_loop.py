"""Agent loop orchestration tests using async mocks."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.agent.loop import AgentLoop
from src.agent.perception import PerceptionError, PerceptionLowQualityError
from src.models.capture import CaptureFrame
from src.models.common import (
    FailureCategory,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from src.models.execution import ExecutedAction, ExecutionAttemptTrace, ExecutionTrace
from src.models.logs import ModelDebugArtifacts
from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.progress import ProgressState
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_store import FileBackedRunStore


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _loop() -> AgentLoop:
    return AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )


@pytest.mark.asyncio
async def test_agent_loop_calls_stages_in_required_order() -> None:
    call_order: list[str] = []

    state = AgentState(run_id="run-1", intent="Create draft", status=RunStatus.PENDING)
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Inbox visible",
        page_hint="unknown",
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
    executor = SimpleNamespace(
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
        executor=executor,
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
        page_hint="unknown",
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
        stop_reason=StopReason.STOP_BEFORE_SEND,
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
    run_store.save_state = AsyncMock()
    policy_service = Mock()
    policy_service.choose_action = AsyncMock(return_value=decision)
    policy_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-2/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-2/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-2/step_1/policy_decision.json",
        )
    )
    executor = Mock()
    executor.execute = AsyncMock(return_value=executed)
    executor.aclose_run = AsyncMock(return_value=1)
    executor.recorded_video_path_for_run = Mock(return_value=Path("runs/run-2/session_video/session.webm"))
    verifier_service = Mock()
    verifier_service.verify = AsyncMock(return_value=verification)
    verifier_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-2/step_1/verification_prompt.txt",
            raw_response_artifact_path="runs/run-2/step_1/verification_raw.txt",
            parsed_artifact_path="runs/run-2/step_1/verification_result.json",
        )
    )
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
        executor=executor,
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
    executor.execute.assert_awaited_once_with(action)
    verifier_service.verify.assert_awaited_once_with(updated_state, decision, remapped_executed)
    recovery_manager.recover.assert_awaited_once_with(updated_state, decision, remapped_executed, verification)
    executor.aclose_run.assert_awaited_once_with("run-2")
    run_store.save_state.assert_awaited()
    assert response.status is RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_agent_loop_does_not_cleanup_non_terminal_runs() -> None:
    initial_state = AgentState(run_id="run-keep-open", intent="Create draft", status=RunStatus.PENDING)
    updated_state = initial_state.model_copy(update={"step_count": 1, "status": RunStatus.RUNNING})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Compose button visible",
        page_hint="unknown",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait.", confidence=0.8, active_subgoal="wait")
    executed = ExecutedAction(action=action, success=True, detail="waited")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="Continue",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="Try again", retry_after_ms=1000)
    final_state = updated_state.model_copy(update={"status": RunStatus.RUNNING})

    capture_service = Mock()
    capture_service.capture = AsyncMock(return_value=frame)
    perception_service = Mock()
    perception_service.perceive = AsyncMock(return_value=perception)
    perception_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-keep-open/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-keep-open/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-keep-open/step_1/perception_parsed.json",
        )
    )
    run_store = Mock()
    run_store.get_run = AsyncMock(return_value=initial_state)
    run_store.set_status = AsyncMock(return_value=final_state)
    run_store.update_state = AsyncMock(return_value=updated_state)
    run_store.save_state = AsyncMock()
    run_store.save_state = AsyncMock()
    run_store.save_state = AsyncMock()
    policy_service = Mock()
    policy_service.choose_action = AsyncMock(return_value=decision)
    policy_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-keep-open/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-keep-open/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-keep-open/step_1/policy_decision.json",
        )
    )
    executor = Mock()
    executor.execute = AsyncMock(return_value=executed)
    executor.aclose_run = AsyncMock(return_value=1)
    executor.recorded_video_path_for_run = Mock(return_value=Path("runs/run-keep-open/session_video/session.webm"))
    verifier_service = Mock()
    verifier_service.verify = AsyncMock(return_value=verification)
    verifier_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-keep-open/step_1/verification_prompt.txt",
            raw_response_artifact_path="runs/run-keep-open/step_1/verification_raw.txt",
            parsed_artifact_path="runs/run-keep-open/step_1/verification_result.json",
        )
    )
    recovery_manager = Mock()
    recovery_manager.recover = AsyncMock(return_value=recovery)
    run_root = _local_test_dir("test-runs-running-no-cleanup")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id="run-keep-open"))

    executor.aclose_run.assert_not_awaited()
    run_store.save_state.assert_not_awaited()
    assert response.status is RunStatus.RUNNING


@pytest.mark.asyncio
async def test_agent_loop_start_run_delegates_to_store_only() -> None:
    created = AgentState(run_id="run-3", intent="Create draft", headless=True, status=RunStatus.PENDING)
    run_store = Mock()
    run_store.create_run = Mock(return_value=created)

    executor = Mock()
    executor.reset_desktop = AsyncMock()
    executor.configure_run = Mock()

    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    response = await loop.start_run(
        RunTaskRequest(intent="Complete the auth-free form and submit it successfully.", headless=True)
    )

    run_store.create_run.assert_called_once_with(
        intent="Complete the auth-free form and submit it successfully.",
        start_url=None,
        headless=True,
        benchmark=None,
    )
    executor.configure_run.assert_called_once_with("run-3", headless=True)
    assert response.run_id == "run-3"
    assert response.status is RunStatus.PENDING


@pytest.mark.asyncio
async def test_agent_loop_start_run_skips_reset_desktop_in_test_safe_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = AgentState(run_id="run-safe", intent="Create draft", status=RunStatus.PENDING)
    run_store = Mock()
    run_store.create_run = Mock(return_value=created)

    executor = Mock()
    executor.reset_desktop = AsyncMock()
    executor.configure_run = Mock()

    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    response = await loop.start_run(RunTaskRequest(intent="Navigate and click a button."))

    executor.reset_desktop.assert_not_awaited()
    assert response.run_id == "run-safe"


@pytest.mark.asyncio
async def test_agent_loop_start_run_skips_reset_desktop_in_browser_mode() -> None:
    """Browser-mode runs must not issue Win+D — Playwright runs in its own
    window and minimizing the user's foreground app on every task start steals
    focus during multi-task benchmark sweeps. Especially harmful in headless
    runs where there's no visible browser to give space to."""
    from src.core.contracts.perception import Environment as UnifiedEnvironment

    created = AgentState(run_id="run-browser", intent="Browser task", status=RunStatus.PENDING)
    run_store = Mock()
    run_store.create_run = Mock(return_value=created)

    executor = Mock()
    executor.reset_desktop = AsyncMock()
    executor.configure_run = Mock()

    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
        environment=UnifiedEnvironment.BROWSER,
    )

    await loop.start_run(RunTaskRequest(intent="Navigate to a site."))

    executor.reset_desktop.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_loop_start_run_runs_reset_desktop_in_desktop_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Desktop mode still gets a clean slate at the start of each run.

    conftest sets OPERON_TEST_SAFE_MODE=true globally to suppress real
    side-effects during the suite — disable it here so we can verify the
    real desktop-mode reset_desktop dispatch.
    """
    from src.core.contracts.perception import Environment as UnifiedEnvironment

    monkeypatch.delenv("OPERON_TEST_SAFE_MODE", raising=False)

    created = AgentState(run_id="run-desktop", intent="Desktop task", status=RunStatus.PENDING)
    run_store = Mock()
    run_store.create_run = Mock(return_value=created)

    executor = Mock()
    executor.reset_desktop = AsyncMock()
    executor.configure_run = Mock()

    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
        environment=UnifiedEnvironment.DESKTOP,
    )

    await loop.start_run(RunTaskRequest(intent="Open Notepad."))

    executor.reset_desktop.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_loop_marks_benchmark_precondition_stop_as_failed() -> None:
    initial_state = AgentState(run_id="run-setup", intent="Create draft", status=RunStatus.PENDING)
    updated_state = initial_state.model_copy(update={"step_count": 1, "status": RunStatus.RUNNING})
    failed_state = updated_state.model_copy(update={"status": RunStatus.FAILED})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Google sign in visible",
        page_hint="unknown",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[],
    )
    action = AgentAction(action_type=ActionType.STOP)
    decision = PolicyDecision(
        action=action,
        rationale="Benchmark requires an authenticated start state; pre-auth screens are out of scope.",
        confidence=1.0,
        active_subgoal="stop for benchmark setup",
    )
    executed = ExecutedAction(action=action, success=True, detail="stop acknowledged")
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=True,
        reason="Benchmark precondition failed: authenticated start state was required.",
        stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED,
    )
    recovery = RecoveryDecision(
        strategy=RecoveryStrategy.STOP,
        message="Benchmark precondition failed.",
        terminal=True,
        recoverable=False,
        stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED,
    )

    capture_service = Mock()
    capture_service.capture = AsyncMock(return_value=frame)
    perception_service = Mock()
    perception_service.perceive = AsyncMock(return_value=perception)
    perception_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-setup/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-setup/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-setup/step_1/perception_parsed.json",
        )
    )
    run_store = Mock()
    run_store.get_run = AsyncMock(return_value=initial_state)
    run_store.set_status = AsyncMock(return_value=failed_state)
    run_store.update_state = AsyncMock(return_value=updated_state)
    policy_service = Mock()
    policy_service.choose_action = AsyncMock(return_value=decision)
    policy_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-setup/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-setup/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-setup/step_1/policy_decision.json",
        )
    )
    executor = Mock()
    executor.execute = AsyncMock(return_value=executed)
    executor.recorded_video_path_for_run = Mock(return_value=None)
    verifier_service = Mock()
    verifier_service.verify = AsyncMock(return_value=verification)
    verifier_service.latest_debug_artifacts = Mock(
        return_value=ModelDebugArtifacts(
            prompt_artifact_path="runs/run-setup/step_1/verification_prompt.txt",
            raw_response_artifact_path="runs/run-setup/step_1/verification_raw.txt",
            parsed_artifact_path="runs/run-setup/step_1/verification_result.json",
        )
    )
    recovery_manager = Mock()
    recovery_manager.recover = AsyncMock(return_value=recovery)
    run_root = _local_test_dir("test-runs-precondition-stop")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id="run-setup"))

    assert response.status is RunStatus.FAILED
    assert run_store.set_status.await_args.args == ("run-setup", RunStatus.FAILED)
    payload = json.loads((run_root / "run-setup" / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["recovery_decision"]["stop_reason"] == StopReason.BENCHMARK_PRECONDITION_FAILED.value


@pytest.mark.asyncio
async def test_agent_loop_logs_pre_step_perception_failure_for_observability() -> None:
    run_root = _local_test_dir("test-runs-pre-step-perception-failure")
    run_store = FileBackedRunStore(root_dir=run_root)
    created = run_store.create_run("Complete the auth-free form and submit it successfully.")
    frame = CaptureFrame(
        artifact_path=str(run_root / created.run_id / "step_1" / "before.png"),
        width=1280,
        height=800,
        mime_type="image/png",
    )

    capture_service = SimpleNamespace(capture=AsyncMock(return_value=frame))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=PerceptionError("Gemini perception output did not match the strict schema.")),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path=str(run_root / created.run_id / "step_1" / "perception_prompt.txt"),
            raw_response_artifact_path=str(run_root / created.run_id / "step_1" / "perception_raw.txt"),
            parsed_artifact_path=str(run_root / created.run_id / "step_1" / "perception_parsed.json"),
        ),
    )
    policy_service = Mock()
    policy_service.choose_action = AsyncMock()
    executor = Mock()
    executor.execute = AsyncMock()
    verifier_service = Mock()
    verifier_service.verify = AsyncMock()
    recovery_manager = Mock()
    recovery_manager.recover = AsyncMock()

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id=created.run_id))

    assert response.status is RunStatus.FAILED
    state = await run_store.get_run(created.run_id)
    assert state is not None
    assert state.status is RunStatus.FAILED
    assert state.step_count == 0
    payload = json.loads((run_root / created.run_id / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["record_type"] == "pre_step_failure"
    assert payload["run_id"] == created.run_id
    assert payload["step_id"] == "step_1"
    assert payload["failure"]["stage"] == "perceive"
    assert payload["failure"]["category"] == FailureCategory.PRE_STEP_PERCEPTION_FAILED.value
    assert payload["failure"]["stop_reason"] == StopReason.PRE_STEP_PERCEPTION_FAILED.value
    assert "strict schema" in payload["error_message"]
    assert payload["perception_debug"]["raw_response_artifact_path"].endswith("perception_raw.txt")
    assert state.stop_reason is StopReason.PRE_STEP_PERCEPTION_FAILED
    policy_service.choose_action.assert_not_called()


@pytest.mark.asyncio
async def test_agent_loop_logs_pre_step_low_quality_perception_failure() -> None:
    run_root = _local_test_dir("test-runs-pre-step-low-quality")
    run_store = FileBackedRunStore(root_dir=run_root)
    created = run_store.create_run("Complete the auth-free form and submit it successfully.")
    frame = CaptureFrame(
        artifact_path=str(run_root / created.run_id / "step_1" / "before.png"),
        width=1280,
        height=800,
        mime_type="image/png",
    )

    capture_service = SimpleNamespace(capture=AsyncMock(return_value=frame))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=PerceptionLowQualityError("Gemini perception output was low quality: zero usable_for_targeting elements")),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path=str(run_root / created.run_id / "step_1" / "perception_prompt.txt"),
            raw_response_artifact_path=str(run_root / created.run_id / "step_1" / "perception_raw.txt"),
            parsed_artifact_path=str(run_root / created.run_id / "step_1" / "perception_parsed.json"),
            retry_log_artifact_path=str(run_root / created.run_id / "step_1" / "perception_retry_log.txt"),
        ),
    )
    Path(perception_service.latest_debug_artifacts().retry_log_artifact_path).parent.mkdir(parents=True, exist_ok=True)
    Path(perception_service.latest_debug_artifacts().retry_log_artifact_path).write_text(
        "attempt=1 reason=zero usable_for_targeting elements\nattempt=2 reason=zero usable_for_targeting elements",
        encoding="utf-8",
    )

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    response = await loop.step_run(StepRequest(run_id=created.run_id))

    assert response.status is RunStatus.FAILED
    state = await run_store.get_run(created.run_id)
    assert state is not None
    assert state.stop_reason is StopReason.PERCEPTION_LOW_QUALITY
    payload = json.loads((run_root / created.run_id / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["failure"]["category"] == FailureCategory.PERCEPTION_LOW_QUALITY.value
    assert payload["failure"]["stop_reason"] == StopReason.PERCEPTION_LOW_QUALITY.value
    assert payload["perception_debug"]["retry_log_artifact_path"].endswith("perception_retry_log.txt")


@pytest.mark.asyncio
async def test_stale_target_triggers_reresolution_once() -> None:
    initial_state = AgentState(run_id="run-retry", intent="Create draft", status=RunStatus.PENDING)
    updated_state = initial_state.model_copy(update={"step_count": 1, "status": RunStatus.RUNNING})
    retry_state = updated_state.model_copy(update={"step_count": 2})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Form visible",
        page_hint="form_page",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=40,
                y=80,
                width=120,
                height=80,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    refreshed_perception = ScreenPerception(
        summary="Form visible",
        page_hint="form_page",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=40,
                y=80,
                width=120,
                height=80,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="name-input", x=10, y=10)
    decision = PolicyDecision(action=action, rationale="Click name.", confidence=0.8, active_subgoal="fill_name")
    stale = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: stale target before action.",
        execution_trace=ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="stale_target_before_action",
                    verification_result="click_not_attempted",
                    failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
                )
            ],
            final_outcome="failure",
            final_failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
        ),
        failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
        failure_stage=None,
    )
    retry_action = action.model_copy(update={"x": 100, "y": 120})
    success = ExecutedAction(
        action=retry_action,
        success=True,
        detail="Clicked target.",
        execution_trace=ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="ok",
                    verification_result="click_verified",
                )
            ],
            final_outcome="success",
        ),
    )
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="clicked",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue")

    capture_service = SimpleNamespace(capture=AsyncMock(side_effect=[frame, frame]))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=[perception, refreshed_perception]),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-retry/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-retry/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-retry/step_1/perception_parsed.json",
        ),
    )
    run_store = SimpleNamespace(
        get_run=AsyncMock(return_value=initial_state),
        set_status=AsyncMock(return_value=updated_state),
        update_state=AsyncMock(side_effect=[updated_state, retry_state]),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(return_value=decision),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-retry/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-retry/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-retry/step_1/policy_decision.json",
        ),
    )
    executor = SimpleNamespace(execute=AsyncMock(side_effect=[stale, success]))
    verifier_service = SimpleNamespace(verify=AsyncMock(return_value=verification))
    recovery_manager = SimpleNamespace(recover=AsyncMock(return_value=recovery))
    run_root = _local_test_dir("test-execution-retry")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    await loop.step_run(StepRequest(run_id="run-retry"))

    assert executor.execute.await_count == 2
    assert executor.execute.await_args_list[1].args[0].x == 100
    assert executor.execute.await_args_list[1].args[0].y == 120
    assert executor.execute.await_args_list[1].args[0].target_context is not None
    payload = json.loads((run_root / "run-retry" / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["executed_action"]["execution_trace"]["retry_attempted"] is True
    assert payload["executed_action"]["execution_trace"]["target_reresolved"] is True
    assert payload["executed_action"]["execution_trace"]["reresolution_trace"]["original_intent"]["target_text"] == "Name"
    assert payload["executed_action"]["execution_trace"]["reresolution_trace"]["final_target_element_id"] == "name-input"
    assert payload["executed_action"]["execution_trace_artifact_path"].endswith("execution_trace.json")


@pytest.mark.asyncio
async def test_reresolution_failure_is_recorded_explicitly() -> None:
    initial_state = AgentState(run_id="run-reresolution-fail", intent="Complete form", status=RunStatus.PENDING)
    updated_state = initial_state.model_copy(update={"step_count": 1, "status": RunStatus.RUNNING})
    retry_state = updated_state.model_copy(update={"step_count": 2})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Form visible",
        page_hint="form_page",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[
            UIElement(
                element_id="email-input",
                element_type=UIElementType.INPUT,
                label="Email",
                x=40,
                y=80,
                width=120,
                height=40,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    refreshed_perception = ScreenPerception(
        summary="Form rerendered",
        page_hint="form_page",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=40,
                y=80,
                width=120,
                height=40,
                is_interactable=True,
                confidence=0.9,
            ),
            UIElement(
                element_id="phone-input",
                element_type=UIElementType.INPUT,
                label="Phone",
                x=40,
                y=132,
                width=120,
                height=40,
                is_interactable=True,
                confidence=0.9,
            ),
        ],
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="email-input", x=10, y=10)
    decision = PolicyDecision(action=action, rationale="Click email.", confidence=0.8, active_subgoal="fill_email")
    stale = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: stale target before action.",
        execution_trace=ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="stale_target_before_action",
                    verification_result="click_not_attempted",
                    failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
                )
            ],
            final_outcome="failure",
            final_failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
        ),
        failure_category=FailureCategory.STALE_TARGET_BEFORE_ACTION,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="not clicked",
        failure_category=FailureCategory.TARGET_RERESOLUTION_FAILED,
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.RETRY_SAME_STEP, message="retry")

    capture_service = SimpleNamespace(capture=AsyncMock(side_effect=[frame, frame]))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=[perception, refreshed_perception]),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-reresolution-fail/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-reresolution-fail/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-reresolution-fail/step_1/perception_parsed.json",
        ),
    )
    run_store = SimpleNamespace(
        get_run=AsyncMock(return_value=initial_state),
        set_status=AsyncMock(return_value=updated_state),
        update_state=AsyncMock(side_effect=[updated_state, retry_state]),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(return_value=decision),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-reresolution-fail/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-reresolution-fail/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-reresolution-fail/step_1/policy_decision.json",
        ),
    )
    executor = SimpleNamespace(execute=AsyncMock(return_value=stale))
    verifier_service = SimpleNamespace(verify=AsyncMock(return_value=verification))
    recovery_manager = SimpleNamespace(recover=AsyncMock(return_value=recovery))
    run_root = _local_test_dir("test-reresolution-failure")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    await loop.step_run(StepRequest(run_id="run-reresolution-fail"))

    assert executor.execute.await_count == 1
    payload = json.loads((run_root / "run-reresolution-fail" / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["executed_action"]["failure_category"] == FailureCategory.TARGET_RERESOLUTION_FAILED.value
    assert payload["executed_action"]["execution_trace"]["reresolution_trace"]["succeeded"] is False


def test_verified_type_action_marks_target_complete() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-1",
        intent="Complete form",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(latest_page_signature="form_page|none|name-input:Name"),
    )
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice")
    decision = PolicyDecision(action=action, rationale="Fill name.", confidence=0.9, active_subgoal="fill_name")
    executed = ExecutedAction(action=action, success=True, detail="typed")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="typed",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue")

    trace = loop._update_progress_state(
        state=state,
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=recovery,
        step_index=1,
    )

    assert "id:name-input" in state.progress_state.completed_targets
    assert state.progress_state.target_value_history["id:name-input"] == "Alice"
    assert "fill_name" in state.progress_state.completed_subgoals
    assert state.progress_state.no_progress_streak == 0
    assert trace.progress_made is True


def test_click_focus_does_not_complete_new_subgoal_without_page_change() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-focus",
        intent="Search site",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            previous_page_signature="form_page|none|search-input:Search",
            latest_page_signature="form_page|none|search-input:Search",
        ),
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="search-input")
    decision = PolicyDecision(
        action=action,
        rationale="Focus search.",
        confidence=0.9,
        active_subgoal="search_for_markov_chain",
    )
    executed = ExecutedAction(action=action, success=True, detail="clicked")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="clicked",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue")

    trace = loop._update_progress_state(
        state=state,
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=recovery,
        step_index=1,
    )

    assert trace.progress_made is True
    assert "search_for_markov_chain" not in state.progress_state.completed_subgoals


def test_type_with_enter_does_not_complete_subgoal_without_page_change() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-submit",
        intent="Search site",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            previous_page_signature="form_page|none|search-input:Search",
            latest_page_signature="form_page|none|search-input:Search",
        ),
    )
    action = AgentAction(
        action_type=ActionType.TYPE,
        target_element_id="search-input",
        text="Markov chain",
        press_enter=True,
    )
    decision = PolicyDecision(
        action=action,
        rationale="Search.",
        confidence=0.9,
        active_subgoal="search_for_markov_chain",
    )
    executed = ExecutedAction(action=action, success=True, detail="typed")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="typed",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue")

    loop._update_progress_state(
        state=state,
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=recovery,
        step_index=1,
    )

    assert "id:search-input" in state.progress_state.completed_targets
    assert state.progress_state.target_value_history["id:search-input"] == "Markov chain"
    assert "search_for_markov_chain" not in state.progress_state.completed_subgoals


def test_infer_focused_element_from_direct_input_click() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-focus-direct",
        intent="Search site",
        status=RunStatus.RUNNING,
        observation_history=[
            ScreenPerception(
                summary="Search input visible.",
                page_hint=PageHint.FORM_PAGE,
                capture_artifact_path="runs/test/step_1/before.png",
                visible_elements=[
                    UIElement(
                        element_id="search-input",
                        element_type=UIElementType.INPUT,
                        label="Search",
                        x=100,
                        y=50,
                        width=200,
                        height=30,
                        is_interactable=True,
                        confidence=0.9,
                    )
                ],
                confidence=0.9,
            )
        ],
        action_history=[
            ExecutedAction(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="search-input"),
                success=True,
                detail="clicked",
            )
        ],
    )
    perception = ScreenPerception(
        summary="Search input still visible.",
        page_hint=PageHint.FORM_PAGE,
        capture_artifact_path="runs/test/step_2/before.png",
        visible_elements=[
            UIElement(
                element_id="search-input",
                element_type=UIElementType.INPUT,
                label="Search",
                x=100,
                y=50,
                width=200,
                height=30,
                is_interactable=True,
                confidence=0.9,
            )
        ],
        confidence=0.9,
    )

    inferred = loop._infer_focused_element(state, perception)

    assert inferred.focused_element_id == "search-input"


def test_infer_focused_element_from_matching_label_after_control_click() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-focus-label",
        intent="Search site",
        status=RunStatus.RUNNING,
        observation_history=[
            ScreenPerception(
                summary="Search control visible.",
                page_hint=PageHint.UNKNOWN,
                capture_artifact_path="runs/test/step_1/before.png",
                visible_elements=[
                    UIElement(
                        element_id="search-button",
                        element_type=UIElementType.BUTTON,
                        label="Search or jump to...",
                        x=100,
                        y=50,
                        width=200,
                        height=30,
                        is_interactable=True,
                        confidence=0.9,
                    )
                ],
                confidence=0.9,
            )
        ],
        action_history=[
            ExecutedAction(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="search-button"),
                success=True,
                detail="clicked",
            )
        ],
    )
    perception = ScreenPerception(
        summary="Search input visible.",
        page_hint=PageHint.FORM_PAGE,
        capture_artifact_path="runs/test/step_2/before.png",
        visible_elements=[
            UIElement(
                element_id="search-input",
                element_type=UIElementType.INPUT,
                label="Search or jump to...",
                x=100,
                y=50,
                width=200,
                height=30,
                is_interactable=True,
                confidence=0.9,
            )
        ],
        confidence=0.9,
    )

    inferred = loop._infer_focused_element(state, perception)

    assert inferred.focused_element_id == "search-input"


def test_attach_target_context_preserves_planner_supplied_click_coords() -> None:
    """Planner-supplied coordinates must be respected; the loop should only fall
    back to the target's bbox center when x/y are missing. Overwriting valid
    coords forces every click to the geometric center and amplifies jitter."""
    loop = _loop()
    perception = ScreenPerception(
        summary="GitHub homepage with search button",
        page_hint=PageHint.FORM_PAGE,
        capture_artifact_path="runs/test/step_1/before.png",
        visible_elements=[
            UIElement(
                element_id="search_button",
                element_type=UIElementType.BUTTON,
                label="Search",
                x=835,
                y=44,
                width=28,
                height=28,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    action = AgentAction(
        action_type=ActionType.CLICK,
        target_element_id="search_button",
        x=835,
        y=44,
    )

    normalized = loop._attach_target_context(action, perception)

    assert normalized.x == 835
    assert normalized.y == 44
    assert normalized.target_context is not None


def test_attach_target_context_fills_missing_click_coordinates_from_target() -> None:
    loop = _loop()
    perception = ScreenPerception(
        summary="GitHub homepage with search button",
        page_hint=PageHint.FORM_PAGE,
        capture_artifact_path="runs/test/step_1/before.png",
        visible_elements=[
            UIElement(
                element_id="search_button",
                element_type=UIElementType.BUTTON,
                label="Search",
                x=834,
                y=44,
                width=28,
                height=28,
                is_interactable=True,
                confidence=1.0,
            )
        ],
    )
    action = AgentAction(
        action_type=ActionType.CLICK,
        target_element_id="search_button",
    )

    normalized = loop._attach_target_context(action, perception)

    assert normalized.x == 848
    assert normalized.y == 58
    assert normalized.target_context is not None


def test_attach_target_context_fills_upload_file_native_coordinates_from_target() -> None:
    """Regression: upload_file_native must get x,y populated from the target
    element, otherwise NativeBrowserExecutor falls through to a broken
    [data-element-id='...'] locator and times out."""
    loop = _loop()
    perception = ScreenPerception(
        summary="File uploader page with Add files button",
        page_hint=PageHint.FORM_PAGE,
        capture_artifact_path="runs/test/step_1/before.png",
        visible_elements=[
            UIElement(
                element_id="add_files_btn",
                element_type=UIElementType.BUTTON,
                label="Add files",
                x=660,
                y=312,
                width=130,
                height=40,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    action = AgentAction(
        action_type=ActionType.UPLOAD_FILE_NATIVE,
        target_element_id="add_files_btn",
        text=r"C:\tmp\test_upload.txt",
    )

    normalized = loop._attach_target_context(action, perception)

    assert normalized.x == 725
    assert normalized.y == 332


def test_repeated_same_value_type_action_is_blocked() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-2",
        intent="Complete form",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        progress_state=ProgressState(
            completed_targets=["id:name-input"],
            completed_subgoals=["fill_name"],
            target_value_history={"id:name-input": "Alice"},
            latest_page_signature="form_page|none|name-input:Name",
            target_completion_page_signatures={"id:name-input": "form_page|none|name-input:Name"},
            subgoal_completion_page_signatures={"fill_name": "form_page|none|name-input:Name"},
        ),
    )

    blocked = loop._block_redundant_action(
        state,
        AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        step_index=2,
    )

    assert blocked is not None
    assert blocked.failure_category is FailureCategory.SUBGOAL_ALREADY_COMPLETED


def test_repeated_same_click_without_progress_is_blocked() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-3",
        intent="Complete form",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            latest_page_signature="form_page|none|submit-button:Submit",
            repeated_action_count={"click|id:submit-button|": 2},
            no_progress_streak=1,
        ),
    )

    blocked = loop._block_redundant_action(
        state,
        AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button"),
        step_index=3,
    )

    assert blocked is not None
    assert blocked.failure_category is FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS


def test_alternating_action_pattern_triggers_loop_detection() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-4",
        intent="Complete form",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            latest_page_signature="form_page|none|name-input:Name|email-input:Email",
            recent_actions=[
                "click|id:name-input|",
                "click|id:email-input|",
                "click|id:name-input|",
            ],
            no_progress_streak=2,
        ),
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="email-input")
    decision = PolicyDecision(action=action, rationale="Click email.", confidence=0.8, active_subgoal="focus email-input")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="blocked",
        failure_category=FailureCategory.CLICK_NO_EFFECT,
        failure_stage=None,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="no effect",
        failure_category=FailureCategory.CLICK_NO_EFFECT,
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.RETRY_SAME_STEP, message="retry")

    trace = loop._update_progress_state(
        state=state,
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=recovery,
        step_index=4,
    )

    assert trace.final_failure_category is FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS
    assert trace.loop_pattern_detected == "alternating_action_pattern_without_progress"
    assert state.progress_state.loop_detected is True


def test_repeated_failure_signatures_trigger_loop_stop() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-5",
        intent="Complete form",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            latest_page_signature="form_page|none|name-input:Name",
            recent_failures=[
                "click_no_effect|id:name-input",
                "click_no_effect|id:name-input",
            ],
            no_progress_streak=2,
        ),
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="name-input")
    decision = PolicyDecision(action=action, rationale="Click name.", confidence=0.8, active_subgoal="focus name-input")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="no effect",
        failure_category=FailureCategory.CLICK_NO_EFFECT,
        failure_stage=None,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="no effect",
        failure_category=FailureCategory.CLICK_NO_EFFECT,
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.RETRY_SAME_STEP, message="retry")

    trace = loop._update_progress_state(
        state=state,
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=recovery,
        step_index=3,
    )

    assert trace.final_failure_category is FailureCategory.REPEATED_FAILURE_LOOP
    guarded = loop._apply_progress_stop_guard(recovery, trace)
    assert guarded.strategy is RecoveryStrategy.STOP
    assert guarded.stop_reason is StopReason.REPEATED_FAILURE_LOOP


def test_page_change_reduces_no_progress_suppression() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-6",
        intent="Complete form",
        status=RunStatus.RUNNING,
        progress_state=ProgressState(
            latest_page_signature="form_page|none|name-input:Name",
            repeated_action_count={"click|id:name-input|": 2},
            repeated_target_count={"id:name-input": 2},
            recent_failures=["click_no_effect|id:name-input"],
            no_progress_streak=2,
            loop_detected=True,
        ),
    )
    perception = ScreenPerception(
        summary="Success visible",
        page_hint="form_success",
        capture_artifact_path="after.png",
        visible_elements=[],
    )

    loop._sync_progress_state_with_perception(state, perception)

    assert state.progress_state.latest_page_signature.startswith("form_success|")
    assert state.progress_state.repeated_action_count == {}
    assert state.progress_state.repeated_target_count == {}
    assert state.progress_state.recent_failures == []
    assert state.progress_state.no_progress_streak == 2  # streak resets in _update_progress_state, not _sync
    assert state.progress_state.loop_detected is False


def test_completed_subgoal_prevents_redundant_reexecution() -> None:
    loop = _loop()
    state = AgentState(
        run_id="run-progress-7",
        intent="Complete form",
        status=RunStatus.RUNNING,
        current_subgoal="submit_form",
        progress_state=ProgressState(
            completed_subgoals=["submit_form"],
            latest_page_signature="form_page|none|submit-button:Submit",
            subgoal_completion_page_signatures={"submit_form": "form_page|none|submit-button:Submit"},
        ),
    )

    blocked = loop._block_redundant_action(
        state,
        AgentAction(action_type=ActionType.WAIT, wait_ms=1000),
        step_index=4,
    )

    assert blocked is not None
    assert blocked.failure_category is FailureCategory.SUBGOAL_ALREADY_COMPLETED


@pytest.mark.asyncio
async def test_no_progress_across_consecutive_steps_terminates_deterministically() -> None:
    run_root = _local_test_dir("test-runs-no-progress-stop")
    initial_state = AgentState(
        run_id="run-no-progress",
        intent="Complete form",
        status=RunStatus.RUNNING,
        step_count=2,
        progress_state=ProgressState(
            latest_page_signature="form_page|none|name-input:Name",
            no_progress_streak=2,
        ),
    )
    updated_state = initial_state.model_copy(update={"step_count": 3})
    failed_state = updated_state.model_copy(update={"status": RunStatus.FAILED})
    frame = CaptureFrame(artifact_path="artifacts/frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(
        summary="Form visible",
        page_hint="form_page",
        capture_artifact_path=frame.artifact_path,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=40,
                y=80,
                width=120,
                height=40,
                is_interactable=True,
                confidence=0.9,
            )
        ],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait.", confidence=0.8, active_subgoal="wait")
    executed = ExecutedAction(action=action, success=False, detail="wait did not help")
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="still no progress",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="retry")

    capture_service = SimpleNamespace(capture=AsyncMock(return_value=frame))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(return_value=perception),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path=str(run_root / "run-no-progress" / "step_3" / "perception_prompt.txt"),
            raw_response_artifact_path=str(run_root / "run-no-progress" / "step_3" / "perception_raw.txt"),
            parsed_artifact_path=str(run_root / "run-no-progress" / "step_3" / "perception_parsed.json"),
        ),
    )
    run_store = SimpleNamespace(
        get_run=AsyncMock(return_value=initial_state),
        set_status=AsyncMock(return_value=failed_state),
        update_state=AsyncMock(return_value=updated_state),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(return_value=decision),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path=str(run_root / "run-no-progress" / "step_3" / "policy_prompt.txt"),
            raw_response_artifact_path=str(run_root / "run-no-progress" / "step_3" / "policy_raw.txt"),
            parsed_artifact_path=str(run_root / "run-no-progress" / "step_3" / "policy_decision.json"),
        ),
    )
    executor = SimpleNamespace(execute=AsyncMock(return_value=executed))
    verifier_service = SimpleNamespace(verify=AsyncMock(return_value=verification))
    recovery_manager = SimpleNamespace(recover=AsyncMock(return_value=recovery))
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    response = await loop.step_run(StepRequest(run_id="run-no-progress"))

    assert response.status is RunStatus.FAILED
    payload = json.loads((run_root / "run-no-progress" / "run.jsonl").read_text(encoding="utf-8").strip())
    assert payload["recovery_decision"]["stop_reason"] == StopReason.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS.value
    assert payload["progress_trace_artifact_path"].endswith("progress_trace.json")
    assert payload["progress_state"]["no_progress_streak"] == 3


# ── force_fresh_perception ordering test ─────────────────────────


@pytest.mark.asyncio
async def test_consume_force_fresh_perception_sleeps_before_reset() -> None:
    """The settle delay must complete BEFORE the flag is reset.
    Otherwise an early return between the reset and the await loses the wait.
    Browser environment uses 1.0s; desktop uses 0.5s."""
    from unittest.mock import patch as _patch

    from src.core.contracts.perception import Environment as UnifiedEnvironment

    desktop_loop = _loop()
    desktop_loop.environment = UnifiedEnvironment.DESKTOP
    state = AgentState(run_id="r1", intent="x", status=RunStatus.RUNNING, force_fresh_perception=True)

    flag_when_sleeping: list[bool] = []

    async def _spy_sleep(_seconds):
        # Capture the flag's value at the moment of sleep — must still be True,
        # proving sleep happens BEFORE reset.
        flag_when_sleeping.append(state.force_fresh_perception)

    with _patch("src.agent.loop.asyncio.sleep", side_effect=_spy_sleep) as mock_sleep:
        await desktop_loop._consume_force_fresh_perception(state)

    assert flag_when_sleeping == [True], "sleep must run while the flag is still True"
    assert state.force_fresh_perception is False, "flag must be reset after the sleep returns"
    mock_sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_consume_force_fresh_perception_uses_longer_delay_for_browser() -> None:
    from unittest.mock import patch as _patch

    from src.core.contracts.perception import Environment as UnifiedEnvironment

    browser_loop = _loop()
    browser_loop.environment = UnifiedEnvironment.BROWSER
    state = AgentState(run_id="r1", intent="x", status=RunStatus.RUNNING, force_fresh_perception=True)

    with _patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await browser_loop._consume_force_fresh_perception(state)

    mock_sleep.assert_awaited_once_with(1.0)
    assert state.force_fresh_perception is False


@pytest.mark.asyncio
async def test_consume_force_fresh_perception_noop_when_flag_clear() -> None:
    from unittest.mock import patch as _patch

    loop = _loop()
    state = AgentState(run_id="r1", intent="x", status=RunStatus.RUNNING, force_fresh_perception=False)

    with _patch("src.agent.loop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await loop._consume_force_fresh_perception(state)

    mock_sleep.assert_not_awaited()


# ── stable-frame perception bypass tests ─────────────────────────


def test_maybe_reuse_prior_perception_returns_none_when_velocity_nonzero() -> None:
    """Any non-zero velocity must prevent the cache from firing — even tiny motion."""
    loop = _loop()
    frame = CaptureFrame(artifact_path="x.png", width=100, height=100, mime_type="image/png", visual_velocity=0.001)
    state = AgentState(run_id="r1", intent="x", status=RunStatus.RUNNING)
    assert loop._maybe_reuse_prior_perception(state, frame) is None


def test_maybe_reuse_prior_perception_requires_last_action_wait() -> None:
    """Cache fires only when the last action was idempotent (WAIT)."""
    from src.models.execution import ExecutedAction
    from src.models.policy import ActionType, AgentAction
    loop = _loop()
    frame = CaptureFrame(artifact_path="x.png", width=100, height=100, mime_type="image/png", visual_velocity=0.0)
    perception = ScreenPerception(summary="x", page_hint="unknown", capture_artifact_path="prev.png", visible_elements=[])
    state = AgentState(
        run_id="r1", intent="x", status=RunStatus.RUNNING,
        observation_history=[perception],
        action_history=[ExecutedAction(action=AgentAction(action_type=ActionType.CLICK, x=10, y=20), success=True, detail="ok")],
    )
    # Last action was CLICK, not WAIT — cache must NOT fire.
    assert loop._maybe_reuse_prior_perception(state, frame) is None


def test_maybe_reuse_prior_perception_fires_when_all_conditions_met() -> None:
    """All four gates green → reuse the prior perception with updated artifact path."""
    from src.models.execution import ExecutedAction
    from src.models.policy import ActionType, AgentAction
    from src.models.verification import VerificationResult, VerificationStatus
    loop = _loop()
    frame = CaptureFrame(artifact_path="current.png", width=100, height=100, mime_type="image/png", visual_velocity=0.0)
    prior = ScreenPerception(summary="prev frame", page_hint="form_page", capture_artifact_path="prev.png", visible_elements=[])
    state = AgentState(
        run_id="r1", intent="x", status=RunStatus.RUNNING,
        observation_history=[prior],
        action_history=[ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=500), success=True, detail="waited")],
        verification_history=[VerificationResult(status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="x")],
    )

    cached = loop._maybe_reuse_prior_perception(state, frame)

    assert cached is not None
    assert cached.summary == "prev frame"
    assert cached.capture_artifact_path == "current.png"


def test_maybe_reuse_prior_perception_blocks_back_to_back_reuse() -> None:
    """A second consecutive call (same run_id) must NOT reuse — forces a fresh
    perception every other step to catch slow renders the first skip might have missed."""
    from src.models.execution import ExecutedAction
    from src.models.policy import ActionType, AgentAction
    from src.models.verification import VerificationResult, VerificationStatus
    loop = _loop()
    frame = CaptureFrame(artifact_path="current.png", width=100, height=100, mime_type="image/png", visual_velocity=0.0)
    prior = ScreenPerception(summary="prev", page_hint="form_page", capture_artifact_path="prev.png", visible_elements=[])
    state = AgentState(
        run_id="r1", intent="x", status=RunStatus.RUNNING,
        observation_history=[prior],
        action_history=[ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=500), success=True, detail="waited")],
        verification_history=[VerificationResult(status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="x")],
    )

    # First call should reuse.
    assert loop._maybe_reuse_prior_perception(state, frame) is not None
    # Second consecutive call must NOT reuse — force a fresh look.
    assert loop._maybe_reuse_prior_perception(state, frame) is None
