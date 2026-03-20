"""Focused tests for explicit failure taxonomy propagation and reporting."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.loop import AgentLoop
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import PlaceholderGeminiClient
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage, RunStatus, StepRequest, StopReason
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts, StepLog
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import VerificationFailureType, VerificationResult, VerificationStatus


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.asyncio
async def test_verifier_propagates_execution_failure_taxonomy() -> None:
    verifier = DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())
    state = AgentState(run_id="run-1", intent="Create draft", status=RunStatus.RUNNING)
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="missing")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.8, active_subgoal="open compose")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: Unable to resolve click target.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )

    result = await verifier.verify(state, decision, executed)

    assert result.failure_category is FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert result.failure_stage is LoopStage.EXECUTE
    assert result.failure_type is VerificationFailureType.ACTION_FAILED


@pytest.mark.asyncio
async def test_recovery_marks_retry_limit_as_terminal_failure() -> None:
    manager = RuleBasedRecoveryManager()
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.8, active_subgoal="open compose")
    retry_key = f"{decision.active_subgoal}:{VerificationFailureType.ACTION_FAILED.value}"
    state = AgentState(
        run_id="run-2",
        intent="Create draft",
        status=RunStatus.RUNNING,
        retry_counts={retry_key: 2},
    )
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed",
        failure_category=FailureCategory.EXECUTION_ERROR,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Action failed repeatedly",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_ERROR,
        failure_stage=LoopStage.EXECUTE,
    )

    recovery = await manager.recover(state, decision, executed, verification)

    assert recovery.failure_category is FailureCategory.RETRY_LIMIT_REACHED
    assert recovery.failure_stage is LoopStage.RECOVER
    assert recovery.terminal is True
    assert recovery.recoverable is False
    assert recovery.stop_reason is StopReason.RETRY_LIMIT_REACHED


@pytest.mark.asyncio
async def test_step_log_records_failure_taxonomy_for_failed_step() -> None:
    state = AgentState(run_id="run-3", intent="Create draft", status=RunStatus.PENDING)
    frame = CaptureFrame(artifact_path="runs/run-3/step_1/before.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(summary="Inbox visible", capture_artifact_path=frame.artifact_path, visible_elements=[])
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="missing")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.8, active_subgoal="open compose")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: Unable to resolve click target.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Executed action reported failure.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        recovery_hint="retry_same_step",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    final_state = state.model_copy(update={"status": RunStatus.RUNNING, "step_count": 1})

    capture_service = SimpleNamespace(capture=AsyncMock(return_value=frame))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(return_value=perception),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-3/step_1/perception_prompt.txt",
            raw_response_artifact_path="runs/run-3/step_1/perception_raw.txt",
            parsed_artifact_path="runs/run-3/step_1/perception_parsed.json",
        ),
    )
    run_store = SimpleNamespace(
        get_run=AsyncMock(return_value=state),
        update_state=AsyncMock(return_value=final_state),
        set_status=AsyncMock(return_value=final_state),
    )
    run_root = _local_test_dir("test-failure-taxonomy")
    run_store.before_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "before.png")
    run_store.after_artifact_path = lambda run_id, step_index: str(run_root / run_id / f"step_{step_index}" / "after.png")
    run_store.run_log_path = lambda run_id: str(run_root / run_id / "run.jsonl")
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(return_value=decision),
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="runs/run-3/step_1/policy_prompt.txt",
            raw_response_artifact_path="runs/run-3/step_1/policy_raw.txt",
            parsed_artifact_path="runs/run-3/step_1/policy_decision.json",
        ),
    )
    browser_executor = SimpleNamespace(execute=AsyncMock(return_value=executed))
    verifier_service = SimpleNamespace(verify=AsyncMock(return_value=verification))
    recovery_manager = RuleBasedRecoveryManager()

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        browser_executor=browser_executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
    )

    await loop.step_run(StepRequest(run_id="run-3"))

    payload = json.loads((run_root / "run-3" / "run.jsonl").read_text(encoding="utf-8").strip())
    log = StepLog.model_validate(payload)
    assert log.failure is not None
    assert log.failure.category is FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert log.failure.stage is LoopStage.EXECUTE
    assert log.failure.retry_count == 1
    assert log.failure.terminal is False
    assert log.failure.recoverable is True
