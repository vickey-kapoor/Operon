"""Phase 4 integration tests for routing the legacy Operon loop through the unified runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import sys
import types

import pytest

from core.contracts.perception import Environment as UnifiedEnvironment
from executors.browser_executor import BrowserExecutor as UnifiedBrowserExecutor
from runtime.state import AgentRuntimeState

gemini_stub = types.ModuleType("src.clients.gemini")
gemini_stub.GeminiClient = object
gemini_stub.GeminiClientError = Exception
sys.modules.setdefault("src.clients.gemini", gemini_stub)

screen_diff_stub = types.ModuleType("src.agent.screen_diff")
screen_diff_stub.SCREEN_CHANGE_THRESHOLD = 0.05
screen_diff_stub.compute_screen_change_ratio = lambda before, after: 0.25
sys.modules.setdefault("src.agent.screen_diff", screen_diff_stub)

from src.agent.loop import AgentLoop
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, RunStatus, RunTaskRequest, StepRequest
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts
from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.verification import VerificationFailureType, VerificationResult, VerificationStatus
from src.store.run_store import FileBackedRunStore


def _debug_artifacts(root: Path, stage: str) -> ModelDebugArtifacts:
    return ModelDebugArtifacts(
        prompt_artifact_path=str(root / f"{stage}_prompt.txt"),
        raw_response_artifact_path=str(root / f"{stage}_raw.txt"),
        parsed_artifact_path=str(root / f"{stage}_parsed.json"),
    )


def _perception(page_hint: PageHint, summary: str, *, target_id: str | None = None) -> ScreenPerception:
    visible_elements = []
    if target_id is not None:
        visible_elements.append(
            UIElement(
                element_id=target_id,
                element_type=UIElementType.BUTTON,
                label="Settings",
                x=100,
                y=100,
                width=40,
                height=20,
                is_interactable=True,
                confidence=0.95,
            )
        )
    return ScreenPerception(
        summary=summary,
        page_hint=page_hint,
        capture_artifact_path="runs/test/before.png",
        visible_elements=visible_elements,
        confidence=0.95,
    )


@pytest.mark.asyncio
async def test_existing_agent_loop_flows_through_unified_retry_path(tmp_path: Path) -> None:
    run_store = FileBackedRunStore(root_dir=tmp_path / "runs")
    frame_1 = CaptureFrame(artifact_path=str(tmp_path / "frame_1.png"), width=1280, height=800, mime_type="image/png")
    frame_2 = CaptureFrame(artifact_path=str(tmp_path / "frame_2.png"), width=1280, height=800, mime_type="image/png")

    perception_1 = _perception(PageHint.FORM_PAGE, "Dashboard visible", target_id="settings-btn")
    perception_2 = _perception(PageHint.FORM_PAGE, "Settings page visible", target_id="settings-btn")
    decision_1 = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="settings-btn"),
        rationale="Open settings.",
        confidence=0.9,
        active_subgoal="open settings",
    )
    decision_2 = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="settings-btn"),
        rationale="Retry open settings.",
        confidence=0.95,
        active_subgoal="open settings",
    )
    executed_1 = ExecutedAction(action=decision_1.action, success=True, detail="clicked")
    executed_2 = ExecutedAction(action=decision_2.action, success=True, detail="clicked")
    verification_1 = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="The page may need more time to respond.",
        failure_type=VerificationFailureType.EXPECTED_OUTCOME_NOT_MET,
        failure_category=FailureCategory.EXPECTED_OUTCOME_NOT_MET,
    )
    verification_2 = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="Settings opened.",
    )

    capture_service = SimpleNamespace(capture=AsyncMock(side_effect=[frame_1, frame_2]))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=[perception_1, perception_2]),
        latest_debug_artifacts=lambda: _debug_artifacts(tmp_path, "perception"),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(side_effect=[decision_1, decision_2]),
        latest_debug_artifacts=lambda: _debug_artifacts(tmp_path, "policy"),
    )
    legacy_executor = SimpleNamespace(
        execute=AsyncMock(side_effect=[executed_1, executed_2]),
        configure_run=Mock(),
    )
    verifier_service = SimpleNamespace(verify=AsyncMock(side_effect=[verification_1, verification_2]))
    recovery_manager = SimpleNamespace(
        recover=AsyncMock(return_value=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue"))
    )

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=legacy_executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
        environment=UnifiedEnvironment.BROWSER,
    )

    started = await loop.start_run(RunTaskRequest(intent="Open settings"))
    response = await loop.step_run(StepRequest(run_id=started.run_id))

    unified_state = loop.unified_state_for_run(started.run_id)
    assert isinstance(loop.executor_adapter, UnifiedBrowserExecutor)
    assert response.status is RunStatus.RUNNING
    assert unified_state is not None
    assert isinstance(unified_state, AgentRuntimeState)
    assert unified_state.environment is UnifiedEnvironment.BROWSER
    assert unified_state.retry_count == 1
    assert unified_state.last_strategy == "wait_then_retry"
    assert unified_state.goal_progress.status == "advanced"
    assert legacy_executor.execute.await_count == 2
    assert recovery_manager.recover.await_count == 1


@pytest.mark.asyncio
async def test_existing_agent_loop_replans_after_target_not_found(tmp_path: Path) -> None:
    run_store = FileBackedRunStore(root_dir=tmp_path / "runs")
    frame_1 = CaptureFrame(artifact_path=str(tmp_path / "frame_a.png"), width=1280, height=800, mime_type="image/png")
    frame_2 = CaptureFrame(artifact_path=str(tmp_path / "frame_b.png"), width=1280, height=800, mime_type="image/png")

    perception_1 = _perception(PageHint.FORM_PAGE, "Old target visible", target_id="old-btn")
    perception_2 = _perception(PageHint.FORM_PAGE, "New target visible", target_id="new-btn")
    decision_1 = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="old-btn"),
        rationale="Click old settings button.",
        confidence=0.8,
        active_subgoal="open settings",
    )
    decision_2 = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="new-btn"),
        rationale="Click refreshed settings button.",
        confidence=0.95,
        active_subgoal="open settings with refreshed target",
    )
    executed_1 = ExecutedAction(
        action=decision_1.action,
        success=False,
        detail="target missing",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )
    executed_2 = ExecutedAction(action=decision_2.action, success=True, detail="clicked refreshed target")
    verification_1 = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="The original target is no longer visible.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )
    verification_2 = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="The refreshed target worked.",
    )

    capture_service = SimpleNamespace(capture=AsyncMock(side_effect=[frame_1, frame_2]))
    perception_service = SimpleNamespace(
        perceive=AsyncMock(side_effect=[perception_1, perception_2]),
        latest_debug_artifacts=lambda: _debug_artifacts(tmp_path, "perception"),
    )
    policy_service = SimpleNamespace(
        choose_action=AsyncMock(side_effect=[decision_1, decision_2]),
        latest_debug_artifacts=lambda: _debug_artifacts(tmp_path, "policy"),
    )
    legacy_executor = SimpleNamespace(
        execute=AsyncMock(side_effect=[executed_1, executed_2]),
        configure_run=Mock(),
    )
    verifier_service = SimpleNamespace(verify=AsyncMock(side_effect=[verification_1, verification_2]))
    recovery_manager = SimpleNamespace(
        recover=AsyncMock(return_value=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue"))
    )

    loop = AgentLoop(
        capture_service=capture_service,
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=legacy_executor,
        verifier_service=verifier_service,
        recovery_manager=recovery_manager,
        environment=UnifiedEnvironment.BROWSER,
    )

    started = await loop.start_run(RunTaskRequest(intent="Open settings"))
    await loop.step_run(StepRequest(run_id=started.run_id))

    unified_state = loop.unified_state_for_run(started.run_id)
    assert unified_state is not None
    assert unified_state.retry_count == 1
    assert unified_state.last_strategy == "reperceive_and_replan"
    assert unified_state.latest_observation_id.endswith(":2")
    assert unified_state.latest_plan_id.endswith(":2")
    assert unified_state.goal_progress.subgoal == "open settings with refreshed target"
    assert legacy_executor.execute.await_count == 2
