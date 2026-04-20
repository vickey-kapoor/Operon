"""Focused tests for deterministic recovery rules."""

from pathlib import Path

import pytest

from src.agent.policy import GeminiPolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.store.memory import FileBackedMemoryStore


class _UnusedGeminiClient:
    async def generate_policy(self, prompt: str) -> str:
        raise AssertionError("Rule-first path should have handled this case before LLM fallback.")

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        raise NotImplementedError


def _decision() -> PolicyDecision:
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    return PolicyDecision(action=action, rationale="Open compose.", confidence=0.8, active_subgoal="open compose")


def _executed() -> ExecutedAction:
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    return ExecutedAction(action=action, success=True, detail="clicked compose")


@pytest.mark.asyncio
async def test_recovery_retry_mapping() -> None:
    manager = RuleBasedRecoveryManager()
    state = AgentState(run_id="run-1", intent="Create draft", status="running")
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Action failed",
        failure_type=VerificationFailureType.ACTION_FAILED,
    )

    result = await manager.recover(state, _decision(), _executed(), verification)

    assert result.strategy is RecoveryStrategy.RETRY_SAME_STEP


@pytest.mark.asyncio
async def test_recovery_advance_mapping() -> None:
    manager = RuleBasedRecoveryManager()
    state = AgentState(run_id="run-2", intent="Create draft", status="running")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="Expected outcome met",
    )

    result = await manager.recover(state, _decision(), _executed(), verification)

    assert result.strategy is RecoveryStrategy.ADVANCE


@pytest.mark.asyncio
async def test_recovery_stop_mapping() -> None:
    manager = RuleBasedRecoveryManager()
    state = AgentState(run_id="run-3", intent="Create draft", status="running")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=True,
        reason="Stop before send",
        failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
    )

    result = await manager.recover(state, _decision(), _executed(), verification)

    assert result.strategy is RecoveryStrategy.STOP


@pytest.mark.asyncio
async def test_recovery_retry_limit_behavior() -> None:
    manager = RuleBasedRecoveryManager()
    decision = _decision()
    retry_key = f"{decision.active_subgoal}:id:compose:{FailureCategory.EXPECTED_OUTCOME_NOT_MET.value}"
    state = AgentState(
        run_id="run-4",
        intent="Create draft",
        status="running",
        retry_counts={retry_key: 4},
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Action failed repeatedly",
        failure_type=VerificationFailureType.EXPECTED_OUTCOME_NOT_MET,
        failure_category=FailureCategory.EXPECTED_OUTCOME_NOT_MET,
    )

    result = await manager.recover(state, decision, _executed(), verification)

    assert result.strategy is RecoveryStrategy.STOP


@pytest.mark.asyncio
async def test_failed_type_uses_focus_first_recovery_path() -> None:
    manager = RuleBasedRecoveryManager()
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="subject-input", text="Hello")
    decision = PolicyDecision(action=action, rationale="Fill subject.", confidence=0.8, active_subgoal="fill subject")
    state = AgentState(run_id="run-5", intent="Create draft", status="running")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: Resolved type target is not editable.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Executed action reported failure.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )

    result = await manager.recover(state, decision, executed, verification)

    assert result.strategy is RecoveryStrategy.WAIT_AND_RETRY
    assert state.current_subgoal == "focus subject-input"
    assert result.failure_category is FailureCategory.EXECUTION_TARGET_NOT_EDITABLE


@pytest.mark.asyncio
async def test_recovery_focus_subgoal_drives_next_policy_click(tmp_path: Path) -> None:
    manager = RuleBasedRecoveryManager()
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="subject-input", text="Hello")
    decision = PolicyDecision(action=action, rationale="Fill subject.", confidence=0.8, active_subgoal="fill subject")
    state = AgentState(run_id="run-6", intent="Create draft", status="running")
    executed = ExecutedAction(
        action=action,
        success=False,
        detail="Execution failed: Resolved type target is not editable.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Executed action reported failure.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )

    recovery = await manager.recover(state, decision, executed, verification)
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=_UnusedGeminiClient(), prompt_path=prompt_path),
        memory_store=FileBackedMemoryStore(root_dir=tmp_path / "runs"),
    )
    perception_path = tmp_path / "runs" / "run-6" / "step_2" / "before.png"
    perception_path.parent.mkdir(parents=True, exist_ok=True)
    next_decision = await coordinator.choose_action(
        state,
        ScreenPerception(
            summary="Compose form visible.",
            page_hint="gmail_compose",
            visible_elements=[
                UIElement(
                    element_id="subject-input",
                    element_type=UIElementType.INPUT,
                    label="Subject",
                    x=320,
                    y=180,
                    width=300,
                    height=28,
                    is_interactable=True,
                    confidence=0.95,
                )
            ],
            capture_artifact_path=str(perception_path),
        ),
    )

    assert recovery.strategy is RecoveryStrategy.WAIT_AND_RETRY
    assert state.current_subgoal == "focus subject-input"
    assert next_decision.action.action_type is ActionType.CLICK
    assert next_decision.action.target_element_id == "subject-input"
