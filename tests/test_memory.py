"""Focused tests for compact local advisory memory."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.models.common import FailureCategory, LoopStage, RunStatus
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.store.memory import FileBackedMemoryStore


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _perception(page_hint: str = "form_page") -> ScreenPerception:
    return ScreenPerception(
        summary="Form page visible.",
        page_hint=page_hint,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=320,
                y=180,
                width=300,
                height=28,
                is_interactable=True,
                confidence=0.95,
            )
        ],
        capture_artifact_path="runs/run-1/step_1/before.png",
        confidence=0.9,
    )


def test_memory_records_compact_failure_and_success_patterns() -> None:
    store = FileBackedMemoryStore(root_dir=_local_test_dir("test-memory-records") / "runs")
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )
    perception = _perception()
    failure_decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        rationale="Fill name",
        confidence=0.8,
        active_subgoal="fill_name",
    )
    executed_failure = ExecutedAction(
        action=failure_decision.action,
        success=False,
        detail="Execution failed: target not editable.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    failure_verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Type failed.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    recovery = RecoveryDecision(
        strategy=RecoveryStrategy.WAIT_AND_RETRY,
        message="Focus first.",
        retry_after_ms=500,
    )

    failure_records = store.record_step(
        state=state,
        perception=perception,
        decision=failure_decision,
        executed_action=executed_failure,
        verification=failure_verification,
        recovery=recovery,
    )

    success_decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="name-input", x=470, y=194),
        rationale="Focus name",
        confidence=0.95,
        active_subgoal="focus name-input",
    )
    executed_success = ExecutedAction(action=success_decision.action, success=True, detail="Clicked target.")
    success_verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="Focused input.",
    )

    success_records = store.record_step(
        state=state,
        perception=perception,
        decision=success_decision,
        executed_action=executed_success,
        verification=success_verification,
        recovery=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="Advance"),
    )

    assert any(record.key == "avoid_identical_type_retry" for record in failure_records)
    assert any(record.key == "click_before_type" for record in success_records)
    memory_lines = Path(store.memory_path).read_text(encoding="utf-8").splitlines()
    assert memory_lines
    assert all("artifact_path" not in line for line in memory_lines)


def test_memory_retrieval_returns_relevant_hints() -> None:
    store = FileBackedMemoryStore(root_dir=_local_test_dir("test-memory-hints") / "runs")
    state = AgentState(
        run_id="run-2",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )
    decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        rationale="Fill name",
        confidence=0.8,
        active_subgoal="fill_name",
    )
    executed = ExecutedAction(
        action=decision.action,
        success=False,
        detail="Execution failed: target not found.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Type failed.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    store.record_step(
        state=state,
        perception=_perception(),
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="Retry"),
    )

    hints = store.get_hints(
        benchmark=state.benchmark or "generic_task",
        page_hint=_perception().page_hint,
        subgoal="fill_name",
        recent_failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )

    keys = {hint.key for hint in hints}
    assert "click_before_type" in keys
    assert "avoid_identical_type_retry" in keys
