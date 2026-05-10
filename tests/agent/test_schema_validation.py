"""Focused validation tests for strict Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.agent.policy import parse_policy_output
from src.models.capture import CaptureFrame
from src.models.common import RunStatus
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts, StepLog
from src.models.perception import (
    ScreenPerception,
    UIElement,
    UIElementNameSource,
    UIElementType,
)
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus


def test_capture_frame_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CaptureFrame(
            artifact_path="artifacts/frame.png",
            width=1280,
            height=800,
            mime_type="image/png",
            unexpected=True,
        )



def test_ui_element_requires_positive_bounds() -> None:
    with pytest.raises(ValidationError):
        UIElement(
            element_id="compose",
            element_type=UIElementType.BUTTON,
            label="Compose",
            x=10,
            y=10,
            width=0,
            height=20,
            is_interactable=True,
        )



def test_agent_action_click_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        AgentAction(
            action_type=ActionType.CLICK,
            text="not allowed",
        )



def test_agent_action_type_requires_text() -> None:
    with pytest.raises(ValidationError):
        AgentAction(
            action_type=ActionType.TYPE,
        )



def test_agent_action_stop_rejects_payload_fields() -> None:
    with pytest.raises(ValidationError):
        AgentAction(
            action_type=ActionType.STOP,
            target_element_id="send-button",
        )


def test_agent_action_stop_allows_text_as_answer_payload() -> None:
    """STOP with text is the answer-extraction protocol for information-retrieval
    tasks (intent: 'find the year X' -> stop with text='1991'). The downstream
    benchmark runner reads action.text to feed the string_match evaluator."""
    action = AgentAction(action_type=ActionType.STOP, text="1991")
    assert action.action_type is ActionType.STOP
    assert action.text == "1991"


def test_agent_action_stop_still_rejects_non_text_payload_fields() -> None:
    """Allowing text on STOP must not relax other payload restrictions."""
    for forbidden_kwargs in (
        {"selector": "button.submit"},
        {"key": "enter"},
        {"url": "https://example.com"},
        {"wait_ms": 1000},
        {"x": 10, "y": 20},
    ):
        with pytest.raises(ValidationError):
            AgentAction(action_type=ActionType.STOP, **forbidden_kwargs)



def test_policy_decision_confidence_bounds() -> None:
    action = AgentAction(
        action_type=ActionType.WAIT,
        wait_ms=1000,
    )
    with pytest.raises(ValidationError):
        PolicyDecision(action=action, rationale="wait", confidence=1.5, active_subgoal="wait for inbox")


def test_parse_policy_output_normalizes_zero_wait_ms() -> None:
    decision = parse_policy_output(
        '{"action":{"action_type":"wait","wait_ms":0},"rationale":"Wait for the page to settle.","confidence":0.2,"active_subgoal":"wait","expected_change":"none"}'
    )
    assert decision.action.action_type is ActionType.WAIT
    assert decision.action.wait_ms == 1



def test_executed_action_wraps_strict_agent_action() -> None:
    action = AgentAction(
        action_type=ActionType.PRESS_KEY,
        key="c",
    )
    executed = ExecutedAction(action=action, success=True, detail="Pressed c")
    assert executed.action.key == "c"



def test_screen_perception_and_agent_state_validate_nested_models() -> None:
    perception = ScreenPerception(
        summary="Page is visible.",
        page_hint="unknown",
        capture_artifact_path="artifacts/frame.png",
        visible_elements=[
            UIElement(
                element_id="compose",
                element_type=UIElementType.BUTTON,
                label="Compose",
                x=20,
                y=40,
                width=120,
                height=36,
                is_interactable=True,
            )
        ],
    )
    state = AgentState(
        run_id="run-123",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.PENDING,
        observation_history=[perception],
    )
    assert state.observation_history[0].summary == "Page is visible."
    assert state.observation_history[0].visible_elements[0].primary_name == "Compose"
    assert state.observation_history[0].visible_elements[0].name_source is UIElementNameSource.LABEL



def test_verification_and_recovery_models_validate_ranges() -> None:
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Draft editor not visible yet.",
    )
    assert verification.status is VerificationStatus.UNCERTAIN

    with pytest.raises(ValidationError):
        RecoveryDecision(
            strategy=RecoveryStrategy.WAIT_AND_RETRY,
            message="Back off briefly.",
            retry_after_ms=0,
        )



def test_step_log_uses_debug_artifact_refs() -> None:
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait for the page to settle.", confidence=0.2, active_subgoal="wait for inbox")
    executed = ExecutedAction(action=action, success=True, detail="waited", artifact_path="runs/run-1/step_1/after.png")
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="continue",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="retry", retry_after_ms=1000)
    perception = ScreenPerception(
        summary="Inbox visible",
        page_hint="unknown",
        capture_artifact_path="runs/run-1/step_1/before.png",
        visible_elements=[],
    )
    debug = ModelDebugArtifacts(
        prompt_artifact_path="runs/run-1/step_1/perception_prompt.txt",
        raw_response_artifact_path="runs/run-1/step_1/perception_raw.txt",
        parsed_artifact_path="runs/run-1/step_1/perception_parsed.json",
    )
    log = StepLog(
        run_id="run-1",
        step_id="step_1",
        step_index=1,
        before_artifact_path="runs/run-1/step_1/before.png",
        after_artifact_path="runs/run-1/step_1/after.png",
        perception_debug=debug,
        policy_debug=debug.model_copy(
            update={
                "prompt_artifact_path": "runs/run-1/step_1/policy_prompt.txt",
                "raw_response_artifact_path": "runs/run-1/step_1/policy_raw.txt",
                "parsed_artifact_path": "runs/run-1/step_1/policy_decision.json",
            }
        ),
        perception=perception,
        policy_decision=decision,
        executed_action=executed,
        verification_result=verification,
        recovery_decision=recovery,
    )
    assert log.step_id == "step_1"
