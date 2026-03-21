"""Focused tests for Gemini policy parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.policy import GeminiPolicyService, PolicyError, parse_policy_output
from src.models.common import RunStatus
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, PolicyDecision
from src.models.state import AgentState


class StubGeminiClient:
    """Simple Gemini client stub for policy tests."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    async def generate_policy(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        raise NotImplementedError


def _perception() -> ScreenPerception:
    return ScreenPerception(
        summary="Auth-free form is visible with Name, Email, Message, and Submit controls.",
        page_hint="form_page",
        visible_elements=[
            UIElement(
                element_id="submit-button",
                element_type=UIElementType.BUTTON,
                label="Submit",
                x=24,
                y=116,
                width=108,
                height=36,
                is_interactable=True,
                confidence=0.97,
            ),
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
            ),
        ],
        capture_artifact_path="runs/run-1/step_1/before.png",
        confidence=0.92,
    )


def test_parse_policy_output_accepts_valid_json() -> None:
    raw_output = """
    {
      "action": {
        "action_type": "click",
        "target_element_id": "submit-button"
      },
      "rationale": "Submit the form.",
      "confidence": 0.91,
      "active_subgoal": "submit_form"
    }
    """

    decision = parse_policy_output(raw_output)

    assert isinstance(decision, PolicyDecision)
    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "submit-button"
    assert decision.rationale == "Submit the form."
    assert decision.confidence == pytest.approx(0.91)


def test_parse_policy_output_rejects_malformed_json() -> None:
    with pytest.raises(PolicyError, match="valid JSON"):
        parse_policy_output("not-json")


def test_parse_policy_output_rejects_schema_invalid_action() -> None:
    raw_output = """
    {
      "action": {
        "action_type": "type"
      },
      "rationale": "Fill the name field",
      "confidence": 0.75,
      "active_subgoal": "fill_name"
    }
    """

    with pytest.raises(PolicyError, match="strict schema"):
        parse_policy_output(raw_output)


def test_policy_decision_schema_rejects_invalid_action_payload() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision.model_validate(
            {
                "action": {
                    "action_type": "stop",
                    "target_element_id": "submit-button",
                },
                "rationale": "Stop on success.",
                "confidence": 1.0,
                "active_subgoal": "verify_success",
            }
        )


@pytest.mark.asyncio
async def test_policy_converts_unsafe_type_to_focus_click(tmp_path: Path) -> None:
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    client = StubGeminiClient(
        response='''{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Fill name.","confidence":0.9,"active_subgoal":"fill_name"}'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=1,
    )
    perception = _perception().model_copy(update={"capture_artifact_path": str(tmp_path / "run-1" / "step_1" / "before.png")})
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "name-input"
    assert decision.active_subgoal == "focus name-input"


@pytest.mark.asyncio
async def test_policy_allows_type_when_input_is_focused(tmp_path: Path) -> None:
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    client = StubGeminiClient(
        response='''{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Fill name.","confidence":0.9,"active_subgoal":"fill_name"}'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=2,
    )
    perception = _perception().model_copy(
        update={
            "capture_artifact_path": str(tmp_path / "run-1" / "step_2" / "before.png"),
            "focused_element_id": "name-input",
        }
    )
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)

    assert decision.action.action_type is ActionType.TYPE
    assert decision.action.text == "Alice"


@pytest.mark.asyncio
async def test_gemini_policy_service_writes_debug_artifacts(tmp_path: Path) -> None:
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    client = StubGeminiClient(
        response='''```json\n{"action":{"action_type":"click","target_element_id":"submit-button"},"rationale":"Submit the form.","confidence":0.89,"active_subgoal":"submit_form"}\n```'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="submit_form",
        step_count=1,
        retry_counts={"submit_form:failure": 1},
    )
    perception = _perception().model_copy(update={"capture_artifact_path": str(tmp_path / "run-1" / "step_1" / "before.png")})
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)
    debug = service.latest_debug_artifacts()

    assert client.last_prompt is not None
    assert "Complete the auth-free form and submit it successfully." in client.last_prompt
    assert "submit-button" in client.last_prompt
    assert "submit_form:failure" in client.last_prompt
    assert decision.action.action_type is ActionType.CLICK
    assert decision.rationale == "Submit the form."
    assert decision.active_subgoal == "submit_form"
    assert debug is not None
    assert Path(debug.prompt_artifact_path).exists()
    assert Path(debug.raw_response_artifact_path).exists()
    assert Path(debug.parsed_artifact_path).exists()
