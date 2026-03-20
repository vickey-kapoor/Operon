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
        summary="Gmail inbox is visible with a Compose button.",
        page_hint="gmail_inbox",
        visible_elements=[
            UIElement(
                element_id="compose-button",
                element_type=UIElementType.BUTTON,
                label="Compose",
                x=24,
                y=116,
                width=108,
                height=36,
                is_interactable=True,
                confidence=0.97,
            ),
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
        "target_element_id": "compose-button"
      },
      "rationale": "Open the Gmail compose dialog.",
      "confidence": 0.91,
      "active_subgoal": "open compose"
    }
    """

    decision = parse_policy_output(raw_output)

    assert isinstance(decision, PolicyDecision)
    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "compose-button"
    assert decision.rationale == "Open the Gmail compose dialog."
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
      "rationale": "Fill the subject line",
      "confidence": 0.75,
      "active_subgoal": "fill subject"
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
                    "target_element_id": "send-button",
                },
                "rationale": "Stop before send.",
                "confidence": 1.0,
                "active_subgoal": "stop before send",
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
        response='''{"action":{"action_type":"type","target_element_id":"subject-input","text":"Hello"},"rationale":"Fill subject.","confidence":0.9,"active_subgoal":"fill subject"}'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Create a Gmail draft and stop before send.",
        status=RunStatus.RUNNING,
        current_subgoal="fill subject",
        step_count=1,
    )
    perception = _perception().model_copy(update={"capture_artifact_path": str(tmp_path / "run-1" / "step_1" / "before.png")})
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "subject-input"
    assert decision.active_subgoal == "focus subject-input"


@pytest.mark.asyncio
async def test_policy_allows_type_when_input_is_focused(tmp_path: Path) -> None:
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    client = StubGeminiClient(
        response='''{"action":{"action_type":"type","target_element_id":"subject-input","text":"Hello"},"rationale":"Fill subject.","confidence":0.9,"active_subgoal":"fill subject"}'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Create a Gmail draft and stop before send.",
        status=RunStatus.RUNNING,
        current_subgoal="fill subject",
        step_count=2,
    )
    perception = _perception().model_copy(
        update={
            "capture_artifact_path": str(tmp_path / "run-1" / "step_2" / "before.png"),
            "focused_element_id": "subject-input",
        }
    )
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)

    assert decision.action.action_type is ActionType.TYPE
    assert decision.action.text == "Hello"


@pytest.mark.asyncio
async def test_gemini_policy_service_writes_debug_artifacts(tmp_path: Path) -> None:
    prompt_path = tmp_path / "policy_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    client = StubGeminiClient(
        response='''```json\n{"action":{"action_type":"click","target_element_id":"compose-button"},"rationale":"Open the compose dialog.","confidence":0.89,"active_subgoal":"open compose"}\n```'''
    )
    service = GeminiPolicyService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-1",
        intent="Create a Gmail draft and stop before send.",
        status=RunStatus.RUNNING,
        current_subgoal="open compose",
        step_count=1,
        retry_counts={"open compose:failure": 1},
    )
    perception = _perception().model_copy(update={"capture_artifact_path": str(tmp_path / "run-1" / "step_1" / "before.png")})
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await service.choose_action(state, perception)
    debug = service.latest_debug_artifacts()

    assert client.last_prompt is not None
    assert "Create a Gmail draft and stop before send." in client.last_prompt
    assert "compose-button" in client.last_prompt
    assert "open compose:failure" in client.last_prompt
    assert decision.action.action_type is ActionType.CLICK
    assert decision.rationale == "Open the compose dialog."
    assert decision.active_subgoal == "open compose"
    assert debug is not None
    assert Path(debug.prompt_artifact_path).exists()
    assert Path(debug.raw_response_artifact_path).exists()
    assert Path(debug.parsed_artifact_path).exists()
