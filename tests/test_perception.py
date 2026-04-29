"""Focused tests for Gemini perception parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.perception import (
    GeminiPerceptionService,
    PerceptionError,
    PerceptionLowQualityError,
    parse_perception_output,
)
from src.agent.selector import DeterministicTargetSelector
from src.models.capture import CaptureFrame
from src.models.common import RunStatus
from src.models.perception import PageHint, UIElementNameSource, UIElementType
from src.models.selector import TargetIntent, TargetIntentAction
from src.models.state import AgentState


class StubGeminiClient:
    """Simple Gemini client stub for perception tests."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_screenshot_path: str | None = None

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        self.last_prompt = prompt
        self.last_screenshot_path = screenshot_path
        return self.response

    async def generate_policy(self, prompt: str) -> str:
        raise NotImplementedError

    def latest_perception_scale_ratio(self) -> float:
        return 1.0


class SequencedGeminiClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)

    async def generate_policy(self, prompt: str) -> str:
        raise NotImplementedError

    def latest_perception_scale_ratio(self) -> float:
        return 1.0


def test_form_page_classification() -> None:
    raw_output = """
    {
      "summary": "Contact form is visible with Name, Email, Message, and Submit fields.",
      "page_hint": "form_page",
      "focused_element_id": "name-input",
      "confidence": 0.96,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_PAGE


def test_form_success_classification() -> None:
    raw_output = """
    {
      "summary": "Thank you, the form was submitted successfully.",
      "page_hint": "form_success",
      "focused_element_id": null,
      "confidence": 0.97,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_2/before.png")

    assert perception.page_hint is PageHint.FORM_SUCCESS


def test_missing_page_hint_uses_small_summary_fallback() -> None:
    raw_output = """
    {
      "summary": "Google sign in page asking for email or phone.",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    # Fallback classifier no longer detects app-specific pages; LLM provides page_hint.
    assert perception.page_hint is PageHint.UNKNOWN


def test_missing_page_hint_uses_form_summary_fallback() -> None:
    raw_output = """
    {
      "summary": "Thank you for your message, the form was submitted successfully.",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_SUCCESS


def test_null_page_hint_is_rejected() -> None:
    raw_output = """
    {
      "summary": "Google sign in page asking for email or phone.",
      "page_hint": null,
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    with pytest.raises(PerceptionError, match="strict schema"):
        parse_perception_output(raw_output, "runs/run-1/step_1/before.png")


def test_empty_visible_element_label_is_normalized_and_accepted() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "email-input",
          "element_type": "input",
          "label": "   ",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    assert perception.visible_elements[0].label is None
    assert perception.visible_elements[0].primary_name == "unlabeled_input"
    assert perception.visible_elements[0].name_source is UIElementNameSource.SYNTHETIC
    assert perception.visible_elements[0].is_unlabeled is True
    assert perception.visible_elements[0].usable_for_targeting is False


def test_missing_visible_element_label_is_normalized_from_element_type() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "email-input",
          "element_type": "input",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    assert perception.visible_elements[0].label is None
    assert perception.visible_elements[0].primary_name == "unlabeled_input"
    assert perception.visible_elements[0].name_source is UIElementNameSource.SYNTHETIC


def test_whitespace_weak_semantic_fields_are_normalized_to_none() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": "  ",
          "text": " ",
          "placeholder": "  ",
          "name": "\\n",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    element = perception.visible_elements[0]
    assert element.label is None
    assert element.text is None
    assert element.placeholder is None
    assert element.name is None
    assert element.primary_name == "unlabeled_input"


def test_structurally_invalid_visible_element_still_fails_cleanly() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": "",
          "x": 100,
          "y": 200,
          "width": 0,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.5
        }
      ]
    }
    """

    with pytest.raises(PerceptionError, match="strict schema"):
        parse_perception_output(raw_output, "runs/run-1/step_1/before.png")


@pytest.mark.asyncio
async def test_gemini_perception_service_writes_debug_artifacts(tmp_path: Path) -> None:
    prompt_path = tmp_path / "perception_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nPrev: {previous_page_hint}",
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "run-123" / "step_1" / "before.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png")

    client = StubGeminiClient(
        response='''```json\n{"summary":"Form page visible with name, email, and message fields.","page_hint":"form_page","focused_element_id":"name-input","confidence":0.88,"visible_elements":[{"element_id":"name-input","element_type":"input","label":"Name","x":320,"y":180,"width":300,"height":28,"is_interactable":true,"confidence":0.94}]}\n```'''
    )
    service = GeminiPerceptionService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-123",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=2,
    )
    frame = CaptureFrame(artifact_path=str(screenshot_path), width=1280, height=800, mime_type="image/png")

    perception = await service.perceive(frame, state)
    debug = service.latest_debug_artifacts()

    assert client.last_screenshot_path == str(screenshot_path)
    assert client.last_prompt is not None
    assert "Complete the auth-free form and submit it successfully." in client.last_prompt
    assert "fill_name" in client.last_prompt
    assert perception.focused_element_id == "name-input"
    assert perception.capture_artifact_path == str(screenshot_path)
    assert perception.page_hint is PageHint.FORM_PAGE
    assert debug is not None
    assert Path(debug.prompt_artifact_path).exists()
    assert Path(debug.raw_response_artifact_path).exists()
    assert Path(debug.parsed_artifact_path).exists()


@pytest.mark.asyncio
async def test_gemini_perception_service_retries_low_quality_once_and_recovers(tmp_path: Path) -> None:
    prompt_path = tmp_path / "perception_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nPrev: {previous_page_hint}",
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "run-retry" / "step_1" / "before.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png")

    client = SequencedGeminiClient(
        responses=[
            """{"summary":"Form page visible.","page_hint":"form_page","focused_element_id":null,"confidence":0.88,"visible_elements":[{"element_id":"name-input","element_type":"input","label":" ","x":320,"y":180,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"email-input","element_type":"input","label":" ","x":320,"y":220,"width":300,"height":28,"is_interactable":true,"confidence":0.94}]}""",
            """{"summary":"Form page visible with name, email, message, and submit.","page_hint":"form_page","focused_element_id":"name-input","confidence":0.92,"visible_elements":[{"element_id":"name-input","element_type":"input","label":"Name","x":320,"y":180,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"email-input","element_type":"input","label":"Email","x":320,"y":220,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"message-input","element_type":"input","label":"Message","x":320,"y":260,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"submit-button","element_type":"button","label":"Submit","x":320,"y":300,"width":120,"height":32,"is_interactable":true,"confidence":0.94}]}""",
        ]
    )
    service = GeminiPerceptionService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-retry",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=0,
    )
    frame = CaptureFrame(artifact_path=str(screenshot_path), width=1280, height=800, mime_type="image/png")

    perception = await service.perceive(frame, state)
    debug = service.latest_debug_artifacts()

    assert perception.focused_element_id == "name-input"
    assert len(client.prompts) == 2
    assert "Do not emit empty strings. Use null for missing fields." in client.prompts[1]
    assert debug is not None
    assert debug.retry_log_artifact_path is not None
    assert Path(debug.retry_log_artifact_path).exists()
    assert "zero usable candidates after salvage" in Path(debug.retry_log_artifact_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_gemini_perception_service_salvages_unlabeled_heavy_output_after_retry(tmp_path: Path) -> None:
    prompt_path = tmp_path / "perception_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nPrev: {previous_page_hint}",
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "run-retry-salvage" / "step_1" / "before.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png")

    response = """{"summary":"Form page visible with name, email, and message fields.","page_hint":"form_page","focused_element_id":null,"confidence":0.88,"visible_elements":[{"element_id":"name-input","element_type":"input","label":" ","x":320,"y":180,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"email-input","element_type":"input","label":" ","x":320,"y":220,"width":300,"height":28,"is_interactable":true,"confidence":0.94},{"element_id":"message-input","element_type":"input","label":" ","x":320,"y":260,"width":300,"height":28,"is_interactable":true,"confidence":0.94}]}"""
    client = SequencedGeminiClient(responses=[response, response])
    service = GeminiPerceptionService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-retry-salvage",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=0,
    )
    frame = CaptureFrame(artifact_path=str(screenshot_path), width=1280, height=800, mime_type="image/png")

    perception = await service.perceive(frame, state)

    debug = service.latest_debug_artifacts()
    assert debug is not None
    assert debug.retry_log_artifact_path is not None
    assert debug.diagnostics_artifact_path is not None
    retry_log = Path(debug.retry_log_artifact_path).read_text(encoding="utf-8")
    diagnostics = json.loads(Path(debug.diagnostics_artifact_path).read_text(encoding="utf-8"))
    assert retry_log.count("attempt=") == 3
    assert "salvage_mode=true" in retry_log
    assert diagnostics["final_decision"] == "salvaged"
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_result"]["quality_metrics"]["candidate_count"] >= 1
    assert any(element.usable_for_targeting for element in perception.visible_elements if element.element_type is UIElementType.INPUT)
    assert any(element.confidence <= 0.49 for element in perception.visible_elements if element.element_type is UIElementType.INPUT)
    assert sum(1 for element in perception.visible_elements if element.usable_for_targeting) >= 1


@pytest.mark.asyncio
async def test_gemini_perception_service_raises_low_quality_after_salvage_when_no_candidates_exist(tmp_path: Path) -> None:
    prompt_path = tmp_path / "perception_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nPrev: {previous_page_hint}",
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "run-retry-fail" / "step_1" / "before.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png")

    response = """{"summary":"Sparse page.","page_hint":"unknown","focused_element_id":null,"confidence":0.88,"visible_elements":[{"element_id":"title","element_type":"text","text":"Welcome","x":20,"y":20,"width":120,"height":24,"is_interactable":false,"confidence":0.94}]}"""
    client = SequencedGeminiClient(responses=[response, response])
    service = GeminiPerceptionService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-retry-fail",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        step_count=0,
    )
    frame = CaptureFrame(artifact_path=str(screenshot_path), width=1280, height=800, mime_type="image/png")

    with pytest.raises(PerceptionLowQualityError, match="low quality"):
        await service.perceive(frame, state)

    debug = service.latest_debug_artifacts()
    assert debug is not None
    assert debug.diagnostics_artifact_path is not None
    diagnostics = json.loads(Path(debug.diagnostics_artifact_path).read_text(encoding="utf-8"))
    assert diagnostics["final_decision"] == "aborted_low_quality"
    assert diagnostics["salvage_attempted"] is True
    assert diagnostics["salvage_reason"] == "zero usable_for_targeting elements"


def test_unlabeled_heavy_perception_is_salvaged_into_selector_candidates() -> None:
    raw_output = """
    {
      "summary": "Form page with visible labels and fields.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-label",
          "element_type": "text",
          "text": "Name",
          "x": 320,
          "y": 140,
          "width": 80,
          "height": 24,
          "is_interactable": false,
          "confidence": 0.94
        },
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": " ",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        },
        {
          "element_id": "submit-label",
          "element_type": "text",
          "text": "Submit",
          "x": 320,
          "y": 260,
          "width": 90,
          "height": 24,
          "is_interactable": false,
          "confidence": 0.94
        },
        {
          "element_id": "submit-button",
          "element_type": "button",
          "label": "",
          "x": 320,
          "y": 300,
          "width": 120,
          "height": 32,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")
    selector = DeterministicTargetSelector()
    result = selector.select(
        perception,
        TargetIntent(
            action=TargetIntentAction.CLICK,
            target_text="submit",
            expected_element_types=[UIElementType.BUTTON],
            expected_section="form",
        ),
    )

    assert any(element.usable_for_targeting for element in perception.visible_elements)
    assert result.trace.candidate_count >= 1
    assert result.selected is not None
    assert result.selected.element_id == "submit-button"


def test_form_page_with_nearby_text_and_unlabeled_inputs_is_minimally_viable() -> None:
    raw_output = """
    {
      "summary": "Practice form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.84,
      "visible_elements": [
        {
          "element_id": "name-label",
          "element_type": "text",
          "text": "Name",
          "x": 280,
          "y": 150,
          "width": 80,
          "height": 20,
          "is_interactable": false,
          "confidence": 0.92
        },
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": " ",
          "x": 280,
          "y": 180,
          "width": 160,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.9
        },
        {
          "element_id": "email-label",
          "element_type": "text",
          "text": "Email",
          "x": 280,
          "y": 220,
          "width": 80,
          "height": 20,
          "is_interactable": false,
          "confidence": 0.92
        },
        {
          "element_id": "email-input",
          "element_type": "input",
          "label": "",
          "x": 280,
          "y": 250,
          "width": 160,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.9
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_PAGE
    assert len([element for element in perception.visible_elements if element.is_interactable]) == 2
    assert any(element.primary_name == "Name" for element in perception.visible_elements if element.element_id == "name-input")
    assert any(element.usable_for_targeting for element in perception.visible_elements if element.element_id == "email-input")
