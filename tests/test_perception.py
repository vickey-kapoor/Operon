"""Focused tests for Gemini perception parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.perception import (
    GeminiPerceptionService,
    PerceptionError,
    parse_perception_output,
)
from src.models.capture import CaptureFrame
from src.models.common import RunStatus
from src.models.perception import PageHint
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



def test_google_sign_in_page_classification() -> None:
    raw_output = """
    {
      "summary": "Google Sign in page is visible with Email or phone input.",
      "page_hint": "google_sign_in",
      "focused_element_id": null,
      "confidence": 0.95,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.GOOGLE_SIGN_IN



def test_gmail_compose_classification() -> None:
    raw_output = """
    {
      "summary": "Gmail compose draft dialog is visible.",
      "page_hint": "gmail_compose",
      "focused_element_id": "to-input",
      "confidence": 0.91,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_2/before.png")

    assert perception.page_hint is PageHint.GMAIL_COMPOSE



def test_gmail_inbox_classification() -> None:
    raw_output = """
    {
      "summary": "Gmail inbox is visible with compose button.",
      "page_hint": "gmail_inbox",
      "focused_element_id": null,
      "confidence": 0.93,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.GMAIL_INBOX



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

    assert perception.page_hint is PageHint.GOOGLE_SIGN_IN



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


@pytest.mark.asyncio
async def test_gemini_perception_service_writes_debug_artifacts(tmp_path: Path) -> None:
    prompt_path = tmp_path / "perception_prompt.txt"
    prompt_path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nPrev: {previous_summary}",
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "run-123" / "step_1" / "before.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path.write_bytes(b"png")

    client = StubGeminiClient(
        response='''```json\n{"summary":"Compose dialog visible.","page_hint":"gmail_compose","focused_element_id":"to-input","confidence":0.88,"visible_elements":[{"element_id":"to-input","element_type":"input","label":"To","x":320,"y":180,"width":300,"height":28,"is_interactable":true,"confidence":0.94}]}\n```'''
    )
    service = GeminiPerceptionService(gemini_client=client, prompt_path=prompt_path)
    state = AgentState(
        run_id="run-123",
        intent="Create a Gmail draft and stop before send.",
        status=RunStatus.RUNNING,
        current_subgoal="open compose",
        step_count=2,
    )
    frame = CaptureFrame(artifact_path=str(screenshot_path), width=1280, height=800, mime_type="image/png")

    perception = await service.perceive(frame, state)
    debug = service.latest_debug_artifacts()

    assert client.last_screenshot_path == str(screenshot_path)
    assert client.last_prompt is not None
    assert "Create a Gmail draft and stop before send." in client.last_prompt
    assert "open compose" in client.last_prompt
    assert perception.focused_element_id == "to-input"
    assert perception.capture_artifact_path == str(screenshot_path)
    assert perception.page_hint is PageHint.GMAIL_COMPOSE
    assert debug is not None
    assert Path(debug.prompt_artifact_path).exists()
    assert Path(debug.raw_response_artifact_path).exists()
    assert Path(debug.parsed_artifact_path).exists()
