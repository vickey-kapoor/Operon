from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.browser_computer_use import BrowserComputerUseBackend
from src.models.capture import CaptureFrame
from src.models.common import RunStatus
from src.models.state import AgentState


class _StubClient:
    async def generate_turn(self, *, contents: list[dict]) -> dict:
        return {
            "perception": {
                "summary": "Open the page and wait briefly.",
                "page_hint": "unknown",
                "visible_elements": [],
                "confidence": 0.8,
            },
            "function_call": {"name": "navigate", "args": {"url": "https://example.com"}},
            "function_calls": [
                {"name": "navigate", "args": {"url": "https://example.com"}},
                {"name": "wait_5_seconds", "args": {}},
            ],
            "model_content": {
                "parts": [
                    {"text": "Open the page and wait briefly."},
                    {"function_call": {"name": "navigate", "args": {"url": "https://example.com"}}},
                    {"function_call": {"name": "wait_5_seconds", "args": {}}},
                ]
            },
            "rationale": "Open the page and wait briefly.",
            "confidence": 0.8,
            "active_subgoal": "navigate",
        }

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        raise AssertionError("run_step should not be called in this test")

    @staticmethod
    def build_function_response_content(
        *,
        function_name: str,
        screenshot_path: str,
        current_url: str | None,
        error: str | None = None,
    ) -> dict:
        payload: dict[str, str] = {}
        if current_url:
            payload["url"] = current_url
        if error:
            payload["error"] = error
        return {
            "role": "user",
            "parts": [
                {"function_response": {"name": function_name, "response": payload}},
                {"inline_data": {"mime_type": "image/png", "data": "AAAA"}},
            ],
        }


@pytest.mark.asyncio
async def test_browser_backend_builds_batch_action_for_multiple_function_calls(tmp_path: Path) -> None:
    prompt_path = tmp_path / "browser_computer_use_prompt.txt"
    prompt_path.write_text("intent={intent}", encoding="utf-8")
    screenshot = CaptureFrame(
        artifact_path=str(tmp_path / "screen.png"),
        width=1200,
        height=800,
        mime_type="image/png",
    )
    Path(screenshot.artifact_path).write_bytes(b"png")
    backend = BrowserComputerUseBackend(client=_StubClient(), prompt_path=prompt_path)
    state = AgentState(run_id="run-1", intent="Open example.com", status=RunStatus.RUNNING)

    perception = await backend.perceive(screenshot, state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type.value == "batch"
    assert decision.action.actions is not None
    assert [item.action_type.value for item in decision.action.actions] == ["navigate", "wait"]
