"""Gemini computer use on the desktop: the desktop environment, launch_app, screen coordinates."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.actions.translation import normalize_computer_use_function_call
from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.backends.desktop_computer_use import (
    GEMINI_LAUNCH_APP_FUNCTION,
    DesktopComputerUseBackend,
)
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState


def _call_reply(name: str, args: dict) -> dict:
    call = {"name": name, "args": args}
    return {
        "perception": {"summary": f"Tool call: {name}.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": name,
        "confidence": 0.7,
        "active_subgoal": name,
        "model_content": {"parts": [{"function_call": call}]},
    }


def _text_reply(text: str) -> dict:
    return {
        "perception": {"summary": text, "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "action": {"action_type": "wait", "wait_ms": 1000},
        "rationale": text,
        "confidence": 0.4,
        "model_content": {"parts": [{"text": text}]},
    }


class ScriptedClient:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = replies
        self.turns: list[list[dict]] = []
        self.function_responses: list[dict] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        return self._replies[min(len(self.turns), len(self._replies)) - 1]

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        response = {"name": function_name, "response": {"url": current_url} if current_url else {}}
        self.function_responses.append(response)
        return {"role": "user", "parts": [{"function_response": response}, {"inline_data": {"data": ""}}]}


def _backend(client: ScriptedClient) -> DesktopComputerUseBackend:
    return DesktopComputerUseBackend(client=client, prompt_path=prompts_dir() / "desktop_computer_use_prompt.txt")


def _frame(tmp_path: Path, name: str = "frame.png") -> CaptureFrame:
    path = tmp_path / name
    path.write_bytes(b"png")
    return CaptureFrame(artifact_path=str(path), width=1920, height=1080)


def _state(*actions: AgentAction) -> AgentState:
    return AgentState(
        run_id="run-1",
        intent="Open Notepad and type hello",
        status=RunStatus.RUNNING,
        action_history=[ExecutedAction(action=action, success=True, detail="ok") for action in actions],
    )


def test_payload_asks_for_the_desktop_environment_and_declares_launch_app() -> None:
    client = GeminiComputerUseHttpClient(
        api_key="test-key",
        model="gemini-3.5-flash",
        environment="ENVIRONMENT_DESKTOP",
        function_declarations=[GEMINI_LAUNCH_APP_FUNCTION],
    )
    payload = client._build_payload(contents=[{"role": "user", "parts": [{"text": "go"}]}])
    assert payload["tools"][0] == {"computerUse": {"environment": "ENVIRONMENT_DESKTOP"}}
    (declaration,) = payload["tools"][1]["functionDeclarations"]
    assert declaration["name"] == "launch_app"
    assert declaration["parameters"]["properties"]["app"]["enum"] == sorted(APP_LAUNCH_TARGETS)


def test_browser_payload_has_no_extra_tools() -> None:
    payload = GeminiComputerUseHttpClient(api_key="k", model="gemini-3.5-flash")._build_payload(contents=[])
    assert payload["tools"] == [{"computerUse": {"environment": "ENVIRONMENT_BROWSER"}}]


def test_desktop_actions_reach_screen_coordinates() -> None:
    def act(name: str, **args) -> dict:
        return normalize_computer_use_function_call({"name": name, "args": args}, screen_width=1920, screen_height=1080)

    assert act("launch_app", app="notepad") == {"action_type": "launch_app", "text": "notepad"}
    assert act("click", x=999, y=999, intent="ok") == {"action_type": "click", "x": 1919, "y": 1079}
    assert act("triple_click", x=0, y=0) == {"action_type": "triple_click", "x": 0, "y": 0}
    assert act("hotkey", keys=["ctrl", "s"]) == {"action_type": "hotkey", "key": "ctrl+s"}


async def test_launch_app_then_type_then_done(tmp_path: Path) -> None:
    client = ScriptedClient([
        _call_reply("launch_app", {"app": "notepad"}),
        _call_reply("type", {"text": "hello", "intent": "type the text"}),
        _text_reply("Notepad shows hello."),
    ])
    backend = _backend(client)

    state = _state()
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "1.png"), state))
    assert decision.action.action_type is ActionType.LAUNCH_APP and decision.action.text == "notepad"
    prompt = client.turns[0][0]["parts"][0]["text"]
    assert "Open Notepad and type hello" in prompt and "launch_app" in prompt

    state = _state(decision.action)
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "2.png"), state))
    assert decision.action.action_type is ActionType.TYPE and decision.action.text == "hello"
    # The launch_app call was answered, with no page URL on the desktop.
    assert client.function_responses[0] == {"name": "launch_app", "response": {}}

    state = _state(AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad"), decision.action)
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "3.png"), state))
    assert decision.action.action_type is ActionType.STOP


def test_launch_app_without_an_app_name_is_a_compatibility_error() -> None:
    for args in ({}, {"app": ""}, {"app": None}):
        with pytest.raises(BackendCompatibilityError, match="app name"):
            normalize_computer_use_function_call({"name": "launch_app", "args": args}, screen_width=10, screen_height=10)


def test_backend_refuses_a_browser_client() -> None:
    browser_client = GeminiComputerUseHttpClient(api_key="k", model="gemini-3.5-flash")
    with pytest.raises(ValueError, match="ENVIRONMENT_DESKTOP"):
        DesktopComputerUseBackend(client=browser_client, prompt_path=prompts_dir() / "desktop_computer_use_prompt.txt")
    no_launch_app = GeminiComputerUseHttpClient(api_key="k", model="gemini-3.5-flash", environment="ENVIRONMENT_DESKTOP")
    with pytest.raises(ValueError, match="launch_app"):
        DesktopComputerUseBackend(client=no_launch_app, prompt_path=prompts_dir() / "desktop_computer_use_prompt.txt")
