"""Claude computer use on the desktop: launch_app tool, coordinate scaling, rule overrides and wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from operon.agent.actions.translation import normalize_claude_computer_calls
from operon.agent.backends.desktop_claude_computer_use import (
    DesktopClaudeComputerUseBackend,
)
from operon.clients.anthropic_computer_use import (
    LAUNCH_APP_TOOL,
    AnthropicComputerUseHttpClient,
)
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState


def _call(name: str, call_id: str, **tool_input) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "toolset_name": "computer", "input": tool_input}


def _launch(call_id: str, app: str) -> dict:
    return {"type": "tool_use", "id": call_id, "name": "launch_app", "input": {"app": app}}


def _reply(*blocks: dict, stop_reason: str = "tool_use") -> dict:
    return {"content": list(blocks), "stop_reason": stop_reason, "usage": {"input_tokens": 10, "output_tokens": 5}}


class ScriptedClient:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = replies
        self.requests: list[list[dict]] = []
        self.systems: list[str] = []

    async def generate_turn(self, *, system: str, messages: list[dict]) -> dict:
        self.requests.append([dict(m) for m in messages])
        self.systems.append(system)
        return self._replies[min(len(self.requests), len(self._replies)) - 1]

    def latest_usage(self):
        return None


def _backend(client: ScriptedClient) -> DesktopClaudeComputerUseBackend:
    return DesktopClaudeComputerUseBackend(
        client=client, prompt_path=prompts_dir() / "desktop_claude_computer_use_prompt.txt"
    )


def _frame(tmp_path: Path, name: str = "frame.png") -> CaptureFrame:
    path = tmp_path / name
    Image.new("RGB", (1920, 1080), "white").save(path)
    return CaptureFrame(artifact_path=str(path), width=1920, height=1080)


def _state(*executed: ExecutedAction) -> AgentState:
    return AgentState(
        run_id="run-1",
        intent="Open Command Prompt and run echo hi",
        status=RunStatus.RUNNING,
        action_history=list(executed),
    )


async def test_launch_app_calls_become_launch_app_actions(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_launch("t1", "command prompt"))])
    backend = _backend(client)
    state = _state()

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.LAUNCH_APP
    assert decision.action.text == "command prompt"
    assert "Windows" in client.systems[0]
    assert "Open Command Prompt" in client.requests[0][0]["content"][0]["text"]


async def test_clicks_are_scaled_from_the_downscaled_screenshot(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_call("left_click", "t1", coordinate=[640, 700]))])
    backend = _backend(client)
    state = _state()

    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path), state))

    # 1920x1080 is sent as 1280x720: the taskbar at y=700 is y=1050 on screen.
    assert (decision.action.x, decision.action.y) == (960, 1050)


async def test_launch_app_is_answered_without_a_toolset_name(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_launch("t1", "notepad")), _reply({"type": "text", "text": "Done."}, stop_reason="end_turn")])
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    launched = ExecutedAction(action=AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad"), success=True, detail="Launched 'notepad.exe'")
    await backend.perceive(_frame(tmp_path, "b.png"), _state(launched))

    result = next(b for b in client.requests[1][-1]["content"] if b["type"] == "tool_result")
    assert result["tool_use_id"] == "t1"
    assert "toolset_name" not in result and "is_error" not in result


async def test_calls_a_rule_replaced_are_reported_as_not_executed(tmp_path: Path) -> None:
    """A deterministic rule can act instead of Claude; Claude must not think its click ran."""
    client = ScriptedClient([_reply(_call("left_click", "t1", coordinate=[10, 10])), _reply(_call("screenshot", "t2"))])
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    by_rule = ExecutedAction(action=AgentAction(action_type=ActionType.LAUNCH_APP, text="cmd"), success=True, detail="Launched 'cmd.exe'")
    await backend.perceive(_frame(tmp_path, "b.png"), _state(by_rule))

    answer = client.requests[1][-1]["content"]
    result = next(b for b in answer if b["type"] == "tool_result")
    assert result["is_error"] is True
    assert "the agent ran launch_app" in result["content"]
    assert any(b["type"] == "image" for b in answer)


def test_normalize_maps_launch_app_in_a_batch() -> None:
    action = normalize_claude_computer_calls(
        [{"name": "launch_app", "input": {"app": "notepad"}}, {"name": "type", "input": {"text": "hi"}}],
        scale_x=1.5, scale_y=1.5, screen_width=1920, screen_height=1080,
    )
    assert action["action_type"] == "batch"
    assert action["actions"][0] == {"action_type": "launch_app", "text": "notepad"}


def test_the_desktop_request_declares_launch_app_not_navigate() -> None:
    client = AnthropicComputerUseHttpClient(model="claude-opus-5-5", effort="low", custom_tools=(LAUNCH_APP_TOOL,))
    payload = client.build_payload(system="s", messages=[{"role": "user", "content": "hi"}])

    toolset, launch = payload["tools"]
    assert toolset["type"] == "computer_toolset_20260801"
    assert launch["name"] == "launch_app"
    assert "command prompt" in launch["input_schema"]["properties"]["app"]["enum"]


def test_the_backend_flag_selects_claude_on_the_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api.runtime.services import build_desktop_services
    from operon.api.runtime_config import validate_modes

    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "claude_computer_use")
    monkeypatch.setenv("OPERON_DESKTOP_CLAUDE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    services = build_desktop_services()

    backend = services.policy_delegate
    assert isinstance(backend, DesktopClaudeComputerUseBackend)
    assert services.perception_service is backend
    assert backend.client.model == "claude-sonnet-5-5"
    assert [tool.get("name") for tool in backend.client.custom_tools] == ["launch_app"]
    assert validate_modes() == []


def test_gemini_stays_the_default_desktop_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api.runtime_config import desktop_mode_config

    monkeypatch.delenv("OPERON_DESKTOP_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    assert desktop_mode_config().backend == "json"


async def test_the_desktop_executor_runs_a_batch_in_order_and_stops_at_a_failure() -> None:
    with patch("operon.executor.desktop._set_dpi_awareness"):
        from operon.executor.desktop import DesktopExecutor

        executor = DesktopExecutor(artifact_dir=Path(".var/test-desktop-batch"))
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
            AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com"),
            AgentAction(action_type=ActionType.PRESS_KEY, key="tab"),
        ],
    )

    with (
        patch("operon.executor.desktop.pyautogui") as gui,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(batch)

    assert result.success is False
    assert "Batch failed on navigate" in result.detail
    gui.press.assert_called_once_with("enter")


async def test_a_click_then_type_run_as_one_type_is_reported_ok(tmp_path: Path) -> None:
    client = ScriptedClient(
        [_reply(_call("left_click", "t1", coordinate=[10, 10]), _call("type", "t2", text="hi")), _reply(_call("screenshot", "t3"))]
    )
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    typed = ExecutedAction(action=AgentAction(action_type=ActionType.TYPE, text="hi", x=15, y=15), success=True, detail="Typed 'hi'")
    await backend.perceive(_frame(tmp_path, "b.png"), _state(typed))

    results = [b for b in client.requests[1][-1]["content"] if b["type"] == "tool_result"]
    assert [r.get("is_error") for r in results] == [None, None]
