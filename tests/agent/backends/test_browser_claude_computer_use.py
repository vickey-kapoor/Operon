"""Claude computer use backend: toolset turns, tool results, coordinate scaling and completion."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from operon.agent.actions.translation import normalize_claude_computer_calls
from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.clients.anthropic import AnthropicClientError
from operon.clients.anthropic_computer_use import AnthropicComputerUseHttpClient
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState


def _call(name: str, call_id: str, **tool_input) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "toolset_name": "computer", "input": tool_input}


def _reply(*blocks: dict, stop_reason: str = "tool_use") -> dict:
    return {"content": list(blocks), "stop_reason": stop_reason, "usage": {"input_tokens": 10, "output_tokens": 5}}


class ScriptedClient:
    def __init__(self, replies: list[dict | Exception]) -> None:
        self._replies = replies
        self.requests: list[list[dict]] = []

    async def generate_turn(self, *, system: str, messages: list[dict]) -> dict:
        self.requests.append([dict(m) for m in messages])
        reply = self._replies[min(len(self.requests), len(self._replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def latest_usage(self):
        return None


def _backend(client: ScriptedClient) -> BrowserClaudeComputerUseBackend:
    return BrowserClaudeComputerUseBackend(
        client=client, prompt_path=prompts_dir() / "browser_claude_computer_use_prompt.txt"
    )


def _frame(tmp_path: Path, name: str = "frame.png", size: tuple[int, int] = (1920, 1080)) -> CaptureFrame:
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return CaptureFrame(artifact_path=str(path), width=size[0], height=size[1])


def _state(*executed: tuple[ActionType, bool]) -> AgentState:
    return AgentState(
        run_id="run-1",
        intent="Open the Products page",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(
                action=AgentAction(action_type=a, x=1, y=1) if a is ActionType.CLICK else AgentAction(action_type=a, wait_ms=500),
                success=ok,
                detail="ok" if ok else "element not found",
            )
            for a, ok in executed
        ],
    )


async def test_first_turn_sends_the_task_and_a_downscaled_screenshot(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_call("left_click", "t1", coordinate=[640, 360]))])
    backend = _backend(client)
    state = _state()

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    first = client.requests[0][0]
    assert first["role"] == "user"
    assert "Open the Products page" in first["content"][0]["text"]
    assert first["content"][1]["source"]["media_type"] == "image/jpeg"
    # The 1920x1080 screen is sent at 1280x720, so (640, 360) is the screen centre.
    assert decision.action.action_type is ActionType.CLICK
    assert (decision.action.x, decision.action.y) == (960, 540)


async def test_next_turn_answers_every_call_with_the_toolset_name_and_a_screenshot(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _reply(_call("left_click", "t1", coordinate=[10, 10]), _call("key", "t2", text="Return")),
            _reply({"type": "text", "text": "Done."}, stop_reason="end_turn"),
        ]
    )
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.CLICK, True)))

    answer = client.requests[1][-1]
    assert answer["role"] == "user"
    results = [block for block in answer["content"] if block["type"] == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert all(r["toolset_name"] == "computer" for r in results)
    assert results[0]["content"] == [{"type": "text", "text": "OK"}]
    # The screen after the batch rides on the last result.
    assert results[-1]["content"][-1]["type"] == "image"
    # The assistant turn is passed back unchanged before the answer.
    assert client.requests[1][1]["role"] == "assistant"


async def test_a_failed_batch_reports_the_error_and_skips_the_rest(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _reply(_call("left_click", "t1", coordinate=[10, 10]), _call("key", "t2", text="Return")),
            _reply(_call("screenshot", "t3")),
        ]
    )
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.CLICK, False)))

    content = client.requests[1][-1]["content"]
    first, second = content[0], content[1]
    assert first["is_error"] and "element not found" in first["content"]
    assert second["is_error"] and second["content"].startswith("Not executed")
    assert any(block["type"] == "image" for block in content[2:])


async def test_a_screenshot_call_is_answered_with_the_image(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_call("screenshot", "t1")), _reply(_call("scroll", "t2", scroll_direction="down", scroll_amount=3))])
    backend = _backend(client)
    state = _state()

    perception = await backend.perceive(_frame(tmp_path, "a.png"), state)
    decision = await backend.choose_action(state, perception)
    assert decision.action.action_type is ActionType.WAIT

    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.WAIT, True)))
    result = client.requests[1][-1]["content"][0]
    assert result["content"][0]["type"] == "image"


async def test_text_only_reply_after_acting_is_a_stop(tmp_path: Path) -> None:
    client = ScriptedClient([_reply({"type": "text", "text": "The Products page is showing."}, stop_reason="end_turn")])
    backend = _backend(client)
    state = _state((ActionType.CLICK, True))

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.STOP
    assert perception.summary == "The Products page is showing."


async def test_text_only_reply_after_a_step_reported_failed_is_still_a_stop(tmp_path: Path) -> None:
    """A login whose submit was reported failed but worked used to loop on WAIT to max_steps."""
    client = ScriptedClient([_reply({"type": "text", "text": "Logged in successfully."}, stop_reason="end_turn")])
    backend = _backend(client)
    state = _state((ActionType.CLICK, False))

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.STOP


async def test_text_only_reply_before_acting_waits_and_the_next_turn_nudges_on(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _reply({"type": "text", "text": "Looking at the page."}, stop_reason="end_turn"),
            _reply(_call("left_click", "t1", coordinate=[1, 1])),
        ]
    )
    backend = _backend(client)
    state = _state()

    perception = await backend.perceive(_frame(tmp_path, "a.png"), state)
    assert (await backend.choose_action(state, perception)).action.action_type is ActionType.WAIT

    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.WAIT, True)))
    nudge = client.requests[1][-1]
    assert nudge["role"] == "user" and nudge["content"][0]["type"] == "text"


async def test_navigate_calls_are_answered_without_a_toolset_name(tmp_path: Path) -> None:
    navigate = {"type": "tool_use", "id": "n1", "name": "navigate", "input": {"url": "https://example.com"}}
    client = ScriptedClient([_reply(navigate), _reply({"type": "text", "text": "Done."}, stop_reason="end_turn")])
    backend = _backend(client)
    state = _state()

    perception = await backend.perceive(_frame(tmp_path, "a.png"), state)
    decision = await backend.choose_action(state, perception)
    assert decision.action.action_type is ActionType.NAVIGATE
    assert decision.action.url == "https://example.com"

    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.CLICK, True)))
    result = client.requests[1][-1]["content"][0]
    assert result["tool_use_id"] == "n1" and "toolset_name" not in result


async def test_a_refusal_hands_over_to_the_fallback_and_keeps_the_conversation(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(stop_reason="refusal")])
    backend = _backend(client)

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path), _state())
    assert backend._conversations["run-1"].messages == []


async def test_an_api_error_is_a_compatibility_error(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([AnthropicClientError("Anthropic HTTP error 400: bad")]))

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path), _state())


def _normalize(*calls: tuple[str, dict]) -> dict:
    return normalize_claude_computer_calls(
        [{"name": name, "input": tool_input} for name, tool_input in calls],
        scale_x=1.5,
        scale_y=1.5,
        screen_width=1920,
        screen_height=1080,
    )


def test_a_click_then_type_becomes_one_atomic_type() -> None:
    action = _normalize(("left_click", {"coordinate": [100, 200]}), ("type", {"text": "Ada"}))

    assert action == {
        "action_type": "type", "text": "Ada", "x": 150, "y": 300, "clear_before_typing": False, "press_enter": False,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Return", {"action_type": "press_key", "key": "Enter"}),
        ("Page_Down", {"action_type": "press_key", "key": "PageDown"}),
        ("ctrl+a", {"action_type": "hotkey", "key": "Control+a"}),
        ("alt+Left", {"action_type": "hotkey", "key": "alt+left"}),
    ],
)
def test_key_names_map_to_playwright(text: str, expected: dict) -> None:
    assert _normalize(("key", {"text": text})) == expected


def test_scroll_drag_and_repeat_keys() -> None:
    assert _normalize(("scroll", {"scroll_direction": "down", "scroll_amount": 5})) == {
        "action_type": "scroll", "x": 960, "y": 540, "scroll_amount": -500,
    }
    assert _normalize(("left_click_drag", {"start_coordinate": [10, 10], "coordinate": [100, 10]})) == {
        "action_type": "drag", "x": 15, "y": 15, "x_end": 150, "y_end": 15,
    }
    batch = _normalize(("key", {"text": "Tab", "repeat": 3}))
    assert batch["action_type"] == "batch" and len(batch["actions"]) == 3


def test_coordinates_are_clamped_to_the_screen() -> None:
    assert _normalize(("mouse_move", {"coordinate": [5000, -10]})) == {"action_type": "hover", "x": 1919, "y": 0}


def test_a_triple_click_runs_as_a_triple_click_and_a_following_type_replaces_the_text() -> None:
    assert _normalize(("triple_click", {"coordinate": [100, 200]})) == {"action_type": "triple_click", "x": 150, "y": 300}
    assert _normalize(("triple_click", {"coordinate": [100, 200]}), ("type", {"text": "C:\\Windows"})) == {
        "action_type": "type", "text": "C:\\Windows", "x": 150, "y": 300, "clear_before_typing": True, "press_enter": False,
    }


def test_an_unsupported_member_is_a_compatibility_error() -> None:
    with pytest.raises(BackendCompatibilityError):
        _normalize(("zoom", {"region": [0, 0, 10, 10]}))


def test_the_request_declares_the_toolset_and_navigate_tool() -> None:
    client = AnthropicComputerUseHttpClient(model="claude-opus-5-5", effort="low")
    payload = client.build_payload(system="s", messages=[{"role": "user", "content": "hi"}])

    toolset, navigate = payload["tools"]
    assert toolset["type"] == "computer_toolset_20260801"
    assert "name" not in toolset and "display_width_px" not in toolset
    assert toolset["configs"]["zoom"] == {"enabled": False}
    assert navigate["name"] == "navigate"
    assert payload["output_config"] == {"effort": "low"}
    assert "thinking" not in payload and "tool_choice" not in payload


def test_the_shared_tools_and_system_prompt_get_their_own_cache_breakpoint() -> None:
    client = AnthropicComputerUseHttpClient(model="claude-opus-5-5", effort="low")
    payload = client.build_payload(system="s", messages=[{"role": "user", "content": "hi"}])

    # Render order is tools, system, messages: this breakpoint covers what every run shares.
    assert payload["system"] == [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]
    assert payload["cache_control"] == {"type": "ephemeral"}


def test_the_backend_flag_selects_claude_with_the_gemini_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.agent.backends.fallback import FallbackBackend
    from operon.api.runtime.services import build_browser_services

    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "claude_computer_use")
    monkeypatch.setenv("OPERON_BROWSER_CLAUDE_MODEL", "claude-sonnet-5-5")
    services = build_browser_services(executor=None)

    backend = services.policy_delegate
    assert isinstance(backend, FallbackBackend)
    assert isinstance(backend.primary, BrowserClaudeComputerUseBackend)
    assert backend.primary.client.model == "claude-sonnet-5-5"


def test_claude_is_the_default_browser_backend_when_an_anthropic_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api.runtime_config import browser_mode_config, validate_modes

    monkeypatch.delenv("OPERON_BROWSER_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    assert browser_mode_config().backend == "claude_computer_use"
    assert validate_modes() == []


def test_gemini_stays_the_default_without_an_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api.runtime_config import browser_mode_config, validate_modes

    monkeypatch.delenv("OPERON_BROWSER_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert browser_mode_config().backend == "computer_use"
    assert validate_modes() == []


def test_an_explicit_backend_wins_over_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api.runtime_config import browser_mode_config

    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "computer_use")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    assert browser_mode_config().backend == "computer_use"


async def test_a_call_to_a_tool_that_does_not_exist_is_answered_with_an_error_and_the_rest_runs(tmp_path: Path) -> None:
    # Seen live: a batch ending in a made-up toolset member used to hand the whole run to the fallback.
    client = ScriptedClient(
        [
            _reply(_call("left_click", "t1", coordinate=[10, 10]), _call("navigate_placeholder", "t2")),
            _reply(_call("navigate_placeholder", "t3")),
            _reply({"type": "text", "text": "Done."}, stop_reason="end_turn"),
        ]
    )
    backend = _backend(client)

    state = _state()
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "a.png"), state))
    assert decision.action.action_type is ActionType.CLICK

    state = _state((ActionType.CLICK, True))
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "b.png"), state))
    content = client.requests[1][-1]["content"]
    assert content[0]["tool_use_id"] == "t1" and content[0]["content"][0] == {"type": "text", "text": "OK"}
    assert content[1]["is_error"] and "no navigate_placeholder tool" in content[1]["content"]
    assert content[-1]["type"] == "image"
    # A turn of only unknown calls runs nothing and is not taken as "done".
    assert decision.action.action_type is ActionType.WAIT

    await backend.perceive(_frame(tmp_path, "c.png"), _state((ActionType.CLICK, True), (ActionType.WAIT, True)))
    content = client.requests[2][-1]["content"]
    assert content[0]["tool_use_id"] == "t3" and content[0]["is_error"]
    assert any(block["type"] == "image" for block in content[1:])


def _zoom_client(replies: list[dict | Exception]) -> ScriptedClient:
    client = ScriptedClient(replies)
    client.zoom = True
    return client


async def test_a_zoom_is_answered_with_a_full_resolution_crop_in_the_same_step(tmp_path: Path) -> None:
    # Seen on ScreenSpot-Pro: on dense 4K screens Claude asks to zoom and, with zoom off, stops.
    client = _zoom_client(
        [
            _reply(_call("zoom", "z1", region=[0, 0, 640, 360])),
            _reply(_call("left_click", "t1", coordinate=[100, 50])),
        ]
    )
    backend = _backend(client)
    state = _state()

    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path), state))

    assert len(client.requests) == 2
    result = client.requests[1][-1]["content"][0]
    assert result["tool_use_id"] == "z1" and result["toolset_name"] == "computer"
    image = result["content"][0]
    assert image["type"] == "image"
    import base64
    import io

    with Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))) as crop:
        # The top-left quarter of the 1920x1080 capture (960x540), enlarged to 1280 wide.
        assert crop.size == (1280, 720)
    # Coordinates stay in the full screenshot's space after a zoom.
    assert decision.action.action_type is ActionType.CLICK
    assert (decision.action.x, decision.action.y) == (150, 75)


async def test_zooms_stop_after_three_in_a_step_and_the_next_turn_says_why(tmp_path: Path) -> None:
    client = _zoom_client([_reply(_call("zoom", f"z{i}", region=[0, 0, 100, 100])) for i in range(5)])
    backend = _backend(client)

    state = _state()
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "a.png"), state))
    assert len(client.requests) == 4
    assert decision.action.action_type is ActionType.WAIT

    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.WAIT, True)))
    result = client.requests[4][-1]["content"][0]
    assert result["is_error"] and "Call zoom on its own" in result["content"]


async def test_a_zoom_with_a_bad_region_is_an_error_result(tmp_path: Path) -> None:
    client = _zoom_client(
        [_reply(_call("zoom", "z1", region=[5, 5, 5, 5])), _reply(_call("left_click", "t1", coordinate=[1, 1]))]
    )
    backend = _backend(client)
    state = _state()

    await backend.choose_action(state, await backend.perceive(_frame(tmp_path), state))

    result = client.requests[1][-1]["content"][0]
    assert result["is_error"] and "region" in result["content"]


async def test_without_the_flag_a_zoom_is_not_run(tmp_path: Path) -> None:
    client = ScriptedClient([_reply(_call("zoom", "z1", region=[0, 0, 10, 10])), _reply(_call("wait", "w1"))])
    backend = _backend(client)

    state = _state()
    decision = await backend.choose_action(state, await backend.perceive(_frame(tmp_path, "a.png"), state))
    assert len(client.requests) == 1 and decision.action.action_type is ActionType.WAIT
    await backend.perceive(_frame(tmp_path, "b.png"), _state((ActionType.WAIT, True)))
    assert "no zoom tool" in client.requests[1][-1]["content"][0]["content"]


def test_the_zoom_flag_turns_the_zoom_member_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_CLAUDE_ZOOM", raising=False)
    assert AnthropicComputerUseHttpClient().zoom is False
    monkeypatch.setenv("OPERON_CLAUDE_ZOOM", "true")
    client = AnthropicComputerUseHttpClient()
    toolset = client.build_payload(system="s", messages=[])["tools"][0]
    assert client.zoom is True
    assert "zoom" not in toolset["configs"] and toolset["configs"]["hold_key"] == {"enabled": False}
