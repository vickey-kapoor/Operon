"""ModelAdapterBackend: prompt and history, parsing, invalid replies, done and fail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operon.agent.backends.fallback import BackendCompatibilityError
from operon.agent.backends.model_adapter.backend import (
    GAVE_UP_PREFIX,
    INVALID_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports import ModelTransportError
from operon.core.paths import prompts_dir
from operon.models.policy import ActionType
from operon.store.background_writer import bg_writer

from .conftest import ScriptedChat, chat_reply, sample, state, tool_call

QWEN = ModelProfile(
    name="qwen3-vl-32b",
    adapter="openai_compat",
    model="Qwen/Qwen3-VL-32B-Instruct",
    coords="relative_1000",
    history_screens=2,
)


def _backend(client: ScriptedChat, profile: ModelProfile = QWEN, environment: str = "browser") -> ModelAdapterBackend:
    return ModelAdapterBackend(
        profile=profile,
        client=client,
        environment=environment,
        task_prompt_path=prompts_dir() / "model_adapter_task_prompt.txt",
    )


async def _step(backend: ModelAdapterBackend, frame, st, name: str):
    perception = await backend.perceive(frame(name), st)
    return await backend.choose_action(st, perception)


async def test_first_turn_sends_task_tools_and_screenshot(frame) -> None:
    client = ScriptedChat([sample("qwen3_vl_click")])
    decision = await _step(_backend(client), frame, state(), "s1")

    request = client.requests[0]
    system, task, screen = request["messages"]
    assert system["role"] == "system" and "relative, from 0 to 1000" in system["content"]
    assert "Task: Open the Products page" in task["content"] and "navigate" in task["content"]
    assert screen["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    names = {tool["function"]["name"] for tool in request["tools"]}
    assert {"click", "navigate", "done", "fail"} <= names and "launch_app" not in names
    # (500, 100) on the 0-1000 grid of a 1280x800 screen.
    assert decision.action.action_type is ActionType.CLICK
    assert (decision.action.x, decision.action.y) == (640, 80)
    assert decision.rationale == "The Products link is in the top navigation bar."


async def test_desktop_offers_launch_app_not_navigate(frame) -> None:
    client = ScriptedChat([chat_reply(tool_calls=[tool_call("launch_app", app="notepad")])])
    decision = await _step(_backend(client, environment="desktop"), frame, state(), "s1")
    names = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
    assert "launch_app" in names and "navigate" not in names
    assert decision.action.action_type is ActionType.LAUNCH_APP and decision.action.text == "notepad"


async def test_hermes_tool_call_in_text_is_parsed(frame) -> None:
    client = ScriptedChat([sample("qwen3_vl_hermes_text")])
    decision = await _step(_backend(client), frame, state(), "s1")
    assert decision.action.action_type is ActionType.TYPE
    assert decision.action.text == "blue mug" and decision.action.press_enter is True
    assert (decision.action.x, decision.action.y) == (320, 40)


async def test_history_answers_calls_and_drops_old_screenshots(frame) -> None:
    client = ScriptedChat([
        chat_reply(tool_calls=[tool_call("click", "c1", x=10, y=10)]),
        chat_reply(tool_calls=[tool_call("click", "c2", x=20, y=20)]),
        chat_reply(tool_calls=[tool_call("click", "c3", x=30, y=30)]),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    await _step(backend, frame, state((ActionType.CLICK, True)), "s2")
    await _step(backend, frame, state((ActionType.CLICK, True), (ActionType.CLICK, False)), "s3")

    messages = client.requests[2]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "user", "assistant", "tool", "user", "assistant", "tool", "user"]
    first_screen, second_screen, current = messages[2], messages[5], messages[8]
    # history_screens=2: the current screenshot plus one earlier one.
    assert first_screen["content"][0] == {"type": "text", "text": "(earlier screenshot omitted)"}
    assert second_screen["content"][0]["type"] == "image_url"
    assert current["content"][0]["type"] == "image_url"
    assert messages[4] == {"role": "tool", "tool_call_id": "c1", "content": "OK"}
    assert messages[7] == {"role": "tool", "tool_call_id": "c2", "content": "Error: element not found"}


async def test_reply_without_a_tool_call_is_invalid_and_explained_next_turn(frame) -> None:
    client = ScriptedChat([
        chat_reply("I think the page is loaded."),
        chat_reply(tool_calls=[tool_call("click", "c2", x=1, y=1)]),
    ])
    backend = _backend(client)
    first = await _step(backend, frame, state(), "s1")
    assert first.action.action_type is ActionType.WAIT
    assert first.rationale == f"{INVALID_PREFIX}the reply has no tool call"
    assert backend.invalid_reply_counts("run-1") == (1, 1)

    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    replayed, current = client.requests[1]["messages"][3], client.requests[1]["messages"][4]
    assert replayed == {"role": "assistant", "content": "I think the page is loaded."}
    assert "could not be used (the reply has no tool call)" in current["content"][-1]["text"]
    assert backend.invalid_reply_counts("run-1") == (1, 2)


@pytest.mark.parametrize(
    ("calls", "reason"),
    [
        ([tool_call("teleport", x=1)], "there is no 'teleport' tool"),
        ([{"id": "c", "type": "function", "function": {"name": "click", "arguments": "{x: 1"}}], "not valid JSON"),
        ([tool_call("click", "a", x=1, y=1), tool_call("done", "b")], "done must be the only tool call"),
        ([tool_call("click", x=5)], "needs x and y"),
    ],
)
async def test_unusable_tool_calls_become_a_recorded_wait(frame, calls, reason) -> None:
    decision = await _step(_backend(ScriptedChat([chat_reply(tool_calls=calls)])), frame, state(), "s1")
    assert decision.action.action_type is ActionType.WAIT
    assert decision.rationale.startswith(INVALID_PREFIX) and reason in decision.rationale


async def test_done_is_a_stop_carrying_the_answer(frame) -> None:
    decision = await _step(_backend(ScriptedChat([sample("qwen3_vl_done")])), frame, state((ActionType.CLICK, True)), "s1")
    assert decision.action.action_type is ActionType.STOP
    assert decision.action.text == "R-1042"
    assert decision.rationale == "The order page shows the refund was issued."


async def test_fail_is_a_stop_marked_as_given_up(frame) -> None:
    reply = chat_reply(tool_calls=[tool_call("fail", reason="The site shows a CAPTCHA.")])
    decision = await _step(_backend(ScriptedChat([reply])), frame, state(), "s1")
    assert decision.action.action_type is ActionType.STOP
    assert decision.rationale == f"{GAVE_UP_PREFIX}The site shows a CAPTCHA."


async def test_transport_errors_surface_as_backend_errors(frame) -> None:
    backend = _backend(ScriptedChat([ModelTransportError("HTTP 503")]))
    with pytest.raises(BackendCompatibilityError, match="HTTP 503"):
        await backend.perceive(frame("s1"), state())


async def test_gate_pause_is_reported_as_not_executed(frame) -> None:
    client = ScriptedChat([
        chat_reply(tool_calls=[tool_call("click", "c1", x=1, y=1)]),
        chat_reply(tool_calls=[tool_call("click", "c2", x=1, y=1)]),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    backend.note_not_executed("run-1", "trust gate: purchase")
    await _step(backend, frame, state(), "s2")
    tool_message = client.requests[1]["messages"][4]
    assert tool_message["content"].startswith("Not executed: the run paused for a human")


async def test_hints_reach_the_next_turn_once(frame) -> None:
    client = ScriptedChat([chat_reply(tool_calls=[tool_call("wait", seconds=1)])])
    backend = _backend(client)
    backend.add_advisory_hints(["The cart is in the top right."], run_id="run-1")
    await _step(backend, frame, state(), "s1")
    assert "The cart is in the top right." in client.requests[0]["messages"][-1]["content"][-1]["text"]
    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert "The cart" not in str(client.requests[1]["messages"][-1])


async def test_parsed_artifact_records_invalid_replies(frame) -> None:
    backend = _backend(ScriptedChat([chat_reply("no tools here")]))
    shot = frame("s1")
    await backend.perceive(shot, state())
    await bg_writer.flush()
    debug = backend.latest_debug_artifacts()
    parsed = json.loads(Path(debug.parsed_artifact_path).read_text(encoding="utf-8"))
    assert parsed["invalid"] is True and parsed["invalid_reason"] == "the reply has no tool call"
    assert parsed["profile"] == "qwen3-vl-32b" and parsed["invalid_replies"] == 1


async def test_ui_tars_history_is_text_with_the_outcome_carried_forward(frame) -> None:
    profile = ModelProfile(name="ui-tars", adapter="openai_compat", parser="ui_tars", model="ui-tars", history_screens=5)
    client = ScriptedChat([
        chat_reply("Thought: Click search.\nAction: click(start_box='(100,50)')"),
        chat_reply("Thought: Done.\nAction: finished(content='ok')"),
    ])
    backend = _backend(client, profile=profile)
    first = await _step(backend, frame, state(), "s1")
    assert (first.action.x, first.action.y) == (100, 50)  # pixels of the unscaled 1280x800 frame
    second = await _step(backend, frame, state((ActionType.CLICK, False)), "s2")
    assert second.action.action_type is ActionType.STOP

    messages = client.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "user", "assistant", "user"]
    assert "## User Instruction\nTask: Open the Products page" in messages[0]["content"]
    assert messages[2]["content"].startswith("Thought: Click search.")
    assert messages[3]["content"][-1]["text"] == "Result of your last action: Error: element not found"


async def test_screenshots_outside_the_history_window_are_freed(frame) -> None:
    client = ScriptedChat([chat_reply(tool_calls=[tool_call("click", f"c{i}", x=10, y=10)]) for i in range(4)])
    backend = _backend(client)
    history = [(ActionType.CLICK, True)]
    for i in range(4):
        await _step(backend, frame, state(*history * i), f"s{i}")
    turns = backend._runs["run-1"].turns
    # history_screens=2: only the last two turns keep their image.
    assert [turn.image is not None for turn in turns] == [False, False, True, True]


async def test_an_action_the_full_harness_replaced_is_reported_as_not_run(frame) -> None:
    client = ScriptedChat([
        chat_reply(tool_calls=[tool_call("click", "c1", x=1, y=1)]),
        chat_reply(tool_calls=[tool_call("click", "c2", x=1, y=1)]),
        chat_reply(tool_calls=[tool_call("click", "c3", x=1, y=1)]),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    backend.note_replaced("run-1", "wait, chosen by rule dismiss_overlay")
    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    tool_message = client.requests[1]["messages"][4]
    assert tool_message["content"].startswith("Not executed: Operon's harness ran a different action instead")
    assert "rule dismiss_overlay" in tool_message["content"] and "It succeeded." in tool_message["content"]
    # The next turn's own action ran: it is reported as usual.
    await _step(backend, frame, state((ActionType.WAIT, True), (ActionType.CLICK, True)), "s3")
    assert "Not executed" not in str(client.requests[2]["messages"][-2])
