"""OpenAI computer use backend: computer_call actions, call outputs, done, safety checks."""

from __future__ import annotations

import json

import httpx
import pytest

from operon.agent.backends.fallback import BackendCompatibilityError
from operon.agent.backends.model_adapter.backend import INVALID_PREFIX
from operon.agent.backends.model_adapter.openai_native import OpenAIComputerUseBackend
from operon.agent.backends.model_adapter.registry import (
    ModelProfile,
    Pricing,
    load_registry,
)
from operon.agent.backends.model_adapter.transports import (
    ModelTransportError,
    OpenAIResponsesClient,
)
from operon.core.paths import prompts_dir
from operon.models.policy import ActionType

from .conftest import sample, state

PROFILE = ModelProfile(name="openai-computer-use", adapter="openai_native", model="gpt-6.1-sol")


def _response(rid: str, *items: dict) -> dict:
    return {"id": rid, "output": list(items), "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}


def _message(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _computer(call_id: str, *actions: dict, checks: list | None = None) -> dict:
    return {"type": "computer_call", "call_id": call_id, "actions": list(actions), "pending_safety_checks": checks or []}


class ScriptedResponses:
    def __init__(self, replies: list) -> None:
        self._replies = replies
        self.requests: list[dict] = []

    async def create(self, **kwargs) -> dict:
        self.requests.append(kwargs)
        reply = self._replies[min(len(self.requests), len(self._replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def latest_usage(self):
        return None


def _backend(client, environment: str = "browser") -> OpenAIComputerUseBackend:
    return OpenAIComputerUseBackend(
        profile=PROFILE, client=client, environment=environment,
        task_prompt_path=prompts_dir() / "model_adapter_task_prompt.txt",
    )


async def _step(backend, frame, st, name):
    perception = await backend.perceive(frame(name, size=(1920, 1080)), st)
    return await backend.choose_action(st, perception)


async def test_first_turn_sends_task_screenshot_and_tools(frame) -> None:
    client = ScriptedResponses([sample("openai_computer_call")])
    decision = await _step(_backend(client), frame, state(), "s1")

    request = client.requests[0]
    assert request["previous_response_id"] is None and "fully done" in request["instructions"]
    assert request["tools"][0] == {"type": "computer"}
    assert request["tools"][1]["name"] == "navigate"
    content = request["input_items"][0]["content"]
    assert content[0]["text"].startswith("Task: Open the Products page")
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    # click + type merge into one atomic TYPE, then Enter; (320, 40) on the 1280x720 image.
    action = decision.action
    assert action.action_type is ActionType.BATCH
    typed, enter = action.actions
    assert typed.action_type is ActionType.TYPE and (typed.x, typed.y) == (480, 60) and typed.text == "blue mug"
    assert enter.action_type is ActionType.PRESS_KEY and enter.key == "Enter"


async def test_next_turn_answers_the_call_with_the_new_screenshot(frame) -> None:
    client = ScriptedResponses([
        sample("openai_computer_call"),
        _response("resp_002", _message("The results page lists the blue mug.")),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    decision = await _step(backend, frame, state((ActionType.CLICK, True)), "s2")

    request = client.requests[1]
    # Instructions do not carry over through previous_response_id, so every turn sends them.
    assert request["previous_response_id"] == "resp_001" and "fully done" in request["instructions"]
    output = request["input_items"][0]
    assert output["type"] == "computer_call_output" and output["call_id"] == "call_abc123"
    assert output["output"]["type"] == "computer_screenshot"
    assert len(request["input_items"]) == 1  # success needs no extra note
    assert decision.action.action_type is ActionType.STOP
    assert decision.rationale == "The results page lists the blue mug."


async def test_a_failed_action_is_reported_with_the_screenshot(frame) -> None:
    client = ScriptedResponses([sample("openai_computer_call"), _response("resp_002", _message("Retrying."))])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    await _step(backend, frame, state((ActionType.CLICK, False)), "s2")
    note = client.requests[1]["input_items"][1]
    assert note["content"][0]["text"] == "Error: element not found"


async def test_text_before_acting_is_a_nudge_not_a_finish(frame) -> None:
    client = ScriptedResponses([_response("r1", _message("I'll start.")), sample("openai_computer_call")])
    backend = _backend(client)
    first = await _step(backend, frame, state(), "s1")
    assert first.action.action_type is ActionType.WAIT
    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert client.requests[1]["input_items"][0]["content"][0]["text"].startswith("Continue with the task")


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"type": "click", "button": "right", "x": 10, "y": 10}, ActionType.RIGHT_CLICK),
        ({"type": "click", "button": "back", "x": 10, "y": 10}, ActionType.HOTKEY),
        ({"type": "double_click", "x": 10, "y": 10}, ActionType.DOUBLE_CLICK),
        ({"type": "move", "x": 10, "y": 10}, ActionType.HOVER),
        ({"type": "scroll", "x": 10, "y": 10, "scroll_x": 0, "scroll_y": 300}, ActionType.SCROLL),
        ({"type": "drag", "path": [{"x": 1, "y": 1}, {"x": 5, "y": 5}, {"x": 9, "y": 9}]}, ActionType.DRAG),
        ({"type": "drag", "path": [[1, 1], [9, 9]]}, ActionType.DRAG),
        ({"type": "wait"}, ActionType.WAIT),
        ({"type": "keypress", "keys": ["CTRL", "A"]}, ActionType.HOTKEY),
    ],
)
async def test_each_computer_action_maps(frame, action, expected) -> None:
    decision = await _step(_backend(ScriptedResponses([_response("r", _computer("c", action))])), frame, state(), "s1")
    assert decision.action.action_type is expected
    if expected is ActionType.SCROLL:
        assert decision.action.scroll_amount == -300


async def test_legacy_single_action_field_is_accepted(frame) -> None:
    call = {"type": "computer_call", "call_id": "c", "action": {"type": "click", "x": 10, "y": 10}}
    decision = await _step(_backend(ScriptedResponses([_response("r", call)])), frame, state(), "s1")
    assert decision.action.action_type is ActionType.CLICK


async def test_unknown_action_is_invalid_and_reported_next_turn(frame) -> None:
    client = ScriptedResponses([
        _response("r1", _computer("c1", {"type": "teleport"})),
        _response("r2", _message("ok")),
    ])
    backend = _backend(client)
    first = await _step(backend, frame, state(), "s1")
    assert first.rationale.startswith(INVALID_PREFIX) and "teleport" in first.rationale
    assert backend.invalid_reply_counts("run-1") == (1, 1)
    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert "could not be run" in client.requests[1]["input_items"][1]["content"][0]["text"]


async def test_navigate_function_call_is_answered_with_its_outcome(frame) -> None:
    call = {"type": "function_call", "call_id": "f1", "name": "navigate", "arguments": json.dumps({"url": "https://shop.test"})}
    client = ScriptedResponses([_response("r1", call), _response("r2", _message("Loaded."))])
    backend = _backend(client)
    first = await _step(backend, frame, state(), "s1")
    assert first.action.action_type is ActionType.NAVIGATE and first.action.url == "https://shop.test"
    await _step(backend, frame, state((ActionType.CLICK, True)), "s2")
    assert client.requests[1]["input_items"][0] == {"type": "function_call_output", "call_id": "f1", "output": "OK"}


async def test_safety_check_pauses_then_runs_the_held_action_and_acknowledges(frame) -> None:
    checks = [{"id": "sc_1", "code": "malicious_instructions", "message": "Page text looks like an instruction."}]
    client = ScriptedResponses([
        _response("r1", _computer("c1", {"type": "click", "x": 10, "y": 10}, checks=checks)),
        _response("r2", _message("Done.")),
    ])
    backend = _backend(client)
    paused = await _step(backend, frame, state(), "s1")
    assert paused.action.action_type is ActionType.WAIT_FOR_USER and "malicious_instructions" not in paused.action.text
    assert "Page text looks like an instruction." in paused.action.text
    resumed = await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert resumed.action.action_type is ActionType.CLICK
    assert len(client.requests) == 1  # the held action ran without a new request
    await _step(backend, frame, state((ActionType.WAIT, True), (ActionType.CLICK, True)), "s3")
    output = client.requests[1]["input_items"][0]
    assert output["acknowledged_safety_checks"] == checks


async def test_a_correction_drops_the_held_action(frame) -> None:
    checks = [{"id": "sc_1", "code": "malicious_instructions", "message": "Looks like an instruction."}]
    client = ScriptedResponses([
        _response("r1", _computer("c1", {"type": "click", "x": 10, "y": 10}, checks=checks)),
        _response("r2", _message("Done.")),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    backend.add_advisory_hints(["[USER OVERRIDE HINT] use the other button"], source="user_override", run_id="run-1")
    await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert len(client.requests) == 2  # asked the model again instead of running the held click
    output = client.requests[1]["input_items"][0]
    assert "acknowledged_safety_checks" not in output or not output["acknowledged_safety_checks"]
    assert "declined" in json.dumps(client.requests[1]["input_items"])


async def test_an_invalid_reply_holds_nothing_for_approval(frame) -> None:
    checks = [{"id": "sc_1", "code": "malicious_instructions", "message": "Looks like an instruction."}]
    client = ScriptedResponses([
        _response(
            "r1",
            _computer("c1", {"type": "click", "x": 10, "y": 10}, checks=checks),
            _computer("c2", {"type": "teleport"}),
        ),
        _response("r2", _message("Done.")),
    ])
    backend = _backend(client)
    first = await _step(backend, frame, state(), "s1")
    assert first.action.action_type is ActionType.WAIT
    second = await _step(backend, frame, state((ActionType.WAIT, True)), "s2")
    assert second.action.action_type is not ActionType.CLICK and len(client.requests) == 2


def test_prompt_log_drops_responses_screenshots() -> None:
    from operon.agent.backends.model_adapter.backend import _without_images

    items = [
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]},
        {"type": "computer_call_output", "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,BBBB"}},
    ]
    assert "base64" not in json.dumps(_without_images(items))


async def test_transport_error_keeps_the_conversation_for_a_retry(frame) -> None:
    client = ScriptedResponses([
        sample("openai_computer_call"),
        ModelTransportError("HTTP 500"),
        _response("r3", _message("Done.")),
    ])
    backend = _backend(client)
    await _step(backend, frame, state(), "s1")
    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(frame("s2", size=(1920, 1080)), state((ActionType.CLICK, True)))
    await _step(backend, frame, state((ActionType.CLICK, True)), "s3")
    assert client.requests[2]["previous_response_id"] == "resp_001"
    assert client.requests[2]["input_items"][0]["call_id"] == "call_abc123"


async def test_desktop_offers_launch_app() -> None:
    from operon.agent.backends.model_adapter.openai_native import _custom_tools

    assert _custom_tools("desktop")[0]["name"] == "launch_app"


async def test_responses_client_request_and_usage(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=sample("openai_computer_call"))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    profile = PROFILE.model_copy(update={"pricing": Pricing(input_per_mtok=1.0, output_per_mtok=4.0)})
    client = OpenAIResponsesClient(profile, transport=httpx.MockTransport(handler))
    await client.create(input_items=[{"role": "user", "content": []}], tools=[{"type": "computer"}], previous_response_id="resp_0")
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/responses"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-6.1-sol" and body["previous_response_id"] == "resp_0"
    usage = client.latest_usage()
    # OpenAI counts the 80 reasoning tokens inside its 120 output tokens; Operon adds
    # thoughts on top of output, so the visible 40 are reported as output.
    assert usage.provider == "openai" and usage.thoughts_tokens == 80 and usage.output_tokens == 40
    assert usage.estimated_cost_usd == pytest.approx((2100 * 1.0 + 120 * 4.0) / 1e6)


async def test_responses_client_needs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelTransportError, match="OPENAI_API_KEY"):
        await OpenAIResponsesClient(PROFILE).create(input_items=[], tools=[], previous_response_id=None)


def test_registry_has_the_openai_profile_and_services_build_it(monkeypatch, tmp_path) -> None:
    from operon.agent.backends.model_adapter.registry import ModelRegistryError
    from operon.api.runtime.services import build_browser_services
    from operon.api.runtime_config import validate_modes

    assert load_registry()["openai-computer-use"].adapter == "openai_native"
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "openai-computer-use")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert any("OPENAI_API_KEY" in e for e in validate_modes())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # A paid profile without prices would count $0 per step, so the run cost cap could never stop it.
    assert any("needs pricing" in e for e in validate_modes())
    with pytest.raises(ModelRegistryError, match="needs pricing"):
        build_browser_services(executor=None)
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "openai-computer-use:\n  adapter: openai_native\n  model: gpt-6.1-sol\n"
        "  pricing: {input_per_mtok: 1.0, output_per_mtok: 4.0}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPERON_MODEL_REGISTRY", str(registry))
    assert validate_modes() == []
    assert isinstance(build_browser_services(executor=None).policy_delegate, OpenAIComputerUseBackend)
