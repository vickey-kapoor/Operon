"""A failed Gemini model call rolls the conversation back and keeps queued hints (KAN-173)."""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
from pathlib import Path

import pytest

from operon.agent.actions.translation import KeyHold
from operon.agent.backends.base import BackendCompatibilityError
from operon.agent.backends.browser_computer_use import (
    BrowserComputerUseBackend,
    _ConversationState,
)
from operon.agent.backends.desktop_computer_use import DesktopComputerUseBackend
from operon.clients.gemini_computer_use import (
    GeminiComputerUseError,
    GeminiComputerUseHttpClient,
    GeminiPromptBlockedError,
)
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.state import AgentState

CALL = {"name": "click_at", "args": {"x": 500, "y": 500}}


def _reply() -> dict:
    return {
        "perception": {"summary": "Tool call: click_at.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": CALL,
        "function_calls": [CALL],
        "rationale": "click",
        "confidence": 0.7,
        "model_content": {"parts": [{"function_call": CALL}]},
    }


class FlakyClient:
    """Plays its outcomes in order; a failed outcome is an exception instance."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.turns: list[list[dict]] = []

    def latest_usage(self):
        return None

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        return GeminiComputerUseHttpClient.build_function_response_content(
            function_name=function_name, screenshot_path=screenshot_path, current_url=current_url, error=error
        )

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        outcome = self.outcomes[len(self.turns) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _backend(client, cls=BrowserComputerUseBackend):
    name = "browser" if cls is BrowserComputerUseBackend else "desktop"
    return cls(client=client, prompt_path=prompts_dir() / f"{name}_computer_use_prompt.txt")


def _frame(tmp_path: Path, name: str) -> CaptureFrame:
    shot = tmp_path / name
    shot.write_bytes(name.encode())
    return CaptureFrame(artifact_path=str(shot), width=1000, height=1000)


def _state() -> AgentState:
    return AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING)


def _seed(backend, **overrides) -> _ConversationState:
    conv = backend._conversations.setdefault("run-1", _ConversationState())
    conv.contents = [
        {"role": "user", "parts": [{"text": "task"}, {"inline_data": {"data": "old"}}]},
        {"role": "model", "parts": [{"function_call": CALL}]},
    ]
    conv.pending_function_names = ["click_at"]
    conv.last_calls = [CALL]
    conv.acknowledge_safety = True
    conv.pending_hold = KeyHold(keys=["Control"])
    conv.key_hold = KeyHold(keys=["Shift"])
    conv.declined_reason = None
    conv.not_run_reason = "paused by a gate"
    for key, value in overrides.items():
        setattr(conv, key, value)
    return conv


def _same(a: _ConversationState, b: _ConversationState) -> None:
    for f in dataclasses.fields(a):
        assert getattr(a, f.name) == getattr(b, f.name), f.name


@pytest.mark.parametrize("reason", [{"not_run_reason": "paused by a gate"}, {"not_run_reason": None, "declined_reason": "no"}])
async def test_failed_turn_restores_the_conversation_and_the_retry_sends_the_new_screen(
    tmp_path: Path, reason: dict
) -> None:
    client = FlakyClient([GeminiComputerUseError("HTTP error 503: unavailable"), _reply()])
    backend = _backend(client)
    conv = _seed(backend, **reason)
    before = copy.deepcopy(conv)
    backend.add_advisory_hints(["look left"], run_id="run-1")
    client_before = backend.client

    with pytest.raises(BackendCompatibilityError, match="503"):
        await backend.perceive(_frame(tmp_path, "failed.png"), _state())

    assert backend._conversations["run-1"] is conv  # restored in place
    _same(conv, before)
    assert len(conv.contents) == 2 and conv.contents[-1]["role"] == "model"
    assert conv.pending_function_names == ["click_at"]
    assert conv.acknowledge_safety is True and conv.acknowledged_actions == 0
    assert conv.pending_hold == KeyHold(keys=["Control"])
    assert backend._advisory_hints["run-1"] == [("look left", "")]
    assert backend.client is client_before  # no backend swap

    await backend.perceive(_frame(tmp_path, "retry.png"), _state())

    sent = client.turns[1]
    assert len(sent) == 3  # exactly one function response added to the seeded two
    last = sent[-1]
    assert last["role"] == "user"
    answer = next(p for p in last["parts"] if "function_response" in p)["function_response"]
    assert answer["response"]["safety_acknowledgement"] == "true"
    images = [p["inline_data"]["data"] for p in last["parts"] if "inline_data" in p]
    assert [base64.b64decode(i) for i in images] == [b"retry.png"]
    assert str(last).count("look left") == 1
    assert not backend._advisory_hints.get("run-1")
    assert backend.client is client_before


async def test_first_turn_failure_leaves_contents_empty(tmp_path: Path) -> None:
    client = FlakyClient([GeminiComputerUseError("HTTP error 503")])
    backend = _backend(client)

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path, "a.png"), _state())

    assert backend._conversations["run-1"].contents == []


async def test_block_then_503_on_the_pruned_retry_leaves_contents_unchanged(tmp_path: Path) -> None:
    client = FlakyClient([GeminiPromptBlockedError("b", block_reason="SAFETY"), GeminiComputerUseError("HTTP error 503")])
    backend = _backend(client)
    conv = backend._conversations.setdefault("run-1", _ConversationState())
    for i in range(6):
        conv.contents.append({"role": "user", "parts": [{"text": f"t{i}"}, {"inline_data": {"data": "x"}}]})
        conv.contents.append({"role": "model", "parts": [{"text": "ok"}]})
    before = copy.deepcopy(conv)

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path, "a.png"), _state())

    assert len(client.turns) == 2
    _same(conv, before)
    assert "run-1" not in backend._blocked_usage


async def test_desktop_backend_inherits_the_rollback(tmp_path: Path) -> None:
    client = FlakyClient([GeminiComputerUseError("HTTP error 503"), _reply()])
    backend = _backend(client, DesktopComputerUseBackend)
    conv = _seed(backend)
    before = copy.deepcopy(conv)
    backend.add_advisory_hints(["note"], run_id="run-1")

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path, "a.png"), _state())

    _same(conv, before)
    assert backend._advisory_hints["run-1"] == [("note", "")]
    assert backend.client is client


async def test_cancellation_restores_too(tmp_path: Path) -> None:
    client = FlakyClient([asyncio.CancelledError()])
    backend = _backend(client)
    conv = _seed(backend)
    before = copy.deepcopy(conv)
    backend.add_advisory_hints(["note"], run_id="run-1")

    with pytest.raises(asyncio.CancelledError):
        await backend.perceive(_frame(tmp_path, "a.png"), _state())

    _same(conv, before)
    assert backend._advisory_hints["run-1"] == [("note", "")]


async def test_hint_from_the_shared_bucket_goes_back_there_and_is_sent_once(tmp_path: Path) -> None:
    client = FlakyClient([GeminiComputerUseError("HTTP error 503"), _reply(), _reply()])
    backend = _backend(client)
    backend.add_advisory_hints(["shared note"])
    assert list(backend._advisory_hints) == [""]

    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path, "a.png"), _state())
    assert backend._advisory_hints[""] == [("shared note", "")]

    await backend.perceive(_frame(tmp_path, "b.png"), _state())
    await backend.perceive(_frame(tmp_path, "c.png"), _state())
    assert str(client.turns[1][0]).count("shared note") == 1  # the first prompt carries it
    assert str(client.turns[2][-1]).count("shared note") == 0
