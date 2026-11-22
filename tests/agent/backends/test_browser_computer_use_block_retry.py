"""A Gemini prompt block is retried once with older screenshots dropped, then ends the run as model_blocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.backends.base import ModelBlockedError
from operon.agent.backends.browser_computer_use import (
    BLOCK_RETRY_KEEP_SCREENSHOTS,
    BrowserComputerUseBackend,
    _ConversationState,
)
from operon.clients.gemini_computer_use import (
    GeminiComputerUseHttpClient,
    GeminiPromptBlockedError,
)
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.state import AgentState
from operon.models.usage import ModelUsage


def _reply() -> dict:
    call = {"name": "click_at", "args": {"x": 500, "y": 500}}
    return {
        "perception": {"summary": "Tool call: click_at.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": "click",
        "confidence": 0.7,
        "model_content": {"parts": [{"function_call": call}]},
    }


class BlockingClient:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.turns: list[list[dict]] = []
        self._usage: ModelUsage | None = None

    def latest_usage(self) -> ModelUsage | None:
        return self._usage

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        return GeminiComputerUseHttpClient.build_function_response_content(
            function_name=function_name, screenshot_path=screenshot_path, current_url=current_url, error=error
        )

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        self._usage = ModelUsage(
            provider="gemini", model="m", request_kind="computer_use", input_tokens=100 * len(self.turns)
        )
        outcome = self.outcomes[len(self.turns) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _shots(contents: list[dict]) -> int:
    return sum(1 for c in contents for p in c["parts"] if "inline_data" in p)


def _backend(client) -> BrowserComputerUseBackend:
    backend = BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")
    return backend


def _seed(backend: BrowserComputerUseBackend, n: int) -> None:
    conv = backend._conversations.setdefault("run-1", _ConversationState())
    for i in range(n):
        conv.contents.append({"role": "user", "parts": [{"text": f"t{i}"}, {"inline_data": {"data": "x"}}]})
        conv.contents.append({"role": "model", "parts": [{"text": "ok"}]})


def _call(backend, tmp_path: Path):
    shot = tmp_path / "s.png"
    shot.write_bytes(b"png")
    frame = CaptureFrame(artifact_path=str(shot), width=1000, height=1000)
    state = AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING)
    return backend._run_with_retry(prompt="p", screenshot=frame, state=state)


async def test_block_retries_once_with_only_recent_screenshots(tmp_path: Path) -> None:
    client = BlockingClient([GeminiPromptBlockedError("blocked", block_reason="SAFETY"), _reply()])
    backend = _backend(client)
    _seed(backend, 6)

    payload = await _call(backend, tmp_path)

    assert payload["function_calls"]
    assert len(client.turns) == 2
    assert _shots(client.turns[0]) == 7  # six seeded plus the current screen
    assert _shots(client.turns[1]) == BLOCK_RETRY_KEEP_SCREENSHOTS
    # text survives pruning and the stored history keeps the pruned copy
    assert len(client.turns[1]) == len(client.turns[0])
    assert _shots(backend._conversations["run-1"].contents) == BLOCK_RETRY_KEEP_SCREENSHOTS


async def test_second_block_raises_model_blocked_and_keeps_history(tmp_path: Path) -> None:
    client = BlockingClient(
        [GeminiPromptBlockedError("b", block_reason="SAFETY"), GeminiPromptBlockedError("b", block_reason="SAFETY")]
    )
    backend = _backend(client)
    _seed(backend, 6)

    with pytest.raises(ModelBlockedError, match=r"Gemini blocked the request \(SAFETY\)"):
        await _call(backend, tmp_path)

    assert len(client.turns) == 2
    # the failed turn is rolled back: only the six seeded screenshots remain
    assert _shots(backend._conversations["run-1"].contents) == 6


async def test_no_retry_when_there_are_no_older_screenshots_to_drop(tmp_path: Path) -> None:
    client = BlockingClient([GeminiPromptBlockedError("b", block_reason="SAFETY"), _reply()])
    backend = _backend(client)
    _seed(backend, 1)

    with pytest.raises(ModelBlockedError):
        await _call(backend, tmp_path)

    assert len(client.turns) == 1


async def test_retry_keeps_the_conversation_valid_after_a_function_response_turn(tmp_path: Path) -> None:
    from operon.models.execution import ExecutedAction
    from operon.models.policy import ActionType, AgentAction

    shot = tmp_path / "s.png"
    shot.write_bytes(b"png")
    frame = CaptureFrame(artifact_path=str(shot), width=1000, height=1000)
    client = BlockingClient([GeminiPromptBlockedError("b", block_reason="SAFETY"), _reply()])
    backend = _backend(client)
    _seed(backend, 5)
    conv = backend._conversations["run-1"]
    conv.pending_function_names = ["click_at"]
    state = AgentState(
        run_id="run-1",
        intent="x",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(action=AgentAction(action_type=ActionType.CLICK, x=1, y=1), success=True, detail="ok")
        ],
    )

    await backend.perceive(frame, state)

    assert len(client.turns) == 2
    assert _shots(client.turns[1]) == BLOCK_RETRY_KEEP_SCREENSHOTS
    assert conv.acknowledged_actions == 1
    assert conv.pending_function_names == ["click_at"]  # the retried turn's own tool call
    assert conv.contents[-1]["role"] == "model"
    function_response_turn = conv.contents[-2]
    assert any("function_response" in part for part in function_response_turn["parts"])
    # the step's usage counts the blocked request and the retry
    usage = backend.latest_debug_artifacts().usage
    assert usage is not None and usage.input_tokens == 300


class WeakFirstClient:
    """First answer has no function call (needs the stronger retry); the retry gives `second`."""

    model = "model-x"

    def __init__(self, second, *, raise_before_usage: bool = False) -> None:
        self.second = second
        self.raise_before_usage = raise_before_usage
        self.runs = 0
        self._usage: ModelUsage | None = None

    def latest_usage(self) -> ModelUsage | None:
        return self._usage

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        self.runs += 1
        if self.runs > 1 and self.raise_before_usage and isinstance(self.second, Exception):
            raise self.second  # failed before the client recorded this request's usage
        self._usage = ModelUsage(provider="gemini", model="m", request_kind="computer_use", input_tokens=100 * self.runs)
        if self.runs == 1:
            return {"action": {"action_type": "wait"}, "perception": {}, "function_calls": [], "confidence": 0.1}
        if isinstance(self.second, Exception):
            raise self.second
        return self.second


async def _first_turn(backend, client, tmp_path: Path) -> dict:
    async def fake_turn(**_kwargs) -> dict:
        return await client.run_step(prompt="p", screenshot_path="s")

    backend._run_turn = fake_turn  # type: ignore[method-assign]
    return await _call(backend, tmp_path)


async def test_stronger_retry_usage_is_summed_with_the_first_turn(tmp_path: Path) -> None:
    client = WeakFirstClient(_reply())
    backend = _backend(client)

    await _first_turn(backend, client, tmp_path)

    # The first turn (100) is carried; perceive adds the client's latest (the retry, 200).
    assert backend._blocked_usage["run-1"].input_tokens == 100
    assert client.latest_usage().input_tokens == 200


async def test_failed_stronger_retry_sums_each_paid_turn_once(tmp_path: Path) -> None:
    client = WeakFirstClient(GeminiPromptBlockedError("x", block_reason="SAFETY"))
    backend = _backend(client)

    payload = await _first_turn(backend, client, tmp_path)

    assert payload["confidence"] == 0.1  # the first answer is kept
    assert backend._blocked_usage["run-1"].input_tokens == 100
    assert backend.model_label == "model-x"


async def _perceive_first_turn(backend, client, tmp_path: Path):
    """The whole step through perceive, so the usage it reports is what the run log gets."""
    async def fake_turn(**_kwargs) -> dict:
        return await client.run_step(prompt="p", screenshot_path="s")

    backend._run_turn = fake_turn  # type: ignore[method-assign]
    backend._conversations.setdefault("run-1", _ConversationState())
    shot = tmp_path / "s.png"
    shot.write_bytes(b"png")
    frame = CaptureFrame(artifact_path=str(shot), width=1000, height=1000)
    await backend.perceive(frame, AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING))
    return backend.latest_debug_artifacts().usage


async def test_successful_stronger_retry_step_usage_is_both_turns(tmp_path: Path) -> None:
    client = WeakFirstClient(_reply())
    usage = await _perceive_first_turn(_backend(client), client, tmp_path)
    assert usage is not None and usage.input_tokens == 300  # 100 + 200


async def test_retry_failing_before_new_usage_counts_the_first_turn_once(tmp_path: Path) -> None:
    client = WeakFirstClient(GeminiPromptBlockedError("x", block_reason="SAFETY"), raise_before_usage=True)
    backend = _backend(client)
    usage = await _perceive_first_turn(backend, client, tmp_path)
    assert client.runs == 2
    assert usage is not None and usage.input_tokens == 100  # not 200


async def test_failed_retry_with_its_own_usage_step_usage_is_both_turns(tmp_path: Path) -> None:
    client = WeakFirstClient(GeminiPromptBlockedError("x", block_reason="SAFETY"))
    usage = await _perceive_first_turn(_backend(client), client, tmp_path)
    assert usage is not None and usage.input_tokens == 300
