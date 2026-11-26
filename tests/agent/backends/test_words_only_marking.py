"""KAN-204: a backend marks a reply that had words only and no action (``PolicyDecision.words_only``).

Only the marking is tested here. The marked reply is still the same wait it always was."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from operon.agent.actions.translation import build_policy_decision
from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.browser_computer_use import (
    GIVE_UP,
    BrowserComputerUseBackend,
)
from operon.agent.backends.model_adapter.backend import (
    INVALID_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.formats import PromptContext, get_format
from operon.agent.backends.model_adapter.openai_native import OpenAIComputerUseBackend
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState

from .model_adapter.conftest import ScriptedChat, chat_reply, tool_call


def _state(*actions: ActionType) -> AgentState:
    return AgentState(
        run_id="run-1",
        intent="Open the Products page",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(
                action=AgentAction(action_type=a, x=1, y=1) if a is ActionType.CLICK else AgentAction(action_type=a, wait_ms=500),
                success=True,
                detail="ok",
            )
            for a in actions
        ],
    )


def _frame(tmp_path: Path, name: str = "step1") -> CaptureFrame:
    step = tmp_path / name
    step.mkdir(parents=True, exist_ok=True)
    path = step / "frame.png"
    Image.new("RGB", (1280, 800), "white").save(path)
    return CaptureFrame(artifact_path=str(path), width=1280, height=800)


async def _decide(backend, tmp_path: Path, state: AgentState) -> PolicyDecision:
    perception = await backend.perceive(_frame(tmp_path), state)
    return await backend.choose_action(state, perception)


# --- build_policy_decision ------------------------------------------------------------------


def test_build_policy_decision_passes_the_field_through() -> None:
    payload = {"action": {"action_type": "wait", "wait_ms": 500}, "rationale": "r", "confidence": 0.4, "active_subgoal": "s"}
    assert build_policy_decision(payload).words_only is None
    assert build_policy_decision({**payload, "words_only": "Which one?"}).words_only == "Which one?"
    assert build_policy_decision({**payload, "words_only": ""}).words_only == ""


# --- Gemini ---------------------------------------------------------------------------------


class GeminiClient:
    def __init__(self, reply: dict) -> None:
        self.reply = reply

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        return self.reply

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return self.reply

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        return {"role": "user", "parts": []}


def _gemini_text(*texts: str, finish: str | None = "STOP") -> dict:
    summary = " ".join(texts).strip() or "Computer Use step evaluated."
    raw: dict = {"text_parts": list(texts), "had_function_call": False}
    if finish is not None:
        raw["finish_reason"] = finish
    return {
        "perception": {"summary": summary, "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "action": {"action_type": "wait", "wait_ms": 1000},
        "rationale": summary,
        "confidence": 0.4,
        "active_subgoal": "gather more browser context",
        "model_content": {"parts": [{"text": t} for t in texts]},
        "raw_candidate": raw,
    }


def _gemini_call(name: str = "click_at", **args) -> dict:
    call = {"name": name, "args": args or {"x": 500, "y": 500}}
    return {
        "perception": {"summary": "Tool call.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": "do it",
        "confidence": 0.7,
        "active_subgoal": name,
        "model_content": {"parts": [{"function_call": call}]},
        "raw_candidate": {"finish_reason": "STOP", "had_function_call": True},
    }


def _gemini(reply: dict) -> BrowserComputerUseBackend:
    return BrowserComputerUseBackend(client=GeminiClient(reply), prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")


async def test_gemini_text_only_reply_before_acting_is_marked(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_text("Which", "account?")), tmp_path, _state())
    assert decision.action.action_type is ActionType.WAIT
    assert decision.words_only == "Which account?"


async def test_gemini_reply_with_no_text_is_marked_with_empty_words(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_text()), tmp_path, _state())
    assert decision.words_only == ""


async def test_gemini_text_only_reply_after_waiting_only_is_marked(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_text("I cannot do that.")), tmp_path, _state(ActionType.WAIT))
    assert decision.words_only == "I cannot do that."


async def test_gemini_long_words_are_capped(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_text("x" * 5000)), tmp_path, _state())
    assert decision.words_only is not None and len(decision.words_only) == 2000


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "MALFORMED_FUNCTION_CALL", "SAFETY"])
async def test_gemini_cut_off_reply_is_not_marked(tmp_path: Path, finish: str) -> None:
    decision = await _decide(_gemini(_gemini_text("Let me", finish=finish)), tmp_path, _state())
    assert decision.words_only is None


async def test_gemini_text_after_acting_is_done_not_marked(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_text("All done.")), tmp_path, _state(ActionType.CLICK))
    assert decision.action.action_type is ActionType.STOP
    assert decision.words_only is None


async def test_gemini_action_and_give_up_are_not_marked(tmp_path: Path) -> None:
    click = await _decide(_gemini(_gemini_call()), tmp_path, _state())
    assert click.action.action_type is ActionType.CLICK and click.words_only is None
    give_up = await _decide(_gemini(_gemini_call(GIVE_UP, reason="blocked")), tmp_path, _state())
    assert give_up.action.action_type is ActionType.STOP and give_up.words_only is None


async def test_gemini_model_chosen_wait_is_not_marked(tmp_path: Path) -> None:
    decision = await _decide(_gemini(_gemini_call("wait_5_seconds")), tmp_path, _state())
    assert decision.action.action_type is ActionType.WAIT
    assert decision.words_only is None


async def test_gemini_step_with_no_model_turn_is_not_marked(tmp_path: Path) -> None:
    backend = _gemini(_gemini_text("Which account?"))
    state = _state()
    perception = await backend.perceive(_frame(tmp_path, "a"), state)
    await backend.choose_action(state, perception)
    # The person approved calls that were held: this step runs them with no model turn.
    conversation = backend._conversations[state.run_id]
    conversation.held_calls = [{"name": "click_at", "args": {"x": 5, "y": 5}}]
    decision = await _decide(backend, tmp_path, state)
    assert decision.action.action_type is ActionType.CLICK and decision.words_only is None


# --- Claude computer use --------------------------------------------------------------------


class ClaudeClient:
    def __init__(self, reply: dict) -> None:
        self.reply = reply

    async def generate_turn(self, *, system: str, messages: list[dict]) -> dict:
        return self.reply

    def latest_usage(self):
        return None


def _claude(*blocks: dict, stop_reason: str = "end_turn") -> BrowserClaudeComputerUseBackend:
    reply = {"content": list(blocks), "stop_reason": stop_reason, "usage": {"input_tokens": 1, "output_tokens": 1}}
    return BrowserClaudeComputerUseBackend(
        client=ClaudeClient(reply), prompt_path=prompts_dir() / "browser_claude_computer_use_prompt.txt"
    )


_TEXT = {"type": "text", "text": "Which account should I use?"}
_CLICK = {"type": "tool_use", "id": "t1", "name": "left_click", "toolset_name": "computer", "input": {"coordinate": [10, 10]}}
_INVENTED = {"type": "tool_use", "id": "t2", "name": "teleport", "toolset_name": "computer", "input": {}}


async def test_claude_text_only_reply_before_acting_is_marked(tmp_path: Path) -> None:
    decision = await _decide(_claude(_TEXT), tmp_path, _state())
    assert decision.action.action_type is ActionType.WAIT
    assert decision.words_only == "Which account should I use?"


async def test_claude_reply_with_no_text_is_marked_with_empty_words(tmp_path: Path) -> None:
    assert (await _decide(_claude(), tmp_path, _state())).words_only == ""


async def test_claude_cut_off_reply_is_not_marked(tmp_path: Path) -> None:
    assert (await _decide(_claude(_TEXT, stop_reason="max_tokens"), tmp_path, _state())).words_only is None


async def test_claude_text_after_acting_is_done_not_marked(tmp_path: Path) -> None:
    decision = await _decide(_claude(_TEXT), tmp_path, _state(ActionType.CLICK))
    assert decision.action.action_type is ActionType.STOP and decision.words_only is None


async def test_claude_action_and_invented_tool_are_not_marked(tmp_path: Path) -> None:
    acted = await _decide(_claude(_TEXT, _CLICK, stop_reason="tool_use"), tmp_path, _state())
    assert acted.action.action_type is ActionType.CLICK and acted.words_only is None
    invented = await _decide(_claude(_INVENTED, stop_reason="tool_use"), tmp_path, _state())
    assert invented.words_only is None


# --- OpenAI native --------------------------------------------------------------------------


class ResponsesClient:
    def __init__(self, reply: dict) -> None:
        self.reply = reply

    async def create(self, **kwargs) -> dict:
        return self.reply

    def latest_usage(self):
        return None


def _openai(*items: dict, status: str | None = "completed") -> OpenAIComputerUseBackend:
    reply: dict = {"id": "resp_1", "output": list(items), "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    if status is not None:
        reply["status"] = status
    return OpenAIComputerUseBackend(
        profile=ModelProfile(name="openai-computer-use", adapter="openai_native", model="gpt-6.1-sol"),
        client=ResponsesClient(reply),
        environment="browser",
        task_prompt_path=prompts_dir() / "model_adapter_task_prompt.txt",
    )


_MESSAGE = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Which account?"}]}
_COMPUTER = {
    "type": "computer_call", "call_id": "c1", "pending_safety_checks": [],
    "actions": [{"type": "click", "x": 10, "y": 10, "button": "left"}],
}


async def test_openai_text_only_reply_before_acting_is_marked(tmp_path: Path) -> None:
    decision = await _decide(_openai(_MESSAGE), tmp_path, _state())
    assert decision.action.action_type is ActionType.WAIT
    assert decision.words_only == "Which account?"
    assert decision.rationale == "Which account?"


async def test_openai_status_missing_still_counts_and_no_text_is_empty(tmp_path: Path) -> None:
    assert (await _decide(_openai(_MESSAGE, status=None), tmp_path, _state())).words_only == "Which account?"
    assert (await _decide(_openai(), tmp_path, _state())).words_only == ""


async def test_openai_cut_off_reply_is_not_marked(tmp_path: Path) -> None:
    assert (await _decide(_openai(_MESSAGE, status="incomplete"), tmp_path, _state())).words_only is None


async def test_openai_text_after_acting_and_action_are_not_marked(tmp_path: Path) -> None:
    done = await _decide(_openai(_MESSAGE), tmp_path, _state(ActionType.CLICK))
    assert done.action.action_type is ActionType.STOP and done.words_only is None
    acted = await _decide(_openai(_COMPUTER), tmp_path, _state())
    assert acted.action.action_type is ActionType.CLICK and acted.words_only is None


# --- model adapter, all three formats -------------------------------------------------------


def _adapter(parser: str, reply: dict) -> ModelAdapterBackend:
    profile = ModelProfile(name=f"p-{parser}", adapter="openai_compat", parser=parser, model="m", coords="relative_1000")
    return ModelAdapterBackend(
        profile=profile,
        client=ScriptedChat([reply]),
        environment="browser",
        task_prompt_path=prompts_dir() / "model_adapter_task_prompt.txt",
    )


_WORDS = {
    "neutral_tools": "I am not sure which account to use.",
    "ui_tars": "Thought: I am not sure which account to use.",
    "opencua": "I am not sure which account to use.",
}


@pytest.mark.parametrize("parser", ["neutral_tools", "ui_tars", "opencua"])
async def test_adapter_word_only_reply_is_marked_and_the_rest_is_unchanged(tmp_path: Path, parser: str) -> None:
    decision = await _decide(_adapter(parser, chat_reply(content=_WORDS[parser])), tmp_path, _state())
    assert decision.action.action_type is ActionType.WAIT
    assert decision.rationale.startswith(INVALID_PREFIX)
    assert decision.words_only is not None and "not sure which account" in decision.words_only


@pytest.mark.parametrize("parser", ["neutral_tools", "ui_tars", "opencua"])
async def test_adapter_cut_off_word_only_reply_is_not_marked(tmp_path: Path, parser: str) -> None:
    reply = chat_reply(content=_WORDS[parser])
    reply["choices"][0]["finish_reason"] = "length"
    decision = await _decide(_adapter(parser, reply), tmp_path, _state())
    assert decision.rationale.startswith(INVALID_PREFIX) and decision.words_only is None


async def test_adapter_empty_reply_is_marked_with_empty_words(tmp_path: Path) -> None:
    decision = await _decide(_adapter("neutral_tools", chat_reply(content="")), tmp_path, _state())
    assert decision.words_only == ""


@pytest.mark.parametrize(
    ("parser", "reply"),
    [
        ("neutral_tools", chat_reply(tool_calls=[tool_call("teleport")])),
        ("neutral_tools", chat_reply(tool_calls=[tool_call("click", x="far", y=1)])),
        ("neutral_tools", chat_reply(content="x <tool_call>{not json}</tool_call>")),
        ("neutral_tools", chat_reply(tool_calls=[tool_call("click", x=500, y=100)])),
        ("neutral_tools", chat_reply(tool_calls=[tool_call("fail", reason="blocked")])),
        ("neutral_tools", chat_reply(tool_calls=[tool_call("done", summary="ok")])),
        ("ui_tars", chat_reply(content="Thought: x\nAction: teleport(start_box='(1,2)')")),
        ("ui_tars", chat_reply(content="Thought: x\nAction: click(start_box='(500,100)')")),
        ("opencua", chat_reply(content="## Code\n```python\nos.system('ls')\n```")),
    ],
)
async def test_adapter_actions_give_ups_and_bad_calls_are_not_marked(tmp_path: Path, parser: str, reply: dict) -> None:
    decision = await _decide(_adapter(parser, reply), tmp_path, _state())
    assert decision.words_only is None


def test_the_three_no_action_parses_set_no_action() -> None:
    ctx = PromptContext(intent="t", environment="browser", mapper=None, task_prompt="t")  # type: ignore[arg-type]
    for name, text in _WORDS.items():
        assert get_format(name).parse({"role": "assistant", "content": text}, ctx).no_action is True
