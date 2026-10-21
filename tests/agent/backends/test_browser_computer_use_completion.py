"""Computer Use backend: text-only completion replies and decisions without a cached turn."""

from __future__ import annotations

from pathlib import Path

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState


def _text_reply(text: str) -> dict:
    return {
        "perception": {"summary": text, "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "action": {"action_type": "wait", "wait_ms": 1000},
        "rationale": text,
        "confidence": 0.4,
        "active_subgoal": "gather more browser context",
        "model_content": {"parts": [{"text": text}]},
    }


def _click_reply() -> dict:
    call = {"name": "click_at", "args": {"x": 500, "y": 500}}
    return {
        "perception": {"summary": "Tool call: click_at.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": "click",
        "confidence": 0.7,
        "active_subgoal": "click_at",
        "model_content": {"parts": [{"function_call": call}]},
    }


class ScriptedClient:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = replies
        self.turns: list[list[dict]] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        return self._replies[min(len(self.turns), len(self._replies)) - 1]

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        return {
            "role": "user",
            "parts": [{"function_response": {"name": function_name, "response": {}}}, {"inline_data": {"data": ""}}],
        }


def _backend(client: ScriptedClient) -> BrowserComputerUseBackend:
    return BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")


def _frame(tmp_path: Path, name: str = "frame.png") -> CaptureFrame:
    path = tmp_path / name
    path.write_bytes(b"png")
    return CaptureFrame(artifact_path=str(path), width=1000, height=1000)


def _state(*actions: ActionType) -> AgentState:
    return AgentState(
        run_id="run-1",
        intent="Open the Products page",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(action=AgentAction(action_type=a, x=1, y=1) if a is ActionType.CLICK else AgentAction(action_type=a, wait_ms=500), success=True, detail="ok")
            for a in actions
        ],
    )


async def test_text_only_reply_after_acting_is_a_stop(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_text_reply("I opened the Products page. The task is complete.")]))
    state = _state(ActionType.CLICK)

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.STOP


async def test_text_only_reply_before_acting_stays_a_wait(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_text_reply("Looking at the page.")]))
    state = _state()

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.WAIT


async def test_text_only_reply_after_only_waiting_stays_a_wait(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_text_reply("Still loading.")]))
    state = _state(ActionType.WAIT)

    perception = await backend.perceive(_frame(tmp_path), state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type is ActionType.WAIT


async def test_choose_action_without_a_cached_turn_takes_a_fresh_turn(tmp_path: Path) -> None:
    client = ScriptedClient([_text_reply("Waiting for the page."), _click_reply()])
    backend = _backend(client)
    state = _state(ActionType.CLICK)
    perception = await backend.perceive(_frame(tmp_path), state)
    await backend.choose_action(state, perception)

    # The loop reused the perception and skipped perceive(); the decision must
    # come from a new turn on the current frame instead of failing the step.
    reused = perception.model_copy(update={"capture_artifact_path": _frame(tmp_path, "next.png").artifact_path})
    decision = await backend.choose_action(state, reused)

    assert decision.action.action_type is ActionType.CLICK
    assert len(client.turns) == 2


async def test_turn_after_a_text_reply_shows_the_current_screen(tmp_path: Path) -> None:
    client = ScriptedClient([_text_reply("Waiting for the page."), _click_reply()])
    backend = _backend(client)
    state = _state(ActionType.WAIT)

    await backend.perceive(_frame(tmp_path), state)
    await backend.perceive(_frame(tmp_path, "next.png"), state)

    roles = [content["role"] for content in client.turns[-1]]
    assert roles == ["user", "model", "user"]


async def test_choose_action_without_any_frame_still_raises(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_click_reply()]))
    perception = ScreenPerception(summary="x", page_hint="unknown", visible_elements=[], capture_artifact_path="x.png")

    try:
        await backend.choose_action(_state(), perception)
    except RuntimeError as exc:
        assert "No cached computer-use decision" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


async def test_perception_marks_elements_as_not_reported(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_click_reply()]))

    perception = await backend.perceive(_frame(tmp_path), _state(ActionType.CLICK))

    assert perception.elements_reported is False


async def test_subgoal_names_the_call_so_a_second_field_is_not_already_done(tmp_path: Path) -> None:
    name = {"name": "type_text_at", "args": {"x": 100, "y": 100, "text": "Ada"}}
    email = {"name": "type_text_at", "args": {"x": 100, "y": 200, "text": "ada@example.com"}}

    subgoals = {BrowserComputerUseBackend._subgoal_for_calls([call]) for call in (name, email)}

    assert len(subgoals) == 2
    assert all(subgoal.startswith("type_text_at ") for subgoal in subgoals)


async def test_a_re_look_before_the_action_is_recorded_answers_the_open_tool_call(tmp_path: Path) -> None:
    """The API rejects a request ending with a model turn; a mid-step re-perceive
    (execution or adaptation retry) must answer the pending call first."""
    client = ScriptedClient([_click_reply()])
    backend = _backend(client)
    state = _state(ActionType.WAIT)

    await backend.perceive(_frame(tmp_path), state)
    await backend.perceive(_frame(tmp_path, "retry.png"), state)

    last = client.turns[-1][-1]
    assert last["role"] == "user"
    assert "function_response" in last["parts"][0]


def _confirm_click_reply() -> dict:
    call = {
        "name": "click_at",
        "args": {"x": 500, "y": 500, "safety_decision": {"decision": "require_confirmation", "explanation": "Submits a form."}},
    }
    reply = _click_reply()
    return {**reply, "function_call": call, "function_calls": [call], "model_content": {"parts": [{"function_call": call}]}}


async def test_a_confirmed_call_runs_after_resume_and_its_answer_acknowledges_the_safety_decision(tmp_path: Path) -> None:
    client = ScriptedClient([_confirm_click_reply(), _text_reply("Done.")])
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    paused = await backend.choose_action(_state(), ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path=str(tmp_path / "a.png")))
    assert paused.action.action_type is ActionType.WAIT_FOR_USER

    # The user resumed: the held click runs without asking the model again.
    await backend.perceive(_frame(tmp_path, "b.png"), _state())
    approved = await backend.choose_action(_state(), ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path=str(tmp_path / "b.png")))
    assert approved.action.action_type is ActionType.CLICK
    assert len(client.turns) == 1

    await backend.perceive(_frame(tmp_path, "c.png"), _state(ActionType.CLICK))
    answer = client.turns[-1][-1]
    assert answer["parts"][0]["function_response"]["response"]["safety_acknowledgement"] == "true"
