"""Computer Use backend: keys held with key_down carry across turns only once the turn ran."""

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


def _reply(*calls: dict) -> dict:
    names = ", ".join(call["name"] for call in calls)
    return {
        "perception": {"summary": f"Tool call: {names}.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": calls[0],
        "function_calls": list(calls),
        "rationale": names,
        "confidence": 0.7,
        "active_subgoal": names,
        "model_content": {"parts": [{"function_call": call} for call in calls]},
    }


def _key_down(key: str) -> dict:
    return {"name": "key_down", "args": {"key": key}}


def _press(key: str) -> dict:
    return {"name": "press_key", "args": {"key": key}}


class ScriptedClient:
    """Replies in order; the function response carries the error so tests can read it."""

    def __init__(self, replies: list[dict]) -> None:
        self._replies = replies
        self.turns: list[list[dict]] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        return self._replies[min(len(self.turns), len(self._replies)) - 1]

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        response = {"error": error} if error else {}
        return {
            "role": "user",
            "parts": [{"function_response": {"name": function_name, "response": response}}, {"inline_data": {"data": ""}}],
        }


def _backend(client: ScriptedClient) -> BrowserComputerUseBackend:
    return BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")


def _frame(tmp_path: Path, name: str) -> CaptureFrame:
    path = tmp_path / name
    path.write_bytes(b"png")
    return CaptureFrame(artifact_path=str(path), width=1000, height=1000)


def _state(*outcomes: bool) -> AgentState:
    """A run whose recorded actions succeeded or failed as given."""
    return AgentState(
        run_id="run-1",
        intent="Select all the text",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=100), success=ok, detail="ok" if ok else "failed")
            for ok in outcomes
        ],
    )


async def _decide(backend: BrowserComputerUseBackend, frame: CaptureFrame, state: AgentState):
    await backend.perceive(frame, state)
    perception = ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path=frame.artifact_path)
    return await backend.choose_action(state, perception)


async def test_a_key_held_in_one_turn_joins_the_next_turns_key_press(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_reply(_key_down("Control")), _reply(_press("a"))]))

    await _decide(backend, _frame(tmp_path, "a.png"), _state())
    decision = await _decide(backend, _frame(tmp_path, "b.png"), _state(True))

    assert decision.action.action_type is ActionType.HOTKEY
    assert decision.action.key == "Control+a"


async def test_a_hold_from_a_turn_that_failed_is_dropped(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_reply(_key_down("Control")), _reply(_press("a"))]))

    await _decide(backend, _frame(tmp_path, "a.png"), _state())
    decision = await _decide(backend, _frame(tmp_path, "b.png"), _state(False))

    # The key_down never ran, so a plain "a" must not become Control+a.
    assert decision.action.action_type is ActionType.PRESS_KEY
    assert decision.action.key == "a"


async def test_a_hold_from_calls_dropped_by_a_re_look_is_dropped(tmp_path: Path) -> None:
    backend = _backend(ScriptedClient([_reply(_key_down("Control")), _reply(_press("a"))]))

    await _decide(backend, _frame(tmp_path, "a.png"), _state())
    # A mid-step re-look: perceive again before the key_down was recorded as run.
    decision = await _decide(backend, _frame(tmp_path, "b.png"), _state())

    assert decision.action.action_type is ActionType.PRESS_KEY
    assert decision.action.key == "a"


async def test_a_click_with_a_key_held_is_reported_back_and_releases_the_key(tmp_path: Path) -> None:
    click = {"name": "click", "args": {"x": 500, "y": 500}}
    client = ScriptedClient([_reply(_key_down("Control"), click), _reply(_press("a"))])
    backend = _backend(client)

    await _decide(backend, _frame(tmp_path, "a.png"), _state())
    decision = await _decide(backend, _frame(tmp_path, "b.png"), _state(True))

    answer = client.turns[-1][-1]
    errors = [part["function_response"]["response"].get("error", "") for part in answer["parts"] if "function_response" in part]
    assert errors and all(error.startswith("Not run: click with Control held") for error in errors)
    assert "release the key (key_up)" in errors[0]
    # The skipped click ended the hold, so the next press is a plain "a".
    assert decision.action.action_type is ActionType.PRESS_KEY
    assert decision.action.key == "a"
