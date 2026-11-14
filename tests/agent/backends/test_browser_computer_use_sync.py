"""KAN-150: the Computer Use conversation stays in step with what Operon actually ran."""

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


def _click_reply(x: int = 500) -> dict:
    call = {"name": "click_at", "args": {"x": x, "y": 500}}
    return {
        "perception": {"summary": "Tool call: click_at.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": "click",
        "confidence": 0.7,
        "model_content": {"parts": [{"function_call": call}]},
    }


class ScriptedClient:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = replies
        self.turns: list[list[dict]] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append([dict(c, parts=list(c["parts"])) for c in contents])
        return self._replies[min(len(self.turns), len(self._replies)) - 1]

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        response = {"url": current_url or ""}
        if error:
            response["error"] = error
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
    return AgentState(
        run_id="run-1",
        intent="Open the Products page",
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(action=AgentAction(action_type=ActionType.CLICK, x=1, y=1), success=ok, detail="ok" if ok else "missed")
            for ok in outcomes
        ],
    )


def _perception(frame: CaptureFrame) -> ScreenPerception:
    return ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path=frame.artifact_path)


async def _step(backend, frame, state):
    await backend.perceive(frame, state)
    return await backend.choose_action(state, _perception(frame))


def _last_answer(client: ScriptedClient) -> dict:
    return client.turns[-1][-1]


def _texts(content: dict) -> list[str]:
    return [part["text"] for part in content["parts"] if "text" in part]


async def test_notes_queued_after_the_first_turn_reach_the_model(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.add_advisory_hints(["The user says: use the second result"], source="override", run_id="run-1")
    await _step(backend, _frame(tmp_path, "b.png"), _state(True))

    assert any("use the second result" in text for text in _texts(_last_answer(client)))


async def test_a_turn_with_no_notes_sends_only_the_answer(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())
    await _step(backend, _frame(tmp_path, "b.png"), _state(True))

    assert _texts(_last_answer(client)) == []


async def test_a_gate_pause_holds_the_call_and_approval_runs_it_without_a_new_turn(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_not_executed("run-1", "high-risk action needs approval")
    assert backend.has_held_actions("run-1")
    approved = await _step(backend, _frame(tmp_path, "b.png"), _state())

    assert approved.action.action_type is ActionType.CLICK
    assert len(client.turns) == 1
    # Not a model safety request: the answer must not claim an acknowledgement.
    await _step(backend, _frame(tmp_path, "c.png"), _state(True))
    response = _last_answer(client)["parts"][0]["function_response"]["response"]
    assert "safety_acknowledgement" not in response
    assert "error" not in response


async def test_a_gate_pause_declined_with_a_correction_reports_the_call_as_not_run(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_not_executed("run-1", "high-risk action needs approval")
    backend.discard_held_actions("run-1", "Not executed: the user declined this action and sent a correction.")
    await _step(backend, _frame(tmp_path, "b.png"), _state())

    assert "declined" in _last_answer(client)["parts"][0]["function_response"]["response"]["error"]


async def test_a_rule_running_something_else_is_reported_as_not_run(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_replaced("run-1", "scroll down")
    # The rule's own action ran and succeeded; the model's click did not.
    await _step(backend, _frame(tmp_path, "b.png"), _state(True))

    error = _last_answer(client)["parts"][0]["function_response"]["response"]["error"]
    assert "Operon ran scroll down instead" in error


async def test_a_perception_only_re_look_takes_no_model_turn_and_keeps_the_decision(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await backend.perceive(_frame(tmp_path, "a.png"), _state())

    backend.set_perception_only(True)
    await backend.perceive(_frame(tmp_path, "retry.png"), _state())
    backend.set_perception_only(False)

    decision = await backend.choose_action(_state(), _perception(_frame(tmp_path, "a.png")))
    assert len(client.turns) == 1
    assert decision.action.x == 500


async def test_a_failed_attempt_not_yet_recorded_is_reported_to_the_model(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(), _click_reply(600)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_attempt_failed("run-1", "click landed on empty space")
    await _step(backend, _frame(tmp_path, "b.png"), _state())

    assert _last_answer(client)["parts"][0]["function_response"]["response"]["error"] == "click landed on empty space"


async def test_decisions_of_two_runs_do_not_mix(tmp_path: Path) -> None:
    client = ScriptedClient([_click_reply(100), _click_reply(900)])
    backend = _backend(client)
    one, two = _state(), _state().model_copy(update={"run_id": "run-2"})

    await backend.perceive(_frame(tmp_path, "a.png"), one)
    await backend.perceive(_frame(tmp_path, "b.png"), two)

    assert (await backend.choose_action(one, _perception(_frame(tmp_path, "a.png")))).action.x == 100
    assert (await backend.choose_action(two, _perception(_frame(tmp_path, "b.png")))).action.x == 900


async def test_approving_held_calls_keeps_the_models_own_reason(tmp_path: Path) -> None:
    # A later gate (the trust gate reads the reason) must still see what the model said.
    reply = dict(_click_reply(), rationale="Click Place order to buy it")
    client = ScriptedClient([reply])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_not_executed("run-1", "low confidence")
    approved = await _step(backend, _frame(tmp_path, "b.png"), _state())

    assert approved.rationale == "Click Place order to buy it"
