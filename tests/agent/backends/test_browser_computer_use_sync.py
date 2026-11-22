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


def _two_click_reply() -> dict:
    calls = [{"name": "click_at", "args": {"x": 500, "y": 500}}, {"name": "click_at", "args": {"x": 600, "y": 500}}]
    reply = _click_reply()
    return {
        **reply,
        "function_call": calls[0],
        "function_calls": calls,
        "model_content": {"parts": [{"function_call": c} for c in calls]},
    }


async def test_a_declined_dialog_tells_the_model_how_many_actions_of_the_turn_ran(tmp_path: Path) -> None:
    from operon.agent.loop import AgentLoop

    client = ScriptedClient([_two_click_reply(), _click_reply(700)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    backend.note_not_executed("run-1", "the page's confirm dialog was cancelled", ran=1)
    backend.discard_held_actions("run-1", "Not executed: the person declined" + AgentLoop._ran_before_text(1))
    await _step(backend, _frame(tmp_path, "b.png"), _state())

    error = _last_answer(client)["parts"][0]["function_response"]["response"]["error"]
    assert "first 1 action(s) of the turn DID run" in error and "must not be repeated" in error


async def test_an_approved_dialog_replays_only_the_calls_from_the_stopped_one(tmp_path: Path) -> None:
    client = ScriptedClient([_two_click_reply(), _click_reply(700)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())
    backend.note_not_executed("run-1", "the page's confirm dialog was cancelled", ran=1)

    assert backend.drop_held_calls_before("run-1", 1, 2) is True
    replay = await _step(backend, _frame(tmp_path, "b.png"), _state())

    assert replay.action.x is not None and len(client.turns) == 1  # one click replayed, no model turn


async def test_a_dialog_stopped_batch_reaches_the_gemini_backend_through_the_policy(tmp_path: Path) -> None:
    from operon.agent.policy.thin import ThinPolicy

    client = ScriptedClient([_two_click_reply(), _click_reply(700)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())

    ThinPolicy(backend).note_not_executed("run-1", "the page's confirm dialog was cancelled", ran=1)

    assert backend.has_held_actions("run-1")  # held: the person's answer replays from the stopped call
    assert backend.drop_held_calls_before("run-1", 1, 2) is True


async def test_a_failed_turn_after_a_declined_dialog_still_reports_the_decline_on_the_retry(tmp_path: Path) -> None:
    """KAN-173 rolls a failed turn back; the dialog decline (and the count of actions that ran) must survive it."""
    import pytest

    from operon.agent.backends.base import BackendCompatibilityError
    from operon.agent.loop import AgentLoop

    class FlakyClient(ScriptedClient):
        fail_next = False

        async def generate_turn(self, *, contents: list[dict]) -> dict:
            if self.fail_next:
                self.fail_next = False
                raise BackendCompatibilityError("HTTP 503")
            return await super().generate_turn(contents=contents)

    client = FlakyClient([_two_click_reply(), _click_reply(700)])
    backend = _backend(client)
    await _step(backend, _frame(tmp_path, "a.png"), _state())
    backend.note_not_executed("run-1", "the page's confirm dialog was cancelled", ran=1)
    backend.discard_held_actions("run-1", "Not executed: the person declined" + AgentLoop._ran_before_text(1))

    client.fail_next = True
    with pytest.raises(BackendCompatibilityError):
        await backend.perceive(_frame(tmp_path, "b.png"), _state())
    await _step(backend, _frame(tmp_path, "c.png"), _state())  # the retry

    error = _last_answer(client)["parts"][0]["function_response"]["response"]["error"]
    assert "declined" in error and "first 1 action(s) of the turn DID run" in error
