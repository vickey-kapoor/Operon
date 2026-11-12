"""Gated actions run only after a human approves them, and a stop stops (KAN-46)."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.actions.translation import normalize_computer_use_actions
from operon.agent.loop import StartUrlBlockedError
from operon.agent.policy.trust_gate import TrustGate
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction
from tests.agent.backends.test_browser_computer_use_completion import (
    ScriptedClient,
    _backend,
    _click_reply,
    _frame,
    _state,
)
from tests.agent.test_trust_gate_resume import PURCHASE, _executed_types, _loop

SAFE = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box")


# G1: /step must not run a run that is paused, cancelled or finished.


@pytest.mark.parametrize(
    "status",
    [RunStatus.WAITING_FOR_USER, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
)
async def test_step_does_not_run_a_run_that_is_not_active(monkeypatch: pytest.MonkeyPatch, status: RunStatus) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE])
    state.status = status

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert response.status is status
    execute.assert_not_called()
    loop.capture_service.capture.assert_not_called()


async def test_stepping_a_gate_pause_does_not_approve_it(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [PURCHASE, PURCHASE])

    first = await loop.step_run(StepRequest(run_id="run-1"))
    assert first.status is RunStatus.WAITING_FOR_USER

    # A client calling /step instead of /resume: the gated action must not run.
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert second.status is RunStatus.WAITING_FOR_USER
    assert _executed_types(execute) == []


# G2: a Gemini turn with any call that needs confirmation pauses as a whole.


def _call(name: str, *, confirm: bool = False, **args) -> dict:
    if confirm:
        args["safety_decision"] = {"decision": "require_confirmation", "explanation": "Places the order."}
    return {"name": name, "args": args}


def test_a_turn_with_a_call_needing_confirmation_pauses_before_any_call_runs() -> None:
    action = normalize_computer_use_actions(
        [
            _call("click_at", x=100, y=100),
            _call("click_at", confirm=True, x=500, y=500),
            _call("navigate", url="https://example.com/next"),
        ],
        screen_width=1000,
        screen_height=1000,
    )

    assert action["action_type"] == "wait_for_user"
    assert action["text"] == "Places the order."
    assert "actions" not in action


def test_a_turn_without_confirmation_still_batches() -> None:
    action = normalize_computer_use_actions(
        [_call("click_at", x=100, y=100), _call("navigate", url="https://example.com/")],
        screen_width=1000,
        screen_height=1000,
    )

    assert action["action_type"] == "batch"
    assert len(action["actions"]) == 2


# G3: resuming with a correction drops Gemini's held calls instead of running them.


class _RecordingClient(ScriptedClient):
    def __init__(self, replies: list[dict]) -> None:
        super().__init__(replies)
        self.errors: list[str | None] = []

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        self.errors.append(error)
        return super().build_function_response_content(
            function_name=function_name, screenshot_path=screenshot_path, current_url=current_url, error=error
        )


def _confirm_reply() -> dict:
    call = _call("click_at", confirm=True, x=500, y=500)
    reply = _click_reply()
    return {**reply, "function_call": call, "function_calls": [call], "model_content": {"parts": [{"function_call": call}]}}


def _perception(tmp_path: Path, name: str) -> ScreenPerception:
    return ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path=str(tmp_path / name))


async def test_a_correction_drops_the_held_calls_and_tells_the_model(tmp_path: Path) -> None:
    client = _RecordingClient([_confirm_reply(), _click_reply()])
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    paused = await backend.choose_action(_state(), _perception(tmp_path, "a.png"))
    assert paused.action.action_type is ActionType.WAIT_FOR_USER

    backend.discard_held_actions("run-1", "Not executed: the user declined.")

    # The next step asks the model again instead of running the held click.
    await backend.perceive(_frame(tmp_path, "b.png"), _state())
    assert len(client.turns) == 2
    assert client.errors == ["Not executed: the user declined."]
    answer = client.turns[-1][-1]
    assert "safety_acknowledgement" not in answer["parts"][0]["function_response"]["response"]


async def test_resume_with_a_correction_discards_held_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Places the order.")
    loop, _, _ = _loop(monkeypatch, [ask])
    discarded: list[str] = []
    loop.policy_service.discard_held_actions = lambda run_id, reason: discarded.append(run_id)

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1", approve_gated_action=False)

    assert discarded == ["run-1"]


async def test_resume_with_approval_keeps_held_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Places the order.")
    loop, _, _ = _loop(monkeypatch, [ask])
    discarded: list[str] = []
    loop.policy_service.discard_held_actions = lambda run_id, reason: discarded.append(run_id)

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")

    assert discarded == []


async def test_approving_the_models_own_confirmation_counts_as_an_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Saves the file.")
    loop, _, _ = _loop(monkeypatch, [ask])
    loop.policy_service.has_held_actions = lambda run_id: True

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")

    assert (await loop.run_store.get_run("run-1")).approval_count == 1


async def test_continuing_after_the_model_asked_for_help_is_not_an_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Please log in.")
    loop, _, _ = _loop(monkeypatch, [ask])
    loop.policy_service.has_held_actions = lambda run_id: False

    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")

    assert (await loop.run_store.get_run("run-1")).approval_count == 0


async def test_the_backend_reports_held_calls_until_they_run(tmp_path: Path) -> None:
    backend = _backend(_RecordingClient([_confirm_reply(), _click_reply()]))

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    assert backend.has_held_actions("run-1")
    await backend.perceive(_frame(tmp_path, "b.png"), _state())  # resumed: runs the held call
    assert not backend.has_held_actions("run-1")


# G4: a stop that lands during a step stops it for real.


async def test_a_stop_during_the_model_call_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE])
    choose = loop.policy_service.choose_action.side_effect

    async def stop_while_deciding(*args, **kwargs):
        state.status = RunStatus.CANCELLED  # /desktop/stop while the model thinks
        return choose(*args, **kwargs) if callable(choose) else next(choose)

    loop.policy_service.choose_action.side_effect = stop_while_deciding

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert response.status is RunStatus.CANCELLED
    assert state.status is RunStatus.CANCELLED
    execute.assert_not_called()


async def test_a_stop_during_execute_is_not_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE])

    async def stop_mid_action(action: AgentAction) -> ExecutedAction:
        state.status = RunStatus.CANCELLED
        return ExecutedAction(action=action, success=True, detail="ok")

    execute.side_effect = stop_mid_action

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert response.status is RunStatus.CANCELLED
    assert state.status is RunStatus.CANCELLED


async def test_the_run_store_keeps_a_stop_that_lands_during_capture(tmp_path: Path) -> None:
    from operon.store.run_store import FileBackedRunStore

    store = FileBackedRunStore(tmp_path)
    record = store.create_run("Open Calculator")
    await store.set_status(record.run_id, RunStatus.CANCELLED)

    updated = await store.update_state(
        record.run_id, ScreenPerception(summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path="f.png")
    )

    assert updated.status is RunStatus.CANCELLED


async def test_no_status_write_overrides_a_stop(tmp_path: Path) -> None:
    # A step in flight (a recovery reset, a perception failure) ends with its own
    # status write after the stop landed: the stop must win.
    from operon.store.run_store import FileBackedRunStore

    store = FileBackedRunStore(tmp_path)
    record = store.create_run("Open Calculator")
    await store.set_status(record.run_id, RunStatus.CANCELLED)

    for status in (RunStatus.RUNNING, RunStatus.FAILED, RunStatus.SUCCEEDED):
        assert (await store.set_status(record.run_id, status)).status is RunStatus.CANCELLED


async def test_a_stop_during_a_batch_runs_none_of_its_later_children(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.executor.desktop import DesktopExecutor

    executor = DesktopExecutor.__new__(DesktopExecutor)
    ran: list[ActionType] = []

    async def fake_execute(action):
        if action.action_type is ActionType.BATCH:
            return await DesktopExecutor._execute_batch(executor, action)
        ran.append(action.action_type)
        stopped["now"] = True  # /desktop/stop lands while the first child runs
        return ExecutedAction(action=action, success=True, detail="ok")

    stopped = {"now": False}

    async def check(child):
        return "the run was stopped" if stopped["now"] else None

    executor.execute = fake_execute
    executor.batch_child_check = check
    executor._ok = lambda action, detail, path: ExecutedAction(action=action, success=True, detail=detail)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[
        AgentAction(action_type=ActionType.WAIT, wait_ms=100), SAFE.model_copy(update={"x": 5, "y": 5})])

    result = await executor.execute(batch)

    assert ran == [ActionType.WAIT]
    assert not result.success and "stopped" in result.detail


async def test_the_loop_installs_a_stop_check_for_every_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, _, state = _loop(monkeypatch, [SAFE])
    loop.executor.batch_child_check = None
    batch = AgentAction(action_type=ActionType.BATCH, actions=[SAFE, SAFE])

    loop._install_batch_child_check("run-1", batch)
    assert await loop.executor.batch_child_check(SAFE) is None
    state.status = RunStatus.CANCELLED
    assert await loop.executor.batch_child_check(SAFE) == "the run was stopped"


async def test_a_correction_reaches_gemini_with_the_decline(tmp_path: Path) -> None:
    client = _RecordingClient([_confirm_reply(), _click_reply()])
    backend = _backend(client)

    await backend.perceive(_frame(tmp_path, "a.png"), _state())
    await backend.choose_action(_state(), _perception(tmp_path, "a.png"))
    backend.discard_held_actions("run-1", "Not executed: the user declined.")
    backend.add_advisory_hints(["Use the cheaper shipping option instead."], source="override", run_id="run-1")

    await backend.perceive(_frame(tmp_path, "b.png"), _state())
    answer = client.turns[-1][-1]
    texts = [part["text"] for part in answer["parts"] if "text" in part]
    assert any("Use the cheaper shipping option instead." in text for text in texts)


# G7: a start_url outside the domain allowlist is refused before the run exists.


async def test_start_url_outside_the_allowlist_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, _ = _loop(monkeypatch, [SAFE])
    loop._trust_gate = TrustGate(enabled=True, allow_domains=("example.com",))
    loop.run_store.create_run = lambda **_: pytest.fail("run must not be created")

    with pytest.raises(StartUrlBlockedError):
        await loop.start_run(RunTaskRequest(intent="Look around", start_url="https://evil.test/"))

    execute.assert_not_called()


async def test_start_url_inside_the_allowlist_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, state = _loop(monkeypatch, [SAFE])
    loop._trust_gate = TrustGate(enabled=True, allow_domains=("example.com",))
    loop.run_store.create_run = lambda **_: state

    await loop.start_run(RunTaskRequest(intent="Look around", start_url="https://shop.example.com/"))

    assert [c.args[0].url for c in execute.call_args_list] == ["https://shop.example.com/"]
