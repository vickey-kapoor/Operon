"""A page's confirm dialog goes through the trust gate's pause, not a silent accept (KAN-172).

The executor cancels confirm/prompt/beforeunload dialogs and records them; after the action ran
the loop tells the model (advisory hint) and pauses the run for the person, the way a gate does.
Approving replays the held action once with the dialog accepted; declining does not re-run it.
Auto permission mode accepts the dialog and never pauses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.trust_gate import TrustGate
from operon.core.runtime import current_runtime
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from tests.fixtures.paths import unique_artifact_dir

CLICK = AgentAction(action_type=ActionType.CLICK, x=50, y=60)
ASKED = "Delete the account?"


def _debug() -> ModelDebugArtifacts:
    return ModelDebugArtifacts(prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="d")


class DialogExecutor:
    """A browser executor whose click raises a confirm dialog, answered like the real one."""

    def __init__(self) -> None:
        self.executed: list[AgentAction] = []
        self.accept_all = False
        self.armed: tuple[str, str] | None = None
        self.pending: tuple[str, str, int | None] | None = None
        self.accepted = 0
        self.fail_on: AgentAction | None = None
        self.driving = False
        # What dialog an action raises: (type, message, batch child) or None.
        self.dialog_for = lambda action: (
            ("confirm", ASKED, None) if action.action_type is ActionType.CLICK else None
        )

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.executed.append(action)
        if self.fail_on is not None and action == self.fail_on:
            raise RuntimeError("executor blew up")
        dialog = self.dialog_for(action)
        if dialog is not None:
            kind, message, child = dialog
            if self.accept_all or self.armed == (kind, message):
                self.armed = None
                self.accepted += 1
            else:
                self.pending = (kind, message, child)
        return ExecutedAction(action=action, success=True, detail="ok")

    def set_dialog_policy(self, run_id: str, *, accept: bool) -> None:
        self.accept_all = accept

    def arm_accept_next_dialog(self, run_id: str, kind: str, message: str) -> None:
        self.armed = (kind, message)

    def disarm_accept_next_dialog(self, run_id: str) -> None:
        self.armed = None

    def set_person_driving(self, run_id: str, driving: bool) -> None:
        self.driving = driving

    def pop_pending_dialog(self, run_id: str):
        pending, self.pending = self.pending, None
        return pending


def _loop(monkeypatch: pytest.MonkeyPatch, actions: list[AgentAction], *, mode: str | None = None, held: bool = True):
    from operon.agent import hitl

    monkeypatch.setattr(hitl, "notify_desktop", lambda **_: None)
    monkeypatch.setattr(hitl, "post_hitl_webhook", AsyncMock())
    monkeypatch.setattr(hitl, "start_escalation_timer", AsyncMock())
    monkeypatch.setattr(current_runtime().control, "confidence_threshold", 0.0)

    run_root = unique_artifact_dir("test-page-dialogs")
    state = AgentState(run_id="run-1", intent="Close my account", status=RunStatus.RUNNING)
    state.permission_mode = mode

    async def set_status(_run_id: str, status: RunStatus) -> AgentState:
        state.status = status
        return state

    async def update_state(_run_id: str, _perception) -> AgentState:
        return state

    run_store = SimpleNamespace(
        root_dir=str(run_root),
        get_run=AsyncMock(return_value=state),
        update_state=update_state,
        save_state=AsyncMock(),
        set_status=set_status,
        before_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "before.png"),
        after_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "after.png"),
        run_log_path=lambda run_id: str(run_root / run_id / "run.jsonl"),
    )
    hints = MagicMock()
    decisions = [PolicyDecision(action=a, rationale="r", confidence=0.9, active_subgoal="s") for a in actions]
    executor = DialogExecutor()
    policy = SimpleNamespace(
        choose_action=AsyncMock(side_effect=decisions),
        latest_debug_artifacts=_debug,
        note_not_executed=MagicMock(),
        has_held_actions=lambda run_id: held,
        drop_held_calls_before=MagicMock(return_value=True),
        discard_held_actions=MagicMock(),
    )
    loop = AgentLoop(
        capture_service=SimpleNamespace(
            capture=AsyncMock(return_value=CaptureFrame(artifact_path="f.png", width=10, height=10, mime_type="image/png"))
        ),
        perception_service=SimpleNamespace(
            perceive=AsyncMock(
                return_value=ScreenPerception(summary="s", page_hint="unknown", capture_artifact_path="f.png", visible_elements=[])
            ),
            latest_debug_artifacts=_debug,
            add_advisory_hints=hints,
        ),
        run_store=run_store,
        policy_service=policy,
        executor=executor,
        verifier_service=SimpleNamespace(
            verify=AsyncMock(
                return_value=VerificationResult(
                    status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="c"
                )
            )
        ),
        recovery_manager=SimpleNamespace(
            recover=AsyncMock(return_value=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="m", retry_after_ms=10))
        ),
    )
    loop._trust_gate = TrustGate(enabled=True, confirm=("purchase",))
    return loop, executor, state, hints, policy


def _hints(hints: MagicMock, source: str) -> list[str]:
    return [c.args[0][0] for c in hints.call_args_list if c.kwargs.get("source") == source]


async def test_a_cancelled_confirm_pauses_the_run_with_the_dialog_text(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK])

    first = await loop.step_run(StepRequest(run_id="run-1"))

    assert first.status is RunStatus.WAITING_FOR_USER
    assert ASKED in (state.pause_reason or "") and "Accept?" in (state.pause_reason or "")
    assert len(executor.executed) == 1  # the click ran once; the dialog was cancelled
    assert _hints(hints, "dialog") == [f"The page showed a confirm dialog: {ASKED!r}. Operon cancelled it and asked the person."]
    policy.note_not_executed.assert_called_once()  # the backend holds the action for the person's answer


async def test_approving_arms_accept_and_reruns_the_held_action_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1")
    assert executor.armed is None  # armed only right before the replay runs
    assert loop._dialog_approved == {"run-1": ("confirm", ASKED)}
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert executor.executed == [CLICK, CLICK]
    assert executor.accepted == 1 and executor.armed is None  # accepted exactly once, then disarmed
    assert second.status is RunStatus.RUNNING  # no second pause


async def test_declining_does_not_rerun_and_tells_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1", approve_gated_action=False)

    assert executor.armed is None and len(executor.executed) == 1
    policy.discard_held_actions.assert_called_once()
    assert "declined" in policy.discard_held_actions.call_args.args[1]
    declined = _hints(hints, "dialog_answer")
    assert len(declined) == 1 and "declined" in declined[0] and ASKED in declined[0]


async def test_auto_mode_accepts_the_dialog_and_never_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK], mode="auto")

    first = await loop.step_run(StepRequest(run_id="run-1"))

    assert first.status is RunStatus.RUNNING
    assert executor.accepted == 1
    assert _hints(hints, "dialog") == []
    policy.note_not_executed.assert_not_called()


async def test_a_later_dialog_is_not_accepted_without_a_new_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    await loop.step_run(StepRequest(run_id="run-1"))  # approved replay

    third = await loop.step_run(StepRequest(run_id="run-1"))  # the same click again: a fresh dialog

    assert third.status is RunStatus.WAITING_FOR_USER
    assert executor.accepted == 1


PURCHASE = AgentAction(action_type=ActionType.TYPE, text="purchase now", target_element_id="buy")
OTHER_CLICK = AgentAction(action_type=ActionType.CLICK, x=70, y=80)


async def test_an_accept_never_survives_a_gate_paused_replay_and_a_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Step 2 stands for the replay being gate-paused; the person then sends a correction and the
    # model's next, different action raises a dialog nobody approved.
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, PURCHASE, OTHER_CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")  # approves the dialog: armed
    gated = await loop.step_run(StepRequest(run_id="run-1"))
    assert gated.status is RunStatus.WAITING_FOR_USER and executor.armed is None  # a gate pause disarms
    await loop.resume_run("run-1", approve_gated_action=False)  # a correction

    third = await loop.step_run(StepRequest(run_id="run-1"))

    assert executor.accepted == 0
    assert third.status is RunStatus.WAITING_FOR_USER  # asked again


async def test_a_gate_approved_replay_still_gets_its_dialog_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    state.permission_mode = "ask"  # every action is gated now, the replay too
    gated = await loop.step_run(StepRequest(run_id="run-1"))
    assert gated.status is RunStatus.WAITING_FOR_USER and executor.armed is None
    await loop.resume_run("run-1")  # the gate is approved

    third = await loop.step_run(StepRequest(run_id="run-1"))

    assert executor.accepted == 1 and third.status is RunStatus.RUNNING  # the dialog approval survived the gate


async def test_an_execute_that_raises_leaves_nothing_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    executor.fail_on = CLICK

    try:
        await loop.step_run(StepRequest(run_id="run-1"))
    except Exception:  # noqa: BLE001 - strict errors may re-raise; only the disarm matters here
        pass

    assert executor.armed is None


async def test_a_different_dialog_than_the_approved_one_is_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    executor.dialog_for = lambda action: ("confirm", "Wipe the whole database?", None)

    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert executor.accepted == 0
    assert second.status is RunStatus.WAITING_FOR_USER
    assert "Wipe the whole database?" in (state.pause_reason or "")


async def test_without_held_calls_the_accept_is_still_bound_to_the_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK, OTHER_CLICK], held=False)
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")
    assert any("approved" in h for h in _hints(hints, "dialog_answer"))  # the model is told to repeat it
    executor.dialog_for = lambda action: ("confirm", "Something else?", None)

    second = await loop.step_run(StepRequest(run_id="run-1"))  # the model's next, different action

    assert executor.accepted == 0
    assert second.status is RunStatus.WAITING_FOR_USER


@pytest.mark.parametrize(
    ("mode", "gate_on", "expected"),
    [("ask", False, False), ("risky", False, True), ("auto", False, True), ("ask", True, False), ("risky", True, False), ("auto", True, True)],
)
def test_the_dialog_policy_follows_the_permission_mode(monkeypatch, mode, gate_on, expected) -> None:
    loop, executor, _state, _hints_mock, _policy = _loop(monkeypatch, [])
    loop._trust_gate = TrustGate(enabled=gate_on)

    loop._set_dialog_policy(SimpleNamespace(run_id="run-1", state=SimpleNamespace(permission_mode=mode)))

    assert executor.accept_all is expected


async def test_a_turn_replays_only_from_the_call_that_raised_the_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    typing = AgentAction(action_type=ActionType.TYPE, text="hello", x=1, y=2)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[typing, CLICK])
    loop, executor, state, hints, policy = _loop(monkeypatch, [batch, batch])
    executor.dialog_for = lambda action: ("confirm", ASKED, 1) if action.action_type is ActionType.BATCH else None
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1")

    policy.drop_held_calls_before.assert_called_once_with("run-1", 1, 2)  # the type already ran: skipped


async def test_when_the_ran_calls_cannot_be_dropped_the_model_repeats_the_action(monkeypatch: pytest.MonkeyPatch) -> None:
    typing = AgentAction(action_type=ActionType.TYPE, text="hello", x=1, y=2)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[typing, CLICK])
    loop, executor, state, hints, policy = _loop(monkeypatch, [batch])
    policy.drop_held_calls_before.return_value = False
    executor.dialog_for = lambda action: ("confirm", ASKED, 1) if action.action_type is ActionType.BATCH else None
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1")

    policy.discard_held_actions.assert_called_once()  # nothing replays: the text is never typed twice
    assert any("Repeat the action" in h for h in _hints(hints, "dialog_answer"))


def test_the_computer_use_backend_drops_only_calls_that_line_up_with_the_actions() -> None:
    from operon.agent.backends.browser_computer_use import (
        BrowserComputerUseBackend,
        _ConversationState,
    )

    backend = BrowserComputerUseBackend.__new__(BrowserComputerUseBackend)
    conversation = _ConversationState()
    conversation.held_calls = [{"name": "type_text_at"}, {"name": "click_at"}]
    backend._conversations = {"r": conversation}

    assert backend.drop_held_calls_before("r", 1, 3) is False  # a merged click+type: positions differ
    assert len(conversation.held_calls) == 2
    assert backend.drop_held_calls_before("r", 1, 2) is True
    assert conversation.held_calls == [{"name": "click_at"}]


async def test_a_dialog_left_over_from_before_the_step_is_not_blamed_on_its_action(monkeypatch: pytest.MonkeyPatch) -> None:
    wait = AgentAction(action_type=ActionType.WAIT, wait_ms=100)
    loop, executor, state, hints, policy = _loop(monkeypatch, [wait])
    executor.pending = ("confirm", "Stale?", None)  # fired after the previous action returned

    first = await loop.step_run(StepRequest(run_id="run-1"))

    assert first.status is RunStatus.RUNNING
    texts = _hints(hints, "dialog")
    assert len(texts) == 1 and "Stale?" in texts[0] and "not asked" in texts[0]


async def test_the_hint_is_truthful_when_the_step_pauses_for_another_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK])
    loop._pause_requests.add("run-1")  # the person pressed Pause while the step ran

    first = await loop.step_run(StepRequest(run_id="run-1"))

    assert first.status is RunStatus.WAITING_FOR_USER
    assert "run-1" not in loop._dialog_pauses
    texts = _hints(hints, "dialog")
    assert len(texts) == 1 and "not asked" in texts[0] and "and asked the person" not in texts[0]


async def test_approve_takeover_hand_back_leaves_nothing_approved_and_asks_again(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK, CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))
    await loop.resume_run("run-1")  # the dialog is approved, the replay has not run yet
    await loop.take_over("run-1")

    await loop.hand_back("run-1")  # resumes without approving anything

    assert loop._dialog_approved == {} and executor.armed is None
    replay = await loop.step_run(StepRequest(run_id="run-1"))
    assert executor.accepted == 0
    assert replay.status is RunStatus.WAITING_FOR_USER  # cancelled and asked about again


async def test_a_dialog_in_the_first_child_replays_from_index_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = AgentAction(action_type=ActionType.BATCH, actions=[CLICK, OTHER_CLICK])
    loop, executor, state, hints, policy = _loop(monkeypatch, [batch])
    executor.dialog_for = lambda action: ("confirm", ASKED, 0) if action.action_type is ActionType.BATCH else None
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1")

    policy.drop_held_calls_before.assert_called_once_with("run-1", 0, 2)  # index 0 is not "no index"


async def test_the_executor_is_told_when_the_person_takes_over_and_hands_back(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, _hints_mock, _policy = _loop(monkeypatch, [CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))  # paused on the dialog

    await loop.take_over("run-1")
    assert executor.driving is True  # their dialogs are theirs to answer
    await loop.hand_back("run-1")

    assert executor.driving is False


async def test_a_declined_dialog_in_a_stopped_batch_says_which_actions_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    typing = AgentAction(action_type=ActionType.TYPE, text="hello", x=1, y=2)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[typing, CLICK])
    loop, executor, state, hints, policy = _loop(monkeypatch, [batch])
    executor.dialog_for = lambda action: ("confirm", ASKED, 1) if action.action_type is ActionType.BATCH else None
    await loop.step_run(StepRequest(run_id="run-1"))
    # The backend is told at the pause that the first action ran, not that nothing did.
    assert policy.note_not_executed.call_args.kwargs == {"ran": 1}

    await loop.resume_run("run-1", approve_gated_action=False)

    reason = policy.discard_held_actions.call_args.args[1]
    assert "first 1 action(s) of the turn DID run and must not be repeated" in reason
    assert "declined" in reason
