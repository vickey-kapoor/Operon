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
        self.armed = False
        self.pending: tuple[str, str] | None = None
        self.accepted = 0

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.executed.append(action)
        if self.accept_all or self.armed:
            self.armed = False
            self.accepted += 1
        else:
            self.pending = ("confirm", ASKED)
        return ExecutedAction(action=action, success=True, detail="ok")

    def set_dialog_policy(self, run_id: str, *, accept: bool) -> None:
        self.accept_all = accept

    def arm_accept_next_dialog(self, run_id: str) -> None:
        self.armed = True

    def disarm_accept_next_dialog(self, run_id: str) -> None:
        self.armed = False

    def pop_pending_dialog(self, run_id: str) -> tuple[str, str] | None:
        pending, self.pending = self.pending, None
        return pending


def _loop(monkeypatch: pytest.MonkeyPatch, actions: list[AgentAction], *, mode: str | None = None):
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
        has_held_actions=lambda run_id: True,
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
    loop._trust_gate = TrustGate(enabled=True)
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
    assert executor.armed is True
    second = await loop.step_run(StepRequest(run_id="run-1"))

    assert executor.executed == [CLICK, CLICK]
    assert executor.accepted == 1 and executor.armed is False  # accepted exactly once, then disarmed
    assert second.status is RunStatus.RUNNING  # no second pause


async def test_declining_does_not_rerun_and_tells_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, executor, state, hints, policy = _loop(monkeypatch, [CLICK])
    await loop.step_run(StepRequest(run_id="run-1"))

    await loop.resume_run("run-1", approve_gated_action=False)

    assert executor.armed is False and len(executor.executed) == 1
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
