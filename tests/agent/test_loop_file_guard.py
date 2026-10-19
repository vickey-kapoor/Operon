"""The loop pauses — and never executes — a file action whose path the user did not name."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.loop import AgentLoop
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


def _debug() -> ModelDebugArtifacts:
    return ModelDebugArtifacts(
        prompt_artifact_path="p.txt", raw_response_artifact_path="r.txt", parsed_artifact_path="d.json"
    )


def _loop(intent: str, action: AgentAction, monkeypatch: pytest.MonkeyPatch):
    from operon.agent import hitl

    monkeypatch.setattr(hitl, "notify_desktop", lambda **_: None)
    monkeypatch.setattr(hitl, "post_hitl_webhook", AsyncMock())
    monkeypatch.setattr(hitl, "start_escalation_timer", AsyncMock())
    monkeypatch.delenv("OPERON_ALLOWED_FILE_ROOTS", raising=False)

    run_root = unique_artifact_dir("test-file-guard")
    state = AgentState(run_id="run-1", intent=intent, status=RunStatus.RUNNING)
    frame = CaptureFrame(artifact_path="frame.png", width=1280, height=800, mime_type="image/png")
    perception = ScreenPerception(summary="Article", page_hint="unknown", capture_artifact_path="frame.png", visible_elements=[])
    decision = PolicyDecision(action=action, rationale="save it", confidence=0.9, active_subgoal="save")
    executed = ExecutedAction(action=action, success=True, detail="done")
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=True, reason="ok"
    )

    run_store = SimpleNamespace(
        root_dir=str(run_root),
        get_run=AsyncMock(return_value=state),
        update_state=AsyncMock(return_value=state),
        save_state=AsyncMock(),
        set_status=AsyncMock(side_effect=lambda _id, status: state.model_copy(update={"status": status})),
        before_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "before.png"),
        after_artifact_path=lambda run_id, i: str(run_root / run_id / f"step_{i}" / "after.png"),
        run_log_path=lambda run_id: str(run_root / run_id / "run.jsonl"),
    )
    executor = SimpleNamespace(execute=AsyncMock(return_value=executed))
    loop = AgentLoop(
        capture_service=SimpleNamespace(capture=AsyncMock(return_value=frame)),
        perception_service=SimpleNamespace(perceive=AsyncMock(return_value=perception), latest_debug_artifacts=_debug),
        run_store=run_store,
        policy_service=SimpleNamespace(choose_action=AsyncMock(return_value=decision), latest_debug_artifacts=_debug),
        executor=executor,
        verifier_service=SimpleNamespace(verify=AsyncMock(return_value=verification)),
        recovery_manager=SimpleNamespace(
            recover=AsyncMock(return_value=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="done"))
        ),
    )
    return loop, executor


async def test_unnamed_read_text_path_pauses_without_executing(monkeypatch: pytest.MonkeyPatch) -> None:
    action = AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/home/me/.bashrc")
    loop, executor = _loop("Read the GPT-4 article and save it", action, monkeypatch)

    response = await loop.step_run(StepRequest(run_id="run-1"))

    assert response.status is RunStatus.WAITING_FOR_USER
    executor.execute.assert_not_called()


async def test_read_text_path_named_in_intent_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    action = AgentAction(action_type=ActionType.READ_TEXT, selector="main", text="/tmp/out/article.txt")
    loop, executor = _loop("Read the GPT-4 article and save it to /tmp/out/article.txt", action, monkeypatch)

    await loop.step_run(StepRequest(run_id="run-1"))

    executor.execute.assert_called_once()
