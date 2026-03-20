"""Tests for file-backed state persistence and JSONL logging."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.models.common import RunStatus
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts, StepLog
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_logger import append_step_log
from src.store.run_store import FileBackedRunStore


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _debug(stage: str) -> ModelDebugArtifacts:
    parsed_name = "policy_decision.json" if stage == "policy" else "perception_parsed.json"
    return ModelDebugArtifacts(
        prompt_artifact_path=f"runs/run-1/step_1/{stage}_prompt.txt",
        raw_response_artifact_path=f"runs/run-1/step_1/{stage}_raw.txt",
        parsed_artifact_path=f"runs/run-1/step_1/{parsed_name}",
    )


@pytest.mark.asyncio
async def test_file_backed_run_store_persists_and_reloads_state() -> None:
    root_dir = _local_test_dir("test-file-backed-run-store") / "runs"
    store = FileBackedRunStore(root_dir=root_dir)
    created = store.create_run("Create a Gmail draft and stop before send.")

    perception = ScreenPerception(
        summary="Inbox visible",
        page_hint="gmail_inbox",
        capture_artifact_path=store.before_artifact_path(created.run_id, 1),
        visible_elements=[],
    )
    updated = await store.update_state(created.run_id, perception)
    updated.current_subgoal = "open compose"
    updated.artifact_paths.append(store.before_artifact_path(created.run_id, 1))
    await store.set_status(created.run_id, RunStatus.RUNNING)

    reloaded_store = FileBackedRunStore(root_dir=root_dir)
    reloaded = await reloaded_store.get_run(created.run_id)

    assert reloaded is not None
    assert reloaded.step_count == 1
    assert reloaded.observation_history[0].summary == "Inbox visible"
    assert reloaded.status is RunStatus.RUNNING



def test_jsonl_logging_appends_complete_step_entries() -> None:
    log_path = _local_test_dir("test-jsonl-logging") / "runs" / "run-1" / "run.jsonl"
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait for the page.", confidence=0.2, active_subgoal="wait for inbox")
    executed = ExecutedAction(action=action, success=True, detail="waited", artifact_path="runs/run-1/step_1/after.png")
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="continue",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="retry", retry_after_ms=1000)
    perception = ScreenPerception(
        summary="Inbox visible",
        page_hint="gmail_inbox",
        capture_artifact_path="runs/run-1/step_1/before.png",
        visible_elements=[],
    )
    entry = StepLog(
        run_id="run-1",
        step_id="step_1",
        step_index=1,
        before_artifact_path="runs/run-1/step_1/before.png",
        after_artifact_path="runs/run-1/step_1/after.png",
        perception_debug=_debug("perception"),
        policy_debug=_debug("policy"),
        perception=perception,
        policy_decision=decision,
        executed_action=executed,
        verification_result=verification,
        recovery_decision=recovery,
    )

    append_step_log(log_path, entry)
    append_step_log(log_path, entry.model_copy(update={"step_id": "step_2", "step_index": 2}))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["step_id"] == "step_1"
    assert payload["perception"]["capture_artifact_path"].endswith("before.png")
    assert payload["executed_action"]["artifact_path"].endswith("after.png")
    assert payload["policy_debug"]["parsed_artifact_path"].endswith("policy_decision.json")


@pytest.mark.asyncio
async def test_log_and_state_consistency_for_artifact_paths() -> None:
    store = FileBackedRunStore(root_dir=_local_test_dir("test-log-state-consistency") / "runs")
    state = store.create_run("Create a Gmail draft and stop before send.")

    before_path = store.before_artifact_path(state.run_id, 1)
    after_path = store.after_artifact_path(state.run_id, 1)
    perception = ScreenPerception(
        summary="Inbox visible",
        page_hint="gmail_inbox",
        capture_artifact_path=before_path,
        visible_elements=[],
    )
    state = await store.update_state(state.run_id, perception)

    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait for Gmail.", confidence=0.3, active_subgoal="wait for inbox")
    executed = ExecutedAction(action=action, success=True, detail="waited", artifact_path=after_path)
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="continue",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="retry", retry_after_ms=1000)
    perception_debug = ModelDebugArtifacts(
        prompt_artifact_path=str(Path(before_path).parent / "perception_prompt.txt"),
        raw_response_artifact_path=str(Path(before_path).parent / "perception_raw.txt"),
        parsed_artifact_path=str(Path(before_path).parent / "perception_parsed.json"),
    )
    policy_debug = ModelDebugArtifacts(
        prompt_artifact_path=str(Path(before_path).parent / "policy_prompt.txt"),
        raw_response_artifact_path=str(Path(before_path).parent / "policy_raw.txt"),
        parsed_artifact_path=str(Path(before_path).parent / "policy_decision.json"),
    )

    state.action_history.append(executed)
    state.verification_history.append(verification)
    state.artifact_paths.extend(
        [
            before_path,
            after_path,
            perception_debug.prompt_artifact_path,
            perception_debug.raw_response_artifact_path,
            perception_debug.parsed_artifact_path,
            policy_debug.prompt_artifact_path,
            policy_debug.raw_response_artifact_path,
            policy_debug.parsed_artifact_path,
        ]
    )
    await store.set_status(state.run_id, RunStatus.RUNNING)

    append_step_log(
        store.run_log_path(state.run_id),
        StepLog(
            run_id=state.run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=before_path,
            after_artifact_path=after_path,
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            perception=perception,
            policy_decision=decision,
            executed_action=executed,
            verification_result=verification,
            recovery_decision=recovery,
        ),
    )

    persisted = await store.get_run(state.run_id)
    assert persisted is not None
    assert persisted.artifact_paths[-1].endswith("policy_decision.json")

    payload = json.loads(store.run_log_path(state.run_id).read_text(encoding="utf-8").strip())
    assert payload["before_artifact_path"] == before_path
    assert payload["after_artifact_path"] == after_path
    assert payload["perception_debug"]["prompt_artifact_path"].endswith("perception_prompt.txt")
