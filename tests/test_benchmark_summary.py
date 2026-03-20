"""Focused tests for local benchmark summary output."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.models.common import FailureCategory, LoopStage, RunStatus, StopReason
from src.models.execution import ExecutedAction
from src.models.logs import FailureRecord, ModelDebugArtifacts, StepLog
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_logger import append_step_log
from src.store.summary import summarize_runs


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _debug(stage: str, step_dir: Path) -> ModelDebugArtifacts:
    parsed_name = "policy_decision.json" if stage == "policy" else "perception_parsed.json"
    return ModelDebugArtifacts(
        prompt_artifact_path=str(step_dir / f"{stage}_prompt.txt"),
        raw_response_artifact_path=str(step_dir / f"{stage}_raw.txt"),
        parsed_artifact_path=str(step_dir / parsed_name),
    )



def _write_state(run_dir: Path, state: AgentState) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")



def test_benchmark_summary_reports_fixture_runs() -> None:
    root_dir = _local_test_dir("test-benchmark-summary") / "runs"

    success_run = root_dir / "run-success"
    success_step = success_run / "step_1"
    success_step.mkdir(parents=True, exist_ok=True)
    success_state = AgentState(
        run_id="run-success",
        intent="Create a Gmail draft",
        status=RunStatus.SUCCEEDED,
        step_count=1,
        retry_counts={},
        stop_reason=StopReason.STOP_BEFORE_SEND,
    )
    _write_state(success_run, success_state)
    success_action = AgentAction(action_type=ActionType.STOP)
    append_step_log(
        success_run / "run.jsonl",
        StepLog(
            run_id="run-success",
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(success_step / "before.png"),
            after_artifact_path=str(success_step / "after.png"),
            perception_debug=_debug("perception", success_step),
            policy_debug=_debug("policy", success_step),
            perception=ScreenPerception(summary="Draft ready", page_hint="gmail_compose", capture_artifact_path=str(success_step / "before.png"), visible_elements=[]),
            policy_decision=PolicyDecision(action=success_action, rationale="Stop before send.", confidence=1.0, active_subgoal="stop before send"),
            executed_action=ExecutedAction(action=success_action, success=True, detail="stopped"),
            verification_result=VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=True, reason="stop boundary reached"),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="stop", terminal=True, recoverable=False, stop_reason=StopReason.STOP_BEFORE_SEND),
            failure=None,
        ),
    )

    retry_run = root_dir / "run-retry"
    retry_step = retry_run / "step_1"
    retry_step.mkdir(parents=True, exist_ok=True)
    retry_state = AgentState(
        run_id="run-retry",
        intent="Create a Gmail draft",
        status=RunStatus.FAILED,
        step_count=2,
        retry_counts={"open compose:action_failed": 3},
        stop_reason=StopReason.RETRY_LIMIT_REACHED,
    )
    _write_state(retry_run, retry_state)
    retry_action = AgentAction(action_type=ActionType.CLICK, target_element_id="missing")
    append_step_log(
        retry_run / "run.jsonl",
        StepLog(
            run_id="run-retry",
            step_id="step_2",
            step_index=2,
            before_artifact_path=str(retry_step / "before.png"),
            after_artifact_path=str(retry_step / "after.png"),
            perception_debug=_debug("perception", retry_step),
            policy_debug=_debug("policy", retry_step),
            perception=ScreenPerception(summary="Inbox visible", page_hint="gmail_inbox", capture_artifact_path=str(retry_step / "before.png"), visible_elements=[]),
            policy_decision=PolicyDecision(action=retry_action, rationale="Open compose.", confidence=0.7, active_subgoal="open compose"),
            executed_action=ExecutedAction(action=retry_action, success=False, detail="Execution failed", failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND, failure_stage=LoopStage.EXECUTE),
            verification_result=VerificationResult(status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="action failed", failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND, failure_stage=LoopStage.EXECUTE),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="retry limit reached", failure_category=FailureCategory.RETRY_LIMIT_REACHED, failure_stage=LoopStage.RECOVER, terminal=True, recoverable=False, stop_reason=StopReason.RETRY_LIMIT_REACHED),
            failure=FailureRecord(category=FailureCategory.RETRY_LIMIT_REACHED, stage=LoopStage.RECOVER, retry_count=3, terminal=True, recoverable=False, reason="retry limit reached", stop_reason=StopReason.RETRY_LIMIT_REACHED),
        ),
    )

    max_run = root_dir / "run-max"
    _write_state(
        max_run,
        AgentState(
            run_id="run-max",
            intent="Create a Gmail draft",
            status=RunStatus.FAILED,
            step_count=12,
            retry_counts={"compose:uncertain": 2},
            stop_reason=StopReason.MAX_STEP_LIMIT_REACHED,
        ),
    )

    output = summarize_runs(str(root_dir), root_dir=root_dir)

    assert "total_runs: 3" in output
    assert "success_count: 1" in output
    assert "failure_count: 2" in output
    assert "success_rate: 33.33%" in output
    assert "average_steps_per_run: 5.00" in output
    assert "average_retries_per_run: 1.67" in output
    assert "stop_before_send: 1" in output
    assert "retry_limit_reached: 1" in output
    assert "max_step_limit_reached: 1" in output
    assert "retry_limit_reached: 1" in output
    assert "recover: 1" in output
    assert "orchestrate: 1" in output



def test_benchmark_summary_accepts_single_run_id() -> None:
    root_dir = _local_test_dir("test-benchmark-summary-single") / "runs"
    run_dir = root_dir / "run-single"
    _write_state(
        run_dir,
        AgentState(
            run_id="run-single",
            intent="Create a Gmail draft",
            status=RunStatus.SUCCEEDED,
            step_count=1,
            stop_reason=StopReason.STOP_BEFORE_SEND,
        ),
    )

    output = summarize_runs("run-single", root_dir=root_dir)

    assert "total_runs: 1" in output
    assert "success_count: 1" in output
