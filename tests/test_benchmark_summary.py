"""Focused tests for local benchmark summary output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from src.models.benchmark import BenchmarkTaskSpec, BenchmarkTaskType
from src.models.common import FailureCategory, LoopStage, RunStatus, StopReason
from src.models.execution import ExecutedAction, ExecutionAttemptTrace, ExecutionTrace
from src.models.logs import FailureRecord, ModelDebugArtifacts, PreStepFailureLog, StepLog
from src.models.perception import ScreenPerception, UIElement, UIElementNameSource, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.progress import ProgressState
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_logger import append_step_log
from src.store.summary import generate_run_metrics, generate_suite_summary, summarize_runs


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
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
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.SUCCEEDED,
        step_count=1,
        retry_counts={},
        stop_reason=StopReason.FORM_SUBMITTED_SUCCESS,
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
            perception=ScreenPerception(summary="Form submitted successfully", page_hint="form_success", capture_artifact_path=str(success_step / "before.png"), visible_elements=[]),
            policy_decision=PolicyDecision(action=success_action, rationale="Form success visible.", confidence=1.0, active_subgoal="verify_success"),
            executed_action=ExecutedAction(action=success_action, success=True, detail="stopped"),
            verification_result=VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=True, reason="form success reached", stop_reason=StopReason.FORM_SUBMITTED_SUCCESS),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="stop", terminal=True, recoverable=False, stop_reason=StopReason.FORM_SUBMITTED_SUCCESS),
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
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.FAILED,
            step_count=12,
            retry_counts={"submit_form:uncertain": 2},
            stop_reason=StopReason.MAX_STEP_LIMIT_REACHED,
        ),
    )

    precondition_run = root_dir / "run-precondition"
    precondition_step = precondition_run / "step_1"
    precondition_step.mkdir(parents=True, exist_ok=True)
    _write_state(
        precondition_run,
        AgentState(
            run_id="run-precondition",
            intent="Create a Gmail draft",
            status=RunStatus.FAILED,
            step_count=1,
            retry_counts={},
            stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED,
        ),
    )
    precondition_action = AgentAction(action_type=ActionType.STOP)
    append_step_log(
        precondition_run / "run.jsonl",
        StepLog(
            run_id="run-precondition",
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(precondition_step / "before.png"),
            after_artifact_path=str(precondition_step / "after.png"),
            perception_debug=_debug("perception", precondition_step),
            policy_debug=_debug("policy", precondition_step),
            perception=ScreenPerception(summary="Sign-in page", page_hint="google_sign_in", capture_artifact_path=str(precondition_step / "before.png"), visible_elements=[]),
            policy_decision=PolicyDecision(action=precondition_action, rationale="Authenticated start required.", confidence=1.0, active_subgoal="stop for benchmark setup"),
            executed_action=ExecutedAction(action=precondition_action, success=True, detail="stopped"),
            verification_result=VerificationResult(status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=True, reason="benchmark precondition failed", stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="stop benchmark", failure_category=FailureCategory.BENCHMARK_PRECONDITION_FAILED, failure_stage=LoopStage.CHOOSE_ACTION, terminal=True, recoverable=False, stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED),
            failure=FailureRecord(category=FailureCategory.BENCHMARK_PRECONDITION_FAILED, stage=LoopStage.CHOOSE_ACTION, retry_count=0, terminal=True, recoverable=False, reason="authenticated start required", stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED),
        ),
    )

    output = summarize_runs(str(root_dir), root_dir=root_dir)

    assert "total_runs: 4" in output
    assert "success_count: 1" in output
    assert "failure_count: 3" in output
    assert "success_rate: 25.00%" in output
    assert "average_steps_per_run: 4.00" in output
    assert "average_retries_per_run: 0.00" in output
    assert "form_submitted_success: 1" in output
    assert "benchmark_precondition_failed: 1" in output
    assert "retry_limit_reached: 1" in output
    assert "max_step_limit_reached: 1" in output
    assert "successful_stop_reasons:" in output
    assert "failed_stop_reasons:" in output
    assert "retry_limit_reached: 1" in output
    assert "choose_action: 1" in output
    assert "recover: 1" in output
    assert "orchestrate: 1" in output



def test_benchmark_summary_accepts_single_run_id() -> None:
    root_dir = _local_test_dir("test-benchmark-summary-single") / "runs"
    run_dir = root_dir / "run-single"
    _write_state(
        run_dir,
        AgentState(
            run_id="run-single",
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.SUCCEEDED,
            step_count=1,
            stop_reason=StopReason.FORM_SUBMITTED_SUCCESS,
        ),
    )

    output = summarize_runs("run-single", root_dir=root_dir)

    assert "total_runs: 1" in output
    assert "success_count: 1" in output


def test_benchmark_summary_skips_invalid_legacy_run_states() -> None:
    root_dir = _local_test_dir("test-benchmark-summary-invalid-legacy") / "runs"

    valid_run = root_dir / "run-valid"
    _write_state(
        valid_run,
        AgentState(
            run_id="run-valid",
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.SUCCEEDED,
            step_count=1,
            stop_reason=StopReason.FORM_SUBMITTED_SUCCESS,
        ),
    )

    invalid_run = root_dir / "run-invalid"
    invalid_run.mkdir(parents=True, exist_ok=True)
    (invalid_run / "state.json").write_text(
        """
        {
          "run_id": "run-invalid",
          "intent": "legacy run",
          "status": "failed",
          "step_count": 1,
          "observation_history": [
            {
              "summary": "legacy invalid page hint",
              "page_hint": null,
              "focused_element_id": null,
              "capture_artifact_path": "runs/run-invalid/step_1/before.png",
              "visible_elements": [],
              "confidence": 0.0
            }
          ],
          "action_history": [],
          "verification_history": [],
          "retry_counts": {},
          "target_failure_counts": {},
          "artifact_paths": [],
          "stop_reason": null
        }
        """,
        encoding="utf-8",
    )

    output = summarize_runs(str(root_dir), root_dir=root_dir)

    assert "total_runs: 1" in output
    assert "invalid_run_state_files_skipped: 1" in output


def test_benchmark_summary_counts_pre_step_perception_failure_records() -> None:
    root_dir = _local_test_dir("test-benchmark-summary-pre-step-failure") / "runs"
    run_dir = root_dir / "run-pre-step-failure"
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_state(
        run_dir,
        AgentState(
            run_id="run-pre-step-failure",
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.FAILED,
            step_count=0,
            stop_reason=StopReason.PRE_STEP_PERCEPTION_FAILED,
        ),
    )
    append_step_log(
        run_dir / "run.jsonl",
        PreStepFailureLog(
            run_id="run-pre-step-failure",
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            perception_debug=_debug("perception", step_dir),
            failure=FailureRecord(
                category=FailureCategory.PRE_STEP_PERCEPTION_FAILED,
                stage=LoopStage.PERCEIVE,
                retry_count=0,
                terminal=True,
                recoverable=False,
                reason="Gemini perception output did not match the strict schema.",
                stop_reason=StopReason.PRE_STEP_PERCEPTION_FAILED,
            ),
            error_message="Gemini perception output did not match the strict schema.",
        ),
    )

    output = summarize_runs("run-pre-step-failure", root_dir=root_dir)

    assert "total_runs: 1" in output
    assert "failure_count: 1" in output
    assert "average_steps_per_run: 0.00" in output
    assert "pre_step_perception_failed: 1" in output
    assert "perceive: 1" in output
    assert "failed_stop_reasons:" in output
    assert "pre_step_perception_failed: 1" in output


def test_generate_run_metrics_collects_selector_perception_and_execution_signals() -> None:
    root_dir = _local_test_dir("test-run-metrics") / "runs"
    run_dir = root_dir / "run-metrics"
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    state = AgentState(
        run_id="run-metrics",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.FAILED,
        step_count=1,
        stop_reason=StopReason.CLICK_NO_EFFECT,
    )
    _write_state(run_dir, state)
    (step_dir / "perception_retry_log.txt").write_text("attempt=1\nattempt=2\n", encoding="utf-8")
    (step_dir / "selector_trace.json").write_text(
        json.dumps(
            [
                {
                    "intent": {
                        "action": "click",
                        "target_text": "submit",
                        "target_role": None,
                        "expected_element_types": ["button"],
                        "value_to_type": None,
                        "expected_section": "form",
                    },
                    "candidate_count": 2,
                    "top_candidates": [
                        {
                            "element_id": "submit-button",
                            "element_type": "button",
                            "primary_name": "Submit",
                            "total_score": 96.0,
                            "matched_signals": ["exact_primary_name_match"],
                            "rejected_by": [],
                            "action_compatible": True,
                            "exact_semantic_match": True,
                            "uses_unlabeled_fallback": False,
                            "nearest_matched_text_candidate_id": None,
                            "spatial_grounding_contributed": False,
                            "confidence_band": "high",
                        }
                    ],
                    "selected_element_id": "submit-button",
                    "decision_reason": "accepted",
                    "rejection_reason": None,
                    "score_margin": 14.0,
                    "initial_failure_reason": "ambiguous_target_candidates",
                    "recovery_attempted": True,
                    "recovery_strategy_used": "margin_relaxation",
                    "adjusted_acceptance_threshold": 80.0,
                    "adjusted_ambiguity_margin": 6.0,
                    "final_decision": "success",
                    "final_stop_reason": "selector_recovery_used",
                    "recovery_changed_selected_candidate": False,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    append_step_log(
        run_dir / "run.jsonl",
        StepLog(
            run_id="run-metrics",
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            after_artifact_path=str(step_dir / "after.png"),
            perception_debug=ModelDebugArtifacts(
                prompt_artifact_path=str(step_dir / "perception_prompt.txt"),
                raw_response_artifact_path=str(step_dir / "perception_raw.txt"),
                parsed_artifact_path=str(step_dir / "perception_parsed.json"),
                retry_log_artifact_path=str(step_dir / "perception_retry_log.txt"),
            ),
            policy_debug=ModelDebugArtifacts(
                prompt_artifact_path=str(step_dir / "policy_prompt.txt"),
                raw_response_artifact_path=str(step_dir / "policy_raw.txt"),
                parsed_artifact_path=str(step_dir / "policy_decision.json"),
                selector_trace_artifact_path=str(step_dir / "selector_trace.json"),
            ),
            perception=ScreenPerception(
                summary="Form visible",
                page_hint="form_page",
                capture_artifact_path=str(step_dir / "before.png"),
                visible_elements=[
                    UIElement(
                        element_id="submit-button",
                        element_type=UIElementType.BUTTON,
                        label="Submit",
                        primary_name="Submit",
                        name_source=UIElementNameSource.LABEL,
                        is_unlabeled=False,
                        usable_for_targeting=True,
                        x=10,
                        y=10,
                        width=20,
                        height=20,
                        is_interactable=True,
                        confidence=0.9,
                    ),
                    UIElement(
                        element_id="mystery-input",
                        element_type=UIElementType.INPUT,
                        primary_name="unlabeled_input",
                        name_source=UIElementNameSource.SYNTHETIC,
                        is_unlabeled=True,
                        usable_for_targeting=False,
                        x=40,
                        y=10,
                        width=20,
                        height=20,
                        is_interactable=True,
                        confidence=0.5,
                    ),
                ],
            ),
            policy_decision=PolicyDecision(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button"),
                rationale="Submit",
                confidence=0.9,
                active_subgoal="submit_form",
            ),
            executed_action=ExecutedAction(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button"),
                success=False,
                detail="no effect",
                failure_category=FailureCategory.CLICK_NO_EFFECT,
                failure_stage=LoopStage.EXECUTE,
                execution_trace=ExecutionTrace(
                    attempts=[
                        ExecutionAttemptTrace(
                            attempt_index=1,
                            revalidation_result="ok",
                            verification_result="click_no_effect",
                            no_progress_detected=True,
                            failure_category=FailureCategory.CLICK_NO_EFFECT,
                        )
                    ],
                    retry_attempted=True,
                    retry_reason=FailureCategory.CLICK_NO_EFFECT,
                    final_outcome="failure",
                    final_failure_category=FailureCategory.CLICK_NO_EFFECT,
                ),
            ),
            verification_result=VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="no effect",
                failure_category=FailureCategory.CLICK_NO_EFFECT,
                failure_stage=LoopStage.VERIFY,
            ),
            recovery_decision=RecoveryDecision(
                strategy=RecoveryStrategy.STOP,
                message="stop",
                failure_category=FailureCategory.CLICK_NO_EFFECT,
                failure_stage=LoopStage.RECOVER,
                terminal=True,
                recoverable=False,
                stop_reason=StopReason.CLICK_NO_EFFECT,
            ),
            progress_state=ProgressState(no_progress_streak=1, loop_detected=True),
            failure=FailureRecord(
                category=FailureCategory.CLICK_NO_EFFECT,
                stage=LoopStage.RECOVER,
                retry_count=1,
                terminal=True,
                recoverable=False,
                reason="no effect",
                stop_reason=StopReason.CLICK_NO_EFFECT,
            ),
        ),
    )

    metrics = generate_run_metrics(
        "run-metrics",
        root_dir=root_dir,
        task_spec=BenchmarkTaskSpec(
            task_id="task-submit",
            page_url="https://example.test/form",
            task_type=BenchmarkTaskType.FORM_SUBMIT,
            intent="Complete the auth-free form and submit it successfully.",
            expected_completion_signal="form success",
            difficulty_tags=["single_page", "unlabeled_fields"],
        ),
    )

    assert metrics.task_id == "task-submit"
    assert metrics.perception_retry_count == 1
    assert metrics.selector_recovery_count == 1
    assert metrics.execution_retry_count == 1
    assert metrics.no_progress_events == 1
    assert metrics.loop_detected is True
    assert metrics.average_top_selector_score == 96.0
    assert metrics.average_selector_margin == 14.0
    assert metrics.average_total_elements == 2.0
    assert metrics.average_labeled_elements == 1.0
    assert metrics.average_unlabeled_elements == 1.0
    assert metrics.average_usable_elements == 1.0
    assert metrics.click_no_effect_events >= 1


def test_generate_suite_summary_groups_by_task_type_and_tag() -> None:
    summary = generate_suite_summary(
        [
            generate_run_metrics_fixture(
                run_id="run-a",
                task_id="task-a",
                task_type=BenchmarkTaskType.FORM_SUBMIT,
                success=True,
                tags=["single_page"],
            ),
            generate_run_metrics_fixture(
                run_id="run-b",
                task_id="task-b",
                task_type=BenchmarkTaskType.MULTI_STEP_FORM,
                success=False,
                tags=["multi_step", "dynamic_update"],
            ),
        ],
        suite_id="suite-fixture",
    )

    assert summary.total_runs == 2
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.success_rate_by_task_type["form_submit"] == 100.0
    assert summary.success_rate_by_task_type["multi_step_form"] == 0.0
    assert summary.tag_summary["single_page"]["run_count"] == 1
    assert summary.tag_summary["multi_step"]["run_count"] == 1


def generate_run_metrics_fixture(
    *,
    run_id: str,
    task_id: str,
    task_type: BenchmarkTaskType,
    success: bool,
    tags: list[str],
):
    from src.models.benchmark import RunMetrics

    return RunMetrics(
        run_id=run_id,
        task_id=task_id,
        page_url="https://example.test",
        task_type=task_type,
        tags=tags,
        status=RunStatus.SUCCEEDED if success else RunStatus.FAILED,
        success=success,
        final_stop_reason=StopReason.FORM_SUBMITTED_SUCCESS if success else StopReason.CLICK_NO_EFFECT,
        failure_category=None if success else FailureCategory.CLICK_NO_EFFECT,
        step_count=3,
        perception_retry_count=1,
        selector_recovery_count=0,
        execution_retry_count=1,
        no_progress_events=0 if success else 1,
        loop_detected=not success,
        average_top_selector_score=90.0,
        average_selector_margin=10.0,
        selector_failure_count=0 if success else 1,
        average_total_elements=5.0,
        average_labeled_elements=4.0,
        average_unlabeled_elements=1.0,
        average_usable_elements=3.0,
        stale_target_events=0,
        focus_failures=0,
        click_no_effect_events=0 if success else 1,
        verification_failures=0 if success else 1,
    )
