"""Local benchmark summary and metrics loaders for individual runs and suites."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from src.models.benchmark import (
    BenchmarkSuiteSummary,
    BenchmarkTaskSpec,
    BenchmarkTaskType,
    RunMetrics,
)
from src.models.common import FailureCategory, LoopStage, RunStatus, StopReason
from src.models.logs import PreStepFailureLog, StepLog
from src.models.selector import SelectorTrace
from src.models.state import AgentState
from src.store.replay import load_run_replay


def summarize_runs(target: str, root_dir: str | Path = "runs") -> str:
    """Summarize one run id or a directory of local runs."""
    states, invalid_state_paths = _load_states(target, root_dir=root_dir)
    run_metrics = [generate_run_metrics(state.run_id, root_dir=root_dir) for state in states]
    suite_summary = generate_suite_summary(run_metrics, suite_id=_summary_suite_id(target))

    stop_reasons = Counter(state.stop_reason.value for state in states if state.stop_reason is not None)
    failure_categories: Counter[str] = Counter()
    failing_stages: Counter[str] = Counter()
    for state in states:
        for step in _load_steps_for_state(state, root_dir=root_dir):
            if isinstance(step, StepLog) and step.failure is not None:
                failure_categories[step.failure.category.value] += 1
                failing_stages[step.failure.stage.value] += 1
            elif isinstance(step, PreStepFailureLog):
                failure_categories[step.failure.category.value] += 1
                failing_stages[step.failure.stage.value] += 1
        if state.stop_reason is StopReason.MAX_STEP_LIMIT_REACHED:
            failure_categories[FailureCategory.MAX_STEP_LIMIT_REACHED.value] += 1
            failing_stages[LoopStage.ORCHESTRATE.value] += 1

    lines = [
        f"total_runs: {suite_summary.total_runs}",
        f"success_count: {suite_summary.success_count}",
        f"failure_count: {suite_summary.failure_count}",
        f"success_rate: {suite_summary.overall_success_rate:.2f}%",
        f"average_steps_per_run: {suite_summary.average_step_count:.2f}",
        f"average_retries_per_run: {_average_total_retries(run_metrics):.2f}",
        f"invalid_run_state_files_skipped: {len(invalid_state_paths)}",
        "stop_reason_counts:",
    ]
    lines.extend(_format_counter(stop_reasons))
    lines.append("successful_stop_reasons:")
    lines.extend(
        _format_counter(Counter(state.stop_reason.value for state in states if state.status is RunStatus.SUCCEEDED and state.stop_reason is not None))
    )
    lines.append("failed_stop_reasons:")
    lines.extend(
        _format_counter(Counter(state.stop_reason.value for state in states if state.status is RunStatus.FAILED and state.stop_reason is not None))
    )
    lines.append("most_common_failure_categories:")
    lines.extend(_format_counter(failure_categories))
    lines.append("most_common_failing_stages:")
    lines.extend(_format_counter(failing_stages))
    lines.append("average_perception_retry_count:")
    lines.append(f"  {suite_summary.average_perception_retry_count:.2f}")
    lines.append("average_selector_recovery_count:")
    lines.append(f"  {suite_summary.average_selector_recovery_count:.2f}")
    lines.append("average_execution_retry_count:")
    lines.append(f"  {suite_summary.average_execution_retry_count:.2f}")
    lines.append("loop_detected_frequency:")
    lines.append(f"  {suite_summary.loop_detected_frequency:.2f}%")
    lines.append("no_progress_event_frequency:")
    lines.append(f"  {suite_summary.no_progress_event_frequency:.2f}")
    return "\n".join(lines)


def generate_run_metrics(
    run_id: str,
    *,
    root_dir: str | Path = "runs",
    task_spec: BenchmarkTaskSpec | None = None,
) -> RunMetrics:
    """Compute one structured metrics record from existing run artifacts."""
    state = _load_state_from_path(Path(root_dir) / run_id / "state.json")
    entries = _load_steps_for_state(state, root_dir=root_dir)
    resolved_task = task_spec or _default_task_spec_for_state(state)

    selector_scores: list[float] = []
    selector_margins: list[float] = []
    selector_failure_count = 0
    selector_recovery_count = 0
    perception_retry_count = 0
    execution_retry_count = 0
    no_progress_events = 0
    loop_detected = False
    stale_target_events = 0
    focus_failures = 0
    click_no_effect_events = 0
    verification_failures = 0
    total_elements: list[int] = []
    labeled_elements: list[int] = []
    unlabeled_elements: list[int] = []
    usable_elements: list[int] = []
    final_failure_category = _state_level_failure_category(state, entries)

    for entry in entries:
        if isinstance(entry, PreStepFailureLog):
            perception_retry_count += _retry_log_attempts(entry.perception_debug.retry_log_artifact_path)
            continue

        perception_retry_count += _retry_log_attempts(entry.perception_debug.retry_log_artifact_path)
        execution_retry_count += 1 if entry.executed_action.execution_trace and entry.executed_action.execution_trace.retry_attempted else 0
        no_progress_events += _execution_no_progress_events(entry)
        loop_detected = loop_detected or bool(entry.progress_state and entry.progress_state.loop_detected)
        verification_failures += 1 if entry.verification_result.status is not None and entry.verification_result.status.value == "failure" else 0
        total_elements.append(len(entry.perception.visible_elements))
        labeled_elements.append(sum(1 for element in entry.perception.visible_elements if not element.is_unlabeled))
        unlabeled_elements.append(sum(1 for element in entry.perception.visible_elements if element.is_unlabeled))
        usable_elements.append(sum(1 for element in entry.perception.visible_elements if element.usable_for_targeting))

        for category in _entry_failure_categories(entry):
            if category is FailureCategory.STALE_TARGET_BEFORE_ACTION:
                stale_target_events += 1
            if category in {FailureCategory.FOCUS_VERIFICATION_FAILED, FailureCategory.CLICK_BEFORE_TYPE_FAILED}:
                focus_failures += 1
            if category is FailureCategory.CLICK_NO_EFFECT:
                click_no_effect_events += 1

        selector_traces = _load_selector_traces(entry.policy_debug.selector_trace_artifact_path)
        for trace in selector_traces:
            if trace.top_candidates:
                selector_scores.append(trace.top_candidates[0].total_score)
            if trace.score_margin is not None:
                selector_margins.append(trace.score_margin)
            if trace.final_decision.value == "failure" or trace.rejection_reason is not None:
                selector_failure_count += 1
            if trace.recovery_attempted:
                selector_recovery_count += 1

    return RunMetrics(
        run_id=state.run_id,
        task_id=resolved_task.task_id,
        page_url=resolved_task.page_url,
        task_type=resolved_task.task_type,
        tags=list(resolved_task.difficulty_tags),
        status=state.status,
        success=state.status is RunStatus.SUCCEEDED,
        final_stop_reason=state.stop_reason,
        failure_category=final_failure_category,
        step_count=state.step_count,
        perception_retry_count=perception_retry_count,
        selector_recovery_count=selector_recovery_count,
        execution_retry_count=execution_retry_count,
        no_progress_events=no_progress_events,
        loop_detected=loop_detected,
        average_top_selector_score=_average(selector_scores),
        average_selector_margin=_average(selector_margins),
        selector_failure_count=selector_failure_count,
        average_total_elements=_average(total_elements),
        average_labeled_elements=_average(labeled_elements),
        average_unlabeled_elements=_average(unlabeled_elements),
        average_usable_elements=_average(usable_elements),
        stale_target_events=stale_target_events,
        focus_failures=focus_failures,
        click_no_effect_events=click_no_effect_events,
        verification_failures=verification_failures,
    )


def generate_suite_summary(
    run_metrics: list[RunMetrics],
    *,
    suite_id: str = "benchmark_suite",
) -> BenchmarkSuiteSummary:
    """Aggregate structured metrics across a suite of runs."""
    total_runs = len(run_metrics)
    success_count = sum(1 for metrics in run_metrics if metrics.success)
    failure_count = total_runs - success_count
    success_rate_by_task_type = _success_rate_by_task_type(run_metrics)
    stop_reasons = Counter(
        metrics.final_stop_reason.value
        for metrics in run_metrics
        if not metrics.success and metrics.final_stop_reason is not None
    )
    failure_categories = Counter(
        metrics.failure_category.value
        for metrics in run_metrics
        if metrics.failure_category is not None
    )
    top_failure_reasons = Counter()
    for metrics in run_metrics:
        if metrics.final_stop_reason is not None:
            top_failure_reasons[metrics.final_stop_reason.value] += 1
        elif metrics.failure_category is not None:
            top_failure_reasons[metrics.failure_category.value] += 1

    return BenchmarkSuiteSummary(
        suite_id=suite_id,
        total_runs=total_runs,
        success_count=success_count,
        failure_count=failure_count,
        overall_success_rate=(success_count / total_runs * 100.0) if total_runs else 0.0,
        success_rate_by_task_type=success_rate_by_task_type,
        failure_breakdown_by_stop_reason=dict(stop_reasons),
        failure_breakdown_by_failure_category=dict(failure_categories),
        average_step_count=_average([metrics.step_count for metrics in run_metrics]) or 0.0,
        average_perception_retry_count=_average([metrics.perception_retry_count for metrics in run_metrics]) or 0.0,
        average_selector_recovery_count=_average([metrics.selector_recovery_count for metrics in run_metrics]) or 0.0,
        average_execution_retry_count=_average([metrics.execution_retry_count for metrics in run_metrics]) or 0.0,
        average_top_selector_score=_average([metrics.average_top_selector_score for metrics in run_metrics if metrics.average_top_selector_score is not None]),
        average_selector_margin=_average([metrics.average_selector_margin for metrics in run_metrics if metrics.average_selector_margin is not None]),
        average_total_elements=_average([metrics.average_total_elements for metrics in run_metrics if metrics.average_total_elements is not None]),
        average_labeled_elements=_average([metrics.average_labeled_elements for metrics in run_metrics if metrics.average_labeled_elements is not None]),
        average_unlabeled_elements=_average([metrics.average_unlabeled_elements for metrics in run_metrics if metrics.average_unlabeled_elements is not None]),
        average_usable_elements=_average([metrics.average_usable_elements for metrics in run_metrics if metrics.average_usable_elements is not None]),
        loop_detected_frequency=(sum(1 for metrics in run_metrics if metrics.loop_detected) / total_runs * 100.0) if total_runs else 0.0,
        no_progress_event_frequency=_average([metrics.no_progress_events for metrics in run_metrics]) or 0.0,
        top_recurring_failure_reasons=dict(top_failure_reasons.most_common(10)),
        tag_summary=_tag_summary(run_metrics),
    )


def write_run_metrics(metrics: RunMetrics, *, root_dir: str | Path = "runs") -> Path:
    """Persist per-run metrics next to the run state."""
    path = Path(root_dir) / metrics.run_id / "run_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_suite_summary(summary: BenchmarkSuiteSummary, *, output_path: str | Path) -> Path:
    """Persist one suite-level summary artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_states(target: str, root_dir: str | Path) -> tuple[list[AgentState], list[Path]]:
    root = Path(root_dir)
    target_path = Path(target)
    if target_path.exists() and target_path.is_dir():
        run_dirs = sorted(path for path in target_path.iterdir() if path.is_dir())
        states: list[AgentState] = []
        invalid_paths: list[Path] = []
        for run_dir in run_dirs:
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                states.append(_load_state_from_path(state_path))
            except ValidationError:
                invalid_paths.append(state_path)
        return states, invalid_paths

    state_path = root / target / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Run state not found: {state_path}")
    try:
        return [_load_state_from_path(state_path)], []
    except ValidationError as exc:
        raise ValueError(f"Run state is invalid and could not be summarized: {state_path}") from exc


def _load_state_from_path(path: Path) -> AgentState:
    return AgentState.model_validate_json(path.read_text(encoding="utf-8"))


def _load_steps_for_state(state: AgentState, root_dir: str | Path) -> list[StepLog | PreStepFailureLog]:
    try:
        return load_run_replay(state.run_id, root_dir=root_dir)
    except FileNotFoundError:
        return []


def _average(values: list[float | int]) -> float | None:
    filtered = [float(value) for value in values]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _format_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["  (none)"]
    return [f"  {name}: {count}" for name, count in counter.most_common()]


def _retry_log_attempts(path: str | None) -> int:
    if not path:
        return 0
    retry_path = Path(path)
    if not retry_path.exists():
        return 0
    attempts = [line for line in retry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(attempts) - 1)


def _load_selector_traces(path: str | None) -> list[SelectorTrace]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    return [SelectorTrace.model_validate(item) for item in payload]


def _entry_failure_categories(entry: StepLog) -> list[FailureCategory]:
    categories: list[FailureCategory] = []
    if entry.executed_action.failure_category is not None:
        categories.append(entry.executed_action.failure_category)
    if entry.verification_result.failure_category is not None:
        categories.append(entry.verification_result.failure_category)
    if entry.recovery_decision.failure_category is not None:
        categories.append(entry.recovery_decision.failure_category)
    if entry.failure is not None:
        categories.append(entry.failure.category)
    if entry.executed_action.execution_trace is not None:
        for attempt in entry.executed_action.execution_trace.attempts:
            if attempt.failure_category is not None:
                categories.append(attempt.failure_category)
    return categories


def _execution_no_progress_events(entry: StepLog) -> int:
    if entry.executed_action.execution_trace is None:
        return 0
    return sum(1 for attempt in entry.executed_action.execution_trace.attempts if attempt.no_progress_detected)


def _state_level_failure_category(state: AgentState, entries: list[StepLog | PreStepFailureLog]) -> FailureCategory | None:
    if state.status is RunStatus.SUCCEEDED:
        return None
    for entry in reversed(entries):
        if isinstance(entry, PreStepFailureLog):
            return entry.failure.category
        if entry.failure is not None:
            return entry.failure.category
        if entry.recovery_decision.failure_category is not None:
            return entry.recovery_decision.failure_category
        if entry.verification_result.failure_category is not None:
            return entry.verification_result.failure_category
        if entry.executed_action.failure_category is not None:
            return entry.executed_action.failure_category
    if state.stop_reason is not None and state.stop_reason.value in FailureCategory._value2member_map_:
        return FailureCategory(state.stop_reason.value)
    return None


def _default_task_spec_for_state(state: AgentState) -> BenchmarkTaskSpec:
    page_url = state.start_url or "unknown"
    lowered = state.intent.lower()
    if "gmail" in lowered:
        return BenchmarkTaskSpec(
            task_id=state.run_id,
            page_url=page_url if page_url != "unknown" else "https://mail.google.com/",
            task_type=BenchmarkTaskType.MULTI_STEP_FORM,
            intent=state.intent,
            expected_completion_signal="draft created or intentional stop",
            difficulty_tags=["multi_step"],
        )
    if "form" in lowered and ("submit" in lowered or "fill" in lowered or "complete" in lowered):
        return BenchmarkTaskSpec(
            task_id=state.run_id,
            page_url=page_url if page_url != "unknown" else "https://practice-automation.com/form-fields/",
            task_type=BenchmarkTaskType.FORM_SUBMIT,
            intent=state.intent,
            expected_completion_signal="form success",
            difficulty_tags=["single_page"],
        )
    return BenchmarkTaskSpec(
        task_id=state.run_id,
        page_url=page_url,
        task_type=BenchmarkTaskType.FORM_SUBMIT,
        intent=state.intent,
        expected_completion_signal="task completed",
        difficulty_tags=[],
    )


def _success_rate_by_task_type(run_metrics: list[RunMetrics]) -> dict[str, float]:
    by_type: dict[str, list[RunMetrics]] = defaultdict(list)
    for metrics in run_metrics:
        by_type[metrics.task_type.value].append(metrics)
    return {
        task_type: (sum(1 for metrics in grouped if metrics.success) / len(grouped) * 100.0) if grouped else 0.0
        for task_type, grouped in by_type.items()
    }


def _tag_summary(run_metrics: list[RunMetrics]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for metrics in run_metrics:
        for tag in metrics.tags:
            grouped[tag].append(metrics)
    summary: dict[str, dict[str, float | int]] = {}
    for tag, metrics_list in grouped.items():
        total = len(metrics_list)
        summary[tag] = {
            "run_count": total,
            "success_rate": (sum(1 for metrics in metrics_list if metrics.success) / total * 100.0) if total else 0.0,
            "average_step_count": _average([metrics.step_count for metrics in metrics_list]) or 0.0,
        }
    return summary


def _summary_suite_id(target: str) -> str:
    target_path = Path(target)
    if target_path.exists() and target_path.is_dir():
        return target_path.name
    return target


def _average_total_retries(run_metrics: list[RunMetrics]) -> float:
    if not run_metrics:
        return 0.0
    return sum(
        metrics.perception_retry_count + metrics.selector_recovery_count + metrics.execution_retry_count
        for metrics in run_metrics
    ) / len(run_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one local benchmark run or a directory of runs")
    parser.add_argument("target", help="Run id under runs/<run_id> or a directory containing run folders")
    parser.add_argument("--root-dir", default="runs", help="Root directory containing run folders")
    args = parser.parse_args()
    print(summarize_runs(args.target, root_dir=args.root_dir))


if __name__ == "__main__":
    main()
