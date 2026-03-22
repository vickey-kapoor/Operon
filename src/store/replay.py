"""Local run replay loader and CLI for run.jsonl inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from src.models.logs import PreStepFailureLog, RunLogEntry, StepLog


def load_run_replay(run_id: str, root_dir: str | Path = "runs") -> list[RunLogEntry]:
    """Load all step logs for a run from the local JSONL file."""
    log_path = Path(root_dir) / run_id / "run.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {log_path}")

    entries: list[RunLogEntry] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(_parse_run_log_entry(line))
    return entries


def render_run_replay(run_id: str, root_dir: str | Path = "runs") -> str:
    """Render a simple CLI-friendly replay summary for one run."""
    entries = load_run_replay(run_id, root_dir=root_dir)
    completed_steps = [entry for entry in entries if isinstance(entry, StepLog)]
    pre_step_failures = [entry for entry in entries if isinstance(entry, PreStepFailureLog)]
    lines = [f"Run: {run_id}", f"Steps: {len(completed_steps)}"]
    if pre_step_failures:
        lines.append(f"Pre-step failures: {len(pre_step_failures)}")

    for entry in completed_steps:
        lines.extend(
            [
                "",
                f"Step {entry.step_index}: {entry.step_id}",
                f"  before: {entry.before_artifact_path}",
                f"  after: {entry.after_artifact_path}",
                f"  perception_prompt: {entry.perception_debug.prompt_artifact_path}",
                f"  perception_raw: {entry.perception_debug.raw_response_artifact_path}",
                f"  perception_parsed: {entry.perception_debug.parsed_artifact_path}",
                f"  perception_retry_log: {(entry.perception_debug.retry_log_artifact_path or 'none')}",
                f"  perception_selector_trace: {(entry.perception_debug.selector_trace_artifact_path or 'none')}",
                f"  policy_prompt: {entry.policy_debug.prompt_artifact_path}",
                f"  policy_raw: {entry.policy_debug.raw_response_artifact_path}",
                f"  policy_decision: {entry.policy_debug.parsed_artifact_path}",
                f"  policy_selector_trace: {(entry.policy_debug.selector_trace_artifact_path or 'none')}",
                f"  action: {entry.executed_action.action.action_type.value}",
                f"  execution_trace: {(entry.executed_action.execution_trace_artifact_path or 'none')}",
                f"  progress_trace: {(entry.progress_trace_artifact_path or 'none')}",
                f"  progress_no_progress_streak: {(entry.progress_state.no_progress_streak if entry.progress_state is not None else 'none')}",
                f"  progress_loop_detected: {(entry.progress_state.loop_detected if entry.progress_state is not None else 'none')}",
                f"  verification: {entry.verification_result.status.value} | {entry.verification_result.reason}",
                f"  verification_stop_reason: {(entry.verification_result.stop_reason.value if entry.verification_result.stop_reason is not None else 'none')}",
                f"  recovery: {entry.recovery_decision.strategy.value} | {entry.recovery_decision.message}",
                f"  recovery_stop_reason: {(entry.recovery_decision.stop_reason.value if entry.recovery_decision.stop_reason is not None else 'none')}",
            ]
        )

        if entry.failure is not None:
            lines.append(
                f"  failure: {entry.failure.category.value} | stop_reason={(entry.failure.stop_reason.value if entry.failure.stop_reason is not None else 'none')}"
            )

    for entry in pre_step_failures:
        lines.extend(
            [
                "",
                f"Pre-step failure {entry.step_index}: {entry.step_id}",
                f"  before: {entry.before_artifact_path}",
                f"  perception_prompt: {entry.perception_debug.prompt_artifact_path}",
                f"  perception_raw: {entry.perception_debug.raw_response_artifact_path}",
                f"  perception_parsed: {entry.perception_debug.parsed_artifact_path}",
                f"  perception_retry_log: {(entry.perception_debug.retry_log_artifact_path or 'none')}",
                f"  perception_selector_trace: {(entry.perception_debug.selector_trace_artifact_path or 'none')}",
                f"  failure_stage: {entry.failure.stage.value}",
                f"  failure_category: {entry.failure.category.value}",
                f"  failure_stop_reason: {(entry.failure.stop_reason.value if entry.failure.stop_reason is not None else 'none')}",
                f"  error: {entry.error_message}",
            ]
        )

    return "\n".join(lines)


def _parse_run_log_entry(payload: str) -> RunLogEntry:
    try:
        return StepLog.model_validate_json(payload)
    except ValidationError:
        return PreStepFailureLog.model_validate_json(payload)


def main() -> None:
    """Replay one stored run to stdout."""
    parser = argparse.ArgumentParser(description="Inspect a local run replay from run.jsonl")
    parser.add_argument("run_id", help="Run identifier under runs/<run_id>/run.jsonl")
    parser.add_argument("--root-dir", default="runs", help="Root directory containing run logs")
    args = parser.parse_args()
    print(render_run_replay(args.run_id, root_dir=args.root_dir))


if __name__ == "__main__":
    main()
