"""Local run replay loader and CLI for run.jsonl inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.logs import StepLog


def load_run_replay(run_id: str, root_dir: str | Path = "runs") -> list[StepLog]:
    """Load all step logs for a run from the local JSONL file."""
    log_path = Path(root_dir) / run_id / "run.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {log_path}")

    entries: list[StepLog] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(StepLog.model_validate_json(line))
    return entries


def render_run_replay(run_id: str, root_dir: str | Path = "runs") -> str:
    """Render a simple CLI-friendly replay summary for one run."""
    entries = load_run_replay(run_id, root_dir=root_dir)
    lines = [f"Run: {run_id}", f"Steps: {len(entries)}"]

    for entry in entries:
        lines.extend(
            [
                "",
                f"Step {entry.step_index}: {entry.step_id}",
                f"  before: {entry.before_artifact_path}",
                f"  after: {entry.after_artifact_path}",
                f"  perception_prompt: {entry.perception_debug.prompt_artifact_path}",
                f"  perception_raw: {entry.perception_debug.raw_response_artifact_path}",
                f"  perception_parsed: {entry.perception_debug.parsed_artifact_path}",
                f"  policy_prompt: {entry.policy_debug.prompt_artifact_path}",
                f"  policy_raw: {entry.policy_debug.raw_response_artifact_path}",
                f"  policy_decision: {entry.policy_debug.parsed_artifact_path}",
                f"  action: {entry.executed_action.action.action_type.value}",
                f"  verification: {entry.verification_result.status.value} | {entry.verification_result.reason}",
                f"  recovery: {entry.recovery_decision.strategy.value} | {entry.recovery_decision.message}",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Replay one stored run to stdout."""
    parser = argparse.ArgumentParser(description="Inspect a local run replay from run.jsonl")
    parser.add_argument("run_id", help="Run identifier under runs/<run_id>/run.jsonl")
    parser.add_argument("--root-dir", default="runs", help="Root directory containing run logs")
    args = parser.parse_args()
    print(render_run_replay(args.run_id, root_dir=args.root_dir))


if __name__ == "__main__":
    main()
