"""Local benchmark summary loader and CLI."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.models.common import FailureCategory, LoopStage, RunStatus, StopReason
from src.models.state import AgentState
from src.store.replay import load_run_replay


def summarize_runs(target: str, root_dir: str | Path = "runs") -> str:
    """Summarize one run id or a directory of local runs."""
    states = _load_states(target, root_dir=root_dir)
    total_runs = len(states)
    success_count = sum(1 for state in states if state.status is RunStatus.SUCCEEDED)
    failure_count = sum(1 for state in states if state.status is RunStatus.FAILED)
    success_rate = (success_count / total_runs * 100.0) if total_runs else 0.0
    average_steps = (sum(state.step_count for state in states) / total_runs) if total_runs else 0.0
    average_retries = (sum(sum(state.retry_counts.values()) for state in states) / total_runs) if total_runs else 0.0

    stop_reasons = Counter(state.stop_reason.value for state in states if state.stop_reason is not None)
    failure_categories: Counter[str] = Counter()
    failing_stages: Counter[str] = Counter()

    for state in states:
        for step in _load_steps_for_state(state, root_dir=root_dir):
            if step.failure is not None:
                failure_categories[step.failure.category.value] += 1
                failing_stages[step.failure.stage.value] += 1
        if state.stop_reason is StopReason.MAX_STEP_LIMIT_REACHED:
            failure_categories[FailureCategory.MAX_STEP_LIMIT_REACHED.value] += 1
            failing_stages[LoopStage.ORCHESTRATE.value] += 1

    lines = [
        f"total_runs: {total_runs}",
        f"success_count: {success_count}",
        f"failure_count: {failure_count}",
        f"success_rate: {success_rate:.2f}%",
        f"average_steps_per_run: {average_steps:.2f}",
        f"average_retries_per_run: {average_retries:.2f}",
        "stop_reason_counts:",
    ]
    lines.extend(_format_counter(stop_reasons))
    lines.append("most_common_failure_categories:")
    lines.extend(_format_counter(failure_categories))
    lines.append("most_common_failing_stages:")
    lines.extend(_format_counter(failing_stages))
    return "\n".join(lines)



def _load_states(target: str, root_dir: str | Path) -> list[AgentState]:
    root = Path(root_dir)
    target_path = Path(target)
    if target_path.exists() and target_path.is_dir():
        run_dirs = sorted(path for path in target_path.iterdir() if path.is_dir())
        return [_load_state_from_path(run_dir / "state.json") for run_dir in run_dirs if (run_dir / "state.json").exists()]

    state_path = root / target / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Run state not found: {state_path}")
    return [_load_state_from_path(state_path)]



def _load_state_from_path(path: Path) -> AgentState:
    return AgentState.model_validate_json(path.read_text(encoding="utf-8"))



def _load_steps_for_state(state: AgentState, root_dir: str | Path) -> list:
    try:
        return load_run_replay(state.run_id, root_dir=root_dir)
    except FileNotFoundError:
        return []



def _format_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["  (none)"]
    return [f"  {name}: {count}" for name, count in counter.most_common()]



def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one local benchmark run or a directory of runs")
    parser.add_argument("target", help="Run id under runs/<run_id> or a directory containing run folders")
    parser.add_argument("--root-dir", default="runs", help="Root directory containing run folders")
    args = parser.parse_args()
    print(summarize_runs(args.target, root_dir=args.root_dir))


if __name__ == "__main__":
    main()
