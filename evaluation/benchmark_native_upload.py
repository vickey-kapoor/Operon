"""Reliability benchmark for the native-upload flow.

Runs the agent N times against the file-uploader test page and aggregates
success-rate, retry counts, duration, and failure distribution.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from src.models.common import RunStatus

_UPLOAD_FILE_PATH = r"C:\tmp\test_upload.txt"

DEFAULT_TASK: dict = {
    "intent": (
        f"Upload {_UPLOAD_FILE_PATH!r} via the file uploader. "
        "Use upload_file_native (not upload_file) on the 'Add files' button to open the "
        "real OS file picker. Type the path in the dialog and press Enter. "
        "After the picker closes, click 'Start upload' and stop when upload succeeds."
    ),
    "start_url": "https://www.realuploader.com/pages/examples/minimal.php",
    "max_steps": 15,
}


class BenchmarkRunResult(BaseModel):
    """Per-run result captured after a single benchmark execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str  # "success" | "failure"
    retries: int
    duration_seconds: float
    failure_type: str | None  # last_failure_type.value or None


class BenchmarkSummary(BaseModel):
    """Aggregated summary across all runs for one benchmark task."""

    model_config = ConfigDict(extra="forbid")

    task: str
    total_runs: int
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_retries: float
    avg_duration_seconds: float
    failure_distribution: dict[str, int] = Field(default_factory=dict)
    runs: list[BenchmarkRunResult] = Field(default_factory=list)


async def run_benchmark(
    n_runs: int,
    task_config: dict,
    loop_builder: Callable[[], Awaitable[object]],
    output_path: Path,
    *,
    runs_dir: Path = Path("runs"),
    task_label: str = "upload_file_native",
) -> BenchmarkSummary:
    """Execute the benchmark N times and return an aggregated summary.

    Parameters
    ----------
    n_runs:
        Number of independent agent runs to execute.
    task_config:
        Dict with keys ``intent``, ``start_url``, and ``max_steps``.
    loop_builder:
        Async callable that returns an ``AgentLoop``-compatible object.
        Injectable for testing.
    output_path:
        Where the full JSON summary should be written.
    runs_dir:
        Root directory for per-run ``summary.json`` files (injectable for
        tests; defaults to ``runs/``).
    task_label:
        Human-readable task name embedded in the summary.
    """
    loop = await loop_builder()

    run_results: list[BenchmarkRunResult] = []
    failure_dist: dict[str, int] = {}

    for _ in range(n_runs):
        started_at = time.perf_counter()

        response = await loop.run_live_benchmark(
            task_config["intent"],
            benchmark_url=task_config["start_url"],
            max_steps=task_config["max_steps"],
        )

        duration = time.perf_counter() - started_at
        run_id = response.run_id
        succeeded = response.status is RunStatus.SUCCEEDED

        # Read retry / failure info from unified state
        unified_state = loop.unified_state_for_run(run_id)
        retries: int = unified_state.retry_count if unified_state is not None else 0
        failure_type_value: str | None = None
        if unified_state is not None and unified_state.last_failure_type is not None:
            failure_type_value = unified_state.last_failure_type.value

        result = BenchmarkRunResult(
            run_id=run_id,
            status="success" if succeeded else "failure",
            retries=retries,
            duration_seconds=round(duration, 3),
            failure_type=failure_type_value,
        )
        run_results.append(result)

        if failure_type_value:
            failure_dist[failure_type_value] = failure_dist.get(failure_type_value, 0) + 1

        # Write per-run summary.json
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        per_run_summary = {
            "run_id": run_id,
            "final_status": response.status.value,
            "retries": retries,
            "last_failure_type": failure_type_value,
            "duration_seconds": round(duration, 3),
            "last_action": None,
            "last_critic_result": None,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(per_run_summary, indent=2),
            encoding="utf-8",
        )

    total = len(run_results)
    success_count = sum(1 for r in run_results if r.status == "success")
    success_rate = success_count / total if total else 0.0
    avg_retries = sum(r.retries for r in run_results) / total if total else 0.0
    avg_duration = sum(r.duration_seconds for r in run_results) / total if total else 0.0

    summary = BenchmarkSummary(
        task=task_label,
        total_runs=total,
        success_rate=success_rate,
        avg_retries=avg_retries,
        avg_duration_seconds=round(avg_duration, 3),
        failure_distribution=failure_dist,
        runs=run_results,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.model_dump(), indent=2),
        encoding="utf-8",
    )

    return summary


def _print_summary_table(summary: BenchmarkSummary) -> None:
    """Print a human-readable summary table to stdout."""
    print(f"\n{'=' * 50}")
    print(f"Benchmark: {summary.task}")
    print(f"{'=' * 50}")
    print(f"  Total runs       : {summary.total_runs}")
    print(f"  Success rate     : {summary.success_rate:.1%}")
    print(f"  Avg retries      : {summary.avg_retries:.2f}")
    print(f"  Avg duration (s) : {summary.avg_duration_seconds:.2f}")
    if summary.failure_distribution:
        print("  Failure types    :")
        for ftype, count in sorted(summary.failure_distribution.items()):
            print(f"    {ftype}: {count}")
    print(f"{'=' * 50}\n")


async def main(args) -> None:
    """Entry point called from __main__ with parsed CLI args."""
    from src.api.routes import get_agent_loop

    async def _loop_builder():
        return get_agent_loop()

    output_path = Path(args.output)
    summary = await run_benchmark(
        n_runs=args.runs,
        task_config=DEFAULT_TASK,
        loop_builder=_loop_builder,
        output_path=output_path,
        task_label=args.task,
    )

    _print_summary_table(summary)
    print(f"Full results written to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Native-upload reliability benchmark")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--task", default="upload_file_native")
    parser.add_argument("--output", default="runs/benchmark_native_upload.json")
    _args = parser.parse_args()
    asyncio.run(main(_args))
