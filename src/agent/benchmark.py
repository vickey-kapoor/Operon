"""Minimal local entry points for the active form benchmark."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.agent.capture import ScreenCaptureService
from src.agent.loop import AgentLoop
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.agent.video_verifier import VideoVerifier
from src.clients.gemini import GeminiHttpClient
from src.executor.desktop import DesktopExecutor
from src.models.benchmark import (
    BenchmarkSuiteSpec,
    BenchmarkSuiteSummary,
    BenchmarkTaskSpec,
    BenchmarkTaskType,
    RunMetrics,
)
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore
from src.store.summary import (
    generate_run_metrics,
    generate_suite_summary,
    write_run_metrics,
    write_suite_summary,
)

DEFAULT_FORM_BENCHMARK_INTENT = "Complete the auth-free form and submit it successfully."
DEFAULT_FORM_BENCHMARK_URL = "https://practice-automation.com/form-fields/"

DEFAULT_BENCHMARK_SUITE = BenchmarkSuiteSpec(
    suite_id="operon_v1_default_suite",
    tasks=[
        BenchmarkTaskSpec(
            task_id="practice_form_submit",
            page_url=DEFAULT_FORM_BENCHMARK_URL,
            task_type=BenchmarkTaskType.FORM_SUBMIT,
            intent=DEFAULT_FORM_BENCHMARK_INTENT,
            expected_completion_signal="form success",
            difficulty_tags=["single_page", "placeholder_heavy"],
        ),
    ],
)


class BenchmarkSuiteResult:
    """Compact return object for suite execution plus persisted metrics artifacts."""

    def __init__(
        self,
        *,
        suite_summary: BenchmarkSuiteSummary,
        run_metrics: list[RunMetrics],
        suite_summary_path: Path,
        run_metrics_paths: list[Path],
    ) -> None:
        self.suite_summary = suite_summary
        self.run_metrics = run_metrics
        self.suite_summary_path = suite_summary_path
        self.run_metrics_paths = run_metrics_paths


def _build_loop(*, root_dir: str | Path = "runs") -> tuple[AgentLoop, DesktopExecutor]:
    gemini_client = GeminiHttpClient()
    executor = DesktopExecutor()
    run_store = FileBackedRunStore(root_dir=root_dir)
    memory_store = FileBackedMemoryStore(root_dir=root_dir)
    perception_service = GeminiPerceptionService(gemini_client=gemini_client)
    video_verifier = VideoVerifier(gemini_client)
    policy_service = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=gemini_client),
        memory_store=memory_store,
        element_buffer=perception_service.element_buffer,
    )
    loop = AgentLoop(
        capture_service=ScreenCaptureService(executor=executor),
        perception_service=perception_service,
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=DeterministicVerifierService(
            gemini_client=gemini_client,
            video_verifier=video_verifier,
        ),
        recovery_manager=RuleBasedRecoveryManager(),
        memory_store=memory_store,
    )
    return loop, executor

async def run_form_benchmark(max_steps: int = 12, *, root_dir: str | Path = "runs"):
    """Run the browser-only auth-free form benchmark until a terminal condition is reached."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    loop, executor = _build_loop(root_dir=root_dir)
    task_spec = DEFAULT_BENCHMARK_SUITE.tasks[0]
    try:
        response = await loop.run_live_benchmark(
            intent=task_spec.intent,
            benchmark_url=os.getenv("FORM_BENCHMARK_URL", task_spec.page_url),
            max_steps=max_steps,
        )
        metrics = generate_run_metrics(response.run_id, root_dir=root_dir, task_spec=task_spec)
        write_run_metrics(metrics, root_dir=root_dir)
        return response
    finally:
        await executor.close()


async def run_benchmark_suite(
    suite_spec: BenchmarkSuiteSpec = DEFAULT_BENCHMARK_SUITE,
    *,
    max_steps: int = 12,
    root_dir: str | Path = "runs",
) -> BenchmarkSuiteResult:
    """Run a sequence of benchmark tasks and persist per-run plus aggregate metrics."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    run_metrics: list[RunMetrics] = []
    run_metrics_paths: list[Path] = []

    for task in suite_spec.tasks:
        loop, executor = _build_loop(root_dir=root_dir)
        try:
            response = await loop.run_live_benchmark(
                intent=task.intent,
                benchmark_url=_task_url(task),
                max_steps=max_steps,
            )
            metrics = generate_run_metrics(response.run_id, root_dir=root_dir, task_spec=task)
            run_metrics.append(metrics)
            run_metrics_paths.append(write_run_metrics(metrics, root_dir=root_dir))
        finally:
            await executor.close()

    suite_summary = generate_suite_summary(run_metrics, suite_id=suite_spec.suite_id)
    suite_summary_path = write_suite_summary(
        suite_summary,
        output_path=Path(root_dir) / "benchmark_suite_summary.json",
    )
    (Path(root_dir) / "benchmark_suite_summary.md").write_text(
        _render_suite_markdown(suite_summary),
        encoding="utf-8",
    )
    return BenchmarkSuiteResult(
        suite_summary=suite_summary,
        run_metrics=run_metrics,
        suite_summary_path=suite_summary_path,
        run_metrics_paths=run_metrics_paths,
    )


def _task_url(task: BenchmarkTaskSpec) -> str:
    if task.page_url == DEFAULT_FORM_BENCHMARK_URL:
        return os.getenv("FORM_BENCHMARK_URL", task.page_url)
    return task.page_url


def _render_suite_markdown(summary: BenchmarkSuiteSummary) -> str:
    lines = [
        f"# Benchmark Suite: {summary.suite_id}",
        "",
        f"- Total runs: {summary.total_runs}",
        f"- Success rate: {summary.overall_success_rate:.2f}%",
        f"- Average step count: {summary.average_step_count:.2f}",
        "",
        "## Failure Breakdown By Stop Reason",
    ]
    if not summary.failure_breakdown_by_stop_reason:
        lines.append("- none")
    else:
        lines.extend(f"- {name}: {count}" for name, count in summary.failure_breakdown_by_stop_reason.items())
    lines.append("")
    lines.append("## Success Rate By Task Type")
    if not summary.success_rate_by_task_type:
        lines.append("- none")
    else:
        lines.extend(f"- {name}: {rate:.2f}%" for name, rate in summary.success_rate_by_task_type.items())
    return "\n".join(lines)


def compute_trajectory_drift(
    run_ids: list[str],
    successful_run_ids: set[str],
    *,
    root_dir: str | Path = "runs",
) -> float | None:
    """Compute mean pixel distance between click coordinates in successful vs failed runs.

    For each step index that appears in both a failed and a successful run, measures
    the Euclidean distance between the (x, y) coordinates used.  The mean of all
    such distances is the 'Trajectory Drift' — a measure of how far the agent's
    click targets drifted on runs that failed compared to runs that succeeded.

    A drift near 0 px means failures were not caused by coordinate error.
    A large drift signals the agent targeted substantially different positions on
    failing runs, which usually indicates a perception or anchor issue.

    Returns None when there are no valid coordinate pairs to compare (e.g. all
    runs either succeeded or all failed, or run artifacts are missing).
    """
    root = Path(root_dir)
    # step_index → list of (x, y)
    success_coords: dict[int, list[tuple[float, float]]] = {}
    failed_coords: dict[int, list[tuple[float, float]]] = {}

    for run_id in run_ids:
        log_path = root / run_id / "run.jsonl"
        if not log_path.exists():
            continue
        bucket = success_coords if run_id in successful_run_ids else failed_coords
        try:
            for raw_line in log_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                step = __import__("json").loads(raw_line)
                action = step.get("policy_decision", {}).get("action", {})
                if action.get("action_type") not in ("click", "type"):
                    continue
                x, y = action.get("x"), action.get("y")
                if x is None or y is None:
                    continue
                idx = int(step.get("step_index", 0))
                bucket.setdefault(idx, []).append((float(x), float(y)))
        except Exception as exc:
            _log.debug("trajectory_drift: skipping run %s: %s", run_id, exc)

    if not success_coords or not failed_coords:
        return None

    distances: list[float] = []
    for step_idx, fail_pts in failed_coords.items():
        # Accept exact step match or one step off to handle minor step-count divergence
        success_pts: list[tuple[float, float]] | None = None
        for offset in (0, 1, -1, 2, -2):
            if step_idx + offset in success_coords:
                success_pts = success_coords[step_idx + offset]
                break
        if not success_pts:
            continue
        for fx, fy in fail_pts:
            min_d = min(math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2) for sx, sy in success_pts)
            distances.append(min_d)

    if not distances:
        return None
    drift = round(sum(distances) / len(distances), 2)
    _log.info("trajectory_drift: %.2f px across %d coord pairs", drift, len(distances))
    return drift


async def run_stress_benchmark(
    suite_spec: BenchmarkSuiteSpec = DEFAULT_BENCHMARK_SUITE,
    *,
    k: int = 3,
    max_steps: int = 12,
    root_dir: str | Path = "runs",
) -> None:
    """Run each task in the suite k times and log trajectory drift + reliability scores.

    Uses StressRunner from benchmark_suite for the k-repetition and window
    randomisation logic.  After each task, the trajectory drift is computed from
    the run artifacts and logged.  The PostRunReflector is called for every
    successful run but only saves the Golden Path episode when the task achieved
    100% reliability (all k repetitions passed).
    """
    from src.api.benchmark_suite import StressRunner

    load_dotenv(find_dotenv(usecwd=True), override=False)
    root = Path(root_dir)
    runner = StressRunner(k=k)

    # Build a task_data dict list compatible with StressRunner from BenchmarkTaskSpec
    tasks_data = [
        {
            "task_id": t.task_id,
            "intent": t.intent,
            "start_url": t.page_url,
            "difficulty": "custom",
            "category": "form",
            "site": "practice-automation",
            "optimal_steps": 12,
        }
        for t in suite_spec.tasks
    ]

    from src.agent.loop import AgentLoop as _AgentLoop  # noqa: F401 — import check

    _current_loop: tuple | None = None

    def _get_loop_fn():
        nonlocal _current_loop
        _current_loop = _build_loop(root_dir=root)
        return _current_loop[0]

    stress_result = await runner.run_suite(
        tasks_data,
        max_steps=max_steps,
        get_loop_fn=_get_loop_fn,
        root_dir=root,
    )

    # Compute drift and call reflector for each task
    from src.agent.reflector import PostRunReflector
    from src.store.memory import FileBackedMemoryStore

    memory_store = FileBackedMemoryStore(root_dir=root)
    reflector = PostRunReflector(memory_store=memory_store, root_dir=root)

    for task_result in stress_result.task_results:
        reliability = task_result.reliability_score
        drift = stress_result.trajectory_drift_px.get(task_result.task_id)
        _log.info(
            "stress_summary task=%s reliability=%.2f drift_px=%s",
            task_result.task_id,
            reliability,
            f"{drift:.1f}" if drift is not None else "n/a",
        )
        # Reflect on every successful run; episode only saved when reliability == 1.0
        for attempt in task_result.attempts:
            if attempt.run_id and attempt.succeeded:
                try:
                    reflector.reflect(attempt.run_id, reliability_score=reliability)
                except Exception as exc:
                    _log.warning("reflector failed for %s: %s", attempt.run_id, exc)

    # Persist stress result JSON
    result_path = root / "stress_result.json"
    result_path.write_text(
        __import__("json").dumps(stress_result.to_dict(), indent=2),
        encoding="utf-8",
    )
    _log.info(
        "stress complete: overall_reliability=%.4f saved to %s",
        stress_result.overall_reliability_score,
        result_path,
    )


_log = logging.getLogger(__name__)


if __name__ == "__main__":
    _log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, _log_level, logging.INFO), format="%(name)s %(levelname)s %(message)s")
    result = asyncio.run(run_form_benchmark())
    print(result.model_dump_json(indent=2))
