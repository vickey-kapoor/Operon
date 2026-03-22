"""Minimal local entry points for the active form benchmark and optional Gmail benchmark."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.agent.capture import BrowserCaptureService
from src.agent.loop import AgentLoop
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import GeminiHttpClient
from src.executor.browser import PlaywrightBrowserExecutor
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
DEFAULT_GMAIL_BENCHMARK_INTENT = "Create a Gmail draft and stop before send."
DEFAULT_GMAIL_BENCHMARK_URL = "https://mail.google.com/"

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
        BenchmarkTaskSpec(
            task_id="gmail_draft_authenticated",
            page_url=DEFAULT_GMAIL_BENCHMARK_URL,
            task_type=BenchmarkTaskType.MULTI_STEP_FORM,
            intent=DEFAULT_GMAIL_BENCHMARK_INTENT,
            expected_completion_signal="draft created or intentional stop",
            difficulty_tags=["multi_step", "dynamic_update"],
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


def _build_loop(*, root_dir: str | Path = "runs") -> tuple[AgentLoop, PlaywrightBrowserExecutor]:
    gemini_client = GeminiHttpClient()
    executor = PlaywrightBrowserExecutor()
    run_store = FileBackedRunStore(root_dir=root_dir)
    memory_store = FileBackedMemoryStore(root_dir=root_dir)
    policy_service = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=gemini_client),
        memory_store=memory_store,
    )
    loop = AgentLoop(
        capture_service=BrowserCaptureService(executor=executor),
        perception_service=GeminiPerceptionService(gemini_client=gemini_client),
        run_store=run_store,
        policy_service=policy_service,
        executor=executor,
        verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
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


async def run_gmail_draft_benchmark(max_steps: int = 12, *, root_dir: str | Path = "runs"):
    """Run the optional Gmail draft benchmark until a terminal condition is reached."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    loop, executor = _build_loop(root_dir=root_dir)
    task_spec = DEFAULT_BENCHMARK_SUITE.tasks[1]
    try:
        response = await loop.run_live_benchmark(
            intent=task_spec.intent,
            benchmark_url=os.getenv("GMAIL_BENCHMARK_URL", task_spec.page_url),
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
    if task.page_url == DEFAULT_GMAIL_BENCHMARK_URL:
        return os.getenv("GMAIL_BENCHMARK_URL", task.page_url)
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


if __name__ == "__main__":
    result = asyncio.run(run_form_benchmark())
    print(result.model_dump_json(indent=2))
