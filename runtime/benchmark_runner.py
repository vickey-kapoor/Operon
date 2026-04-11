"""Phase 5 benchmark harness for the unified browser and desktop agent."""

from __future__ import annotations

import time
from collections import Counter
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from src.models.common import RunStatus


class Phase5BenchmarkTask(BaseModel):
    """One supported Phase 5 benchmark task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    environment: str = Field(pattern="^(browser|desktop|cross_environment)$")
    intent: str = Field(min_length=1)
    start_url: str | None = None
    max_steps: int = Field(default=6, ge=1, le=50)


class Phase5BenchmarkResult(BaseModel):
    """Compact benchmark result with measurable reliability fields."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    success: bool
    retry_count: int = Field(ge=0)
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    duration_seconds: float = Field(ge=0.0)
    run_ids: list[str] = Field(default_factory=list)


class Phase5BenchmarkSummary(BaseModel):
    """Aggregate benchmark summary across a small task suite."""

    model_config = ConfigDict(extra="forbid")

    total_runs: int = Field(ge=0)
    success_rate: float = Field(ge=0.0)
    average_retry_count: float = Field(ge=0.0)
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)
    average_task_duration: float = Field(ge=0.0)
    results: list[Phase5BenchmarkResult] = Field(default_factory=list)


DEFAULT_PHASE5_TASKS = [
    Phase5BenchmarkTask(
        task_id="browser_only_basic",
        environment="browser",
        intent="The browser has navigated to example.com. Observe that the page has loaded and verify the main heading is visible.",
        start_url="https://example.com",
        max_steps=4,
    ),
    Phase5BenchmarkTask(
        task_id="desktop_only_basic",
        environment="desktop",
        intent="Launch Notepad and wait for the window to settle.",
        max_steps=4,
    ),
    Phase5BenchmarkTask(
        task_id="cross_environment_upload",
        environment="cross_environment",
        intent="Open the upload page, then switch to the desktop file picker and confirm the upload flow.",
        start_url="https://example.com/upload",
        max_steps=4,
    ),
]


async def _default_browser_loop_builder():
    from src.api.routes import get_agent_loop

    return get_agent_loop()


async def _default_desktop_loop_builder():
    from src.api.routes import get_desktop_agent_loop

    return get_desktop_agent_loop()


async def _run_until_terminal(loop, run_id: str, max_steps: int):
    from src.models.common import StepRequest

    response = None
    for _ in range(max_steps):
        response = await loop.step_run(StepRequest(run_id=run_id))
        if response.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_FOR_USER}:
            return response
    return response


async def _run_single_environment_task(loop, task: Phase5BenchmarkTask):
    from src.models.common import RunTaskRequest

    started = await loop.start_run(RunTaskRequest(intent=task.intent, start_url=task.start_url))
    response = await _run_until_terminal(loop, started.run_id, task.max_steps)
    unified_state = loop.unified_state_for_run(started.run_id)
    failure_counter = Counter()
    if unified_state is not None and unified_state.last_failure_type is not None:
        failure_counter[unified_state.last_failure_type.value] += 1
    retry_count = unified_state.retry_count if unified_state is not None else 0
    success = response is not None and response.status is RunStatus.SUCCEEDED
    return success, retry_count, dict(failure_counter), [started.run_id]


async def run_phase5_benchmark_suite(
    tasks: list[Phase5BenchmarkTask] | None = None,
    *,
    browser_loop_builder: Callable[[], Awaitable[object]] | None = None,
    desktop_loop_builder: Callable[[], Awaitable[object]] | None = None,
) -> Phase5BenchmarkSummary:
    """Run the small Phase 5 benchmark suite through the unified architecture."""

    tasks = tasks or DEFAULT_PHASE5_TASKS
    browser_loop_builder = browser_loop_builder or _default_browser_loop_builder
    desktop_loop_builder = desktop_loop_builder or _default_desktop_loop_builder
    results: list[Phase5BenchmarkResult] = []

    for task in tasks:
        started_at = time.perf_counter()
        if task.environment == "browser":
            loop = await browser_loop_builder()
            success, retry_count, failures, run_ids = await _run_single_environment_task(loop, task)
        elif task.environment == "desktop":
            loop = await desktop_loop_builder()
            success, retry_count, failures, run_ids = await _run_single_environment_task(loop, task)
        else:
            browser_loop = await browser_loop_builder()
            desktop_loop = await desktop_loop_builder()
            browser_task = task.model_copy(update={"environment": "browser"})
            desktop_task = task.model_copy(update={"environment": "desktop", "start_url": None})
            browser_success, browser_retries, browser_failures, browser_run_ids = await _run_single_environment_task(
                browser_loop,
                browser_task,
            )
            desktop_success, desktop_retries, desktop_failures, desktop_run_ids = await _run_single_environment_task(
                desktop_loop,
                desktop_task,
            )
            merged_failures = Counter(browser_failures)
            merged_failures.update(desktop_failures)
            success = browser_success and desktop_success
            retry_count = browser_retries + desktop_retries
            failures = dict(merged_failures)
            run_ids = [*browser_run_ids, *desktop_run_ids]

        duration_seconds = time.perf_counter() - started_at
        results.append(
            Phase5BenchmarkResult(
                task_id=task.task_id,
                success=success,
                retry_count=retry_count,
                failure_type_distribution=failures,
                duration_seconds=duration_seconds,
                run_ids=run_ids,
            )
        )

    total_runs = len(results)
    success_count = sum(1 for result in results if result.success)
    failure_counter = Counter()
    for result in results:
        failure_counter.update(result.failure_type_distribution)
    average_retry_count = sum(result.retry_count for result in results) / total_runs if total_runs else 0.0
    average_task_duration = sum(result.duration_seconds for result in results) / total_runs if total_runs else 0.0
    return Phase5BenchmarkSummary(
        total_runs=total_runs,
        success_rate=(success_count / total_runs * 100.0) if total_runs else 0.0,
        average_retry_count=average_retry_count,
        failure_type_distribution=dict(failure_counter),
        average_task_duration=average_task_duration,
        results=results,
    )
