"""Phase 5 tests for the unified benchmark harness."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.core.contracts.critic import FailureType
from src.core.contracts.perception import Environment as UnifiedEnvironment
from src.models.common import RunResponse, RunStatus
from src.runtime.benchmark_runner import (
    Phase5BenchmarkTask,
    run_phase5_benchmark_suite,
)
from src.runtime.state import AgentRuntimeState


class _FakeLoop:
    def __init__(self, *, status: RunStatus, retry_count: int, failure_type: FailureType | None, environment: UnifiedEnvironment) -> None:
        self._status = status
        self._retry_count = retry_count
        self._failure_type = failure_type
        self._environment = environment
        self._run_id = f"run-{uuid4().hex[:8]}"
        self._state = AgentRuntimeState(environment=environment, retry_count=retry_count, last_failure_type=failure_type)

    async def start_run(self, request) -> RunResponse:
        return RunResponse(run_id=self._run_id, status=RunStatus.PENDING, intent=request.intent, step_count=0)

    async def step_run(self, request) -> RunResponse:
        return RunResponse(run_id=request.run_id, status=self._status, intent="task", step_count=1)

    def unified_state_for_run(self, run_id: str) -> AgentRuntimeState | None:
        return self._state if run_id == self._run_id else None


@pytest.mark.asyncio
async def test_phase5_benchmark_suite_reports_metrics() -> None:
    async def build_browser():
        return _FakeLoop(
            status=RunStatus.SUCCEEDED,
            retry_count=1,
            failure_type=None,
            environment=UnifiedEnvironment.BROWSER,
        )

    async def build_desktop():
        return _FakeLoop(
            status=RunStatus.FAILED,
            retry_count=2,
            failure_type=FailureType.TARGET_NOT_FOUND,
            environment=UnifiedEnvironment.DESKTOP,
        )

    tasks = [
        Phase5BenchmarkTask(task_id="browser", environment="browser", intent="browser task"),
        Phase5BenchmarkTask(task_id="desktop", environment="desktop", intent="desktop task"),
        Phase5BenchmarkTask(task_id="cross", environment="cross_environment", intent="cross task"),
    ]

    summary = await run_phase5_benchmark_suite(
        tasks,
        browser_loop_builder=build_browser,
        desktop_loop_builder=build_desktop,
    )

    assert summary.total_runs == 3
    assert summary.success_rate >= 0.0
    assert summary.average_retry_count >= 1.0
    assert "target_not_found" in summary.failure_type_distribution
    assert len(summary.results) == 3


@pytest.mark.asyncio
async def test_cross_environment_benchmark_merges_retry_counts() -> None:
    async def build_browser():
        return _FakeLoop(
            status=RunStatus.SUCCEEDED,
            retry_count=1,
            failure_type=None,
            environment=UnifiedEnvironment.BROWSER,
        )

    async def build_desktop():
        return _FakeLoop(
            status=RunStatus.SUCCEEDED,
            retry_count=2,
            failure_type=None,
            environment=UnifiedEnvironment.DESKTOP,
        )

    tasks = [
        Phase5BenchmarkTask(task_id="cross", environment="cross_environment", intent="cross task"),
    ]

    summary = await run_phase5_benchmark_suite(
        tasks,
        browser_loop_builder=build_browser,
        desktop_loop_builder=build_desktop,
    )

    result = summary.results[0]
    assert result.success is True
    assert result.retry_count == 3
    assert len(result.run_ids) == 2
