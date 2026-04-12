"""Unit tests for the native-upload reliability benchmark."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from core.contracts.critic import FailureType
from core.contracts.perception import Environment as UnifiedEnvironment
from evaluation.benchmark_native_upload import (
    BenchmarkRunResult,
    BenchmarkSummary,
    run_benchmark,
)
from runtime.state import AgentRuntimeState
from src.models.common import RunResponse, RunStatus

# ---------------------------------------------------------------------------
# Fake loop helpers
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Minimal loop stub that satisfies the run_benchmark contract."""

    def __init__(
        self,
        *,
        status: RunStatus,
        retry_count: int,
        failure_type: FailureType | None,
    ) -> None:
        self._status = status
        self._retry_count = retry_count
        self._failure_type = failure_type
        self._run_id = f"run-{uuid4().hex[:8]}"
        self._state = AgentRuntimeState(
            environment=UnifiedEnvironment.BROWSER,
            retry_count=retry_count,
            last_failure_type=failure_type,
        )

    async def start_run(self, request) -> RunResponse:
        return RunResponse(
            run_id=self._run_id,
            status=RunStatus.PENDING,
            intent=request.intent,
            step_count=0,
        )

    async def run_live_benchmark(
        self,
        intent: str,
        *,
        benchmark_url: str,
        max_steps: int,
    ) -> RunResponse:
        return RunResponse(
            run_id=self._run_id,
            status=self._status,
            intent=intent,
            step_count=1,
        )

    def unified_state_for_run(self, run_id: str) -> AgentRuntimeState | None:
        return self._state if run_id == self._run_id else None


class _CyclingFakeLoop:
    """Loop stub that cycles through a list of (status, failure_type) pairs."""

    def __init__(self, outcomes: list[tuple[RunStatus, FailureType | None]]) -> None:
        self._outcomes = outcomes
        self._index = 0

    async def run_live_benchmark(
        self,
        intent: str,
        *,
        benchmark_url: str,
        max_steps: int,
    ) -> RunResponse:
        status, failure_type = self._outcomes[self._index % len(self._outcomes)]
        self._current_run_id = f"run-{uuid4().hex[:8]}"
        self._current_failure_type = failure_type
        self._current_status = status
        self._index += 1
        return RunResponse(
            run_id=self._current_run_id,
            status=status,
            intent=intent,
            step_count=1,
        )

    def unified_state_for_run(self, run_id: str) -> AgentRuntimeState | None:
        return AgentRuntimeState(
            environment=UnifiedEnvironment.BROWSER,
            retry_count=0,
            last_failure_type=self._current_failure_type,
        )


MINIMAL_TASK = {
    "intent": "Do something",
    "start_url": "https://example.com",
    "max_steps": 2,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_benchmark_run_result_model() -> None:
    """BenchmarkRunResult can be instantiated with valid fields."""
    result = BenchmarkRunResult(
        run_id="run-abc123",
        status="success",
        retries=0,
        duration_seconds=3.14,
        failure_type=None,
    )
    assert result.run_id == "run-abc123"
    assert result.status == "success"
    assert result.retries == 0
    assert result.failure_type is None


def test_benchmark_summary_model() -> None:
    """BenchmarkSummary can be instantiated with valid fields."""
    summary = BenchmarkSummary(
        task="upload_file_native",
        total_runs=5,
        success_rate=0.8,
        avg_retries=1.0,
        avg_duration_seconds=12.5,
        failure_distribution={"target_not_found": 1},
        runs=[
            BenchmarkRunResult(
                run_id="run-001",
                status="success",
                retries=0,
                duration_seconds=10.0,
                failure_type=None,
            )
        ],
    )
    assert summary.total_runs == 5
    assert summary.success_rate == pytest.approx(0.8)
    assert summary.failure_distribution["target_not_found"] == 1


@pytest.mark.asyncio
async def test_run_benchmark_success(tmp_path: Path) -> None:
    """All-success loop produces success_rate=1.0 and avg_retries=0."""
    fake = _FakeLoop(status=RunStatus.SUCCEEDED, retry_count=0, failure_type=None)

    async def _builder():
        return fake

    summary = await run_benchmark(
        n_runs=3,
        task_config=MINIMAL_TASK,
        loop_builder=_builder,
        output_path=tmp_path / "out.json",
        runs_dir=tmp_path / "runs",
        task_label="upload_file_native",
    )

    assert summary.total_runs == 3
    assert summary.success_rate == pytest.approx(1.0)
    assert summary.avg_retries == pytest.approx(0.0)
    assert summary.failure_distribution == {}
    assert all(r.status == "success" for r in summary.runs)


@pytest.mark.asyncio
async def test_run_benchmark_failure(tmp_path: Path) -> None:
    """All-failure loop produces success_rate=0.0."""
    fake = _FakeLoop(
        status=RunStatus.FAILED,
        retry_count=2,
        failure_type=FailureType.TARGET_NOT_FOUND,
    )

    async def _builder():
        return fake

    summary = await run_benchmark(
        n_runs=4,
        task_config=MINIMAL_TASK,
        loop_builder=_builder,
        output_path=tmp_path / "out.json",
        runs_dir=tmp_path / "runs",
        task_label="upload_file_native",
    )

    assert summary.total_runs == 4
    assert summary.success_rate == pytest.approx(0.0)
    assert all(r.status == "failure" for r in summary.runs)


@pytest.mark.asyncio
async def test_run_benchmark_failure_distribution(tmp_path: Path) -> None:
    """Mixed outcomes with distinct failure types are tallied correctly."""
    outcomes = [
        (RunStatus.FAILED, FailureType.TARGET_NOT_FOUND),
        (RunStatus.FAILED, FailureType.PICKER_NOT_DETECTED),
        (RunStatus.SUCCEEDED, None),
        (RunStatus.FAILED, FailureType.TARGET_NOT_FOUND),
    ]
    cycling = _CyclingFakeLoop(outcomes)

    async def _builder():
        return cycling

    summary = await run_benchmark(
        n_runs=4,
        task_config=MINIMAL_TASK,
        loop_builder=_builder,
        output_path=tmp_path / "out.json",
        runs_dir=tmp_path / "runs",
        task_label="upload_file_native",
    )

    assert summary.total_runs == 4
    assert summary.success_rate == pytest.approx(0.25)
    assert summary.failure_distribution.get("target_not_found") == 2
    assert summary.failure_distribution.get("picker_not_detected") == 1


@pytest.mark.asyncio
async def test_per_run_summary_json_written(tmp_path: Path) -> None:
    """run_benchmark writes a summary.json under runs/<run_id>/ for each run."""
    fake = _FakeLoop(status=RunStatus.SUCCEEDED, retry_count=0, failure_type=None)

    async def _builder():
        return fake

    runs_dir = tmp_path / "runs"
    await run_benchmark(
        n_runs=1,
        task_config=MINIMAL_TASK,
        loop_builder=_builder,
        output_path=tmp_path / "summary.json",
        runs_dir=runs_dir,
        task_label="upload_file_native",
    )

    # There should be exactly one run sub-directory
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run dir, found: {run_dirs}"

    summary_file = run_dirs[0] / "summary.json"
    assert summary_file.exists(), "summary.json was not written"

    import json

    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "run_id" in data
    assert "final_status" in data
    assert data["final_status"] == "succeeded"
    assert "retries" in data
    assert "duration_seconds" in data
