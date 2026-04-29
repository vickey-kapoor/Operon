"""Unit tests for the native-upload reliability benchmark."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.evaluation.benchmark_native_upload import (
    BenchmarkRunResult,
    BenchmarkSummary,
    run_benchmark,
)
from src.models.common import RunResponse, RunStatus, StopReason
from src.models.state import AgentState

# ---------------------------------------------------------------------------
# Fake loop helpers
# ---------------------------------------------------------------------------


class _FakeRunStore:
    """Minimal async run-store stub: maps run_id -> AgentState."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentState] = {}

    def put(self, state: AgentState) -> None:
        self._runs[state.run_id] = state

    async def get_run(self, run_id: str) -> AgentState | None:
        return self._runs.get(run_id)


def _make_agent_state(
    *,
    run_id: str,
    status: RunStatus,
    retries: int,
    stop_reason: StopReason | None,
) -> AgentState:
    retry_counts: dict[str, int] = {"subgoal:action_failed": retries} if retries else {}
    return AgentState(
        run_id=run_id,
        intent="test",
        status=status,
        retry_counts=retry_counts,
        stop_reason=stop_reason,
    )


class _FakeLoop:
    """Minimal loop stub that satisfies the run_benchmark contract."""

    def __init__(
        self,
        *,
        status: RunStatus,
        retry_count: int,
        stop_reason: StopReason | None,
    ) -> None:
        self._status = status
        self._run_id = f"run-{uuid4().hex[:8]}"
        self.run_store = _FakeRunStore()
        self.run_store.put(
            _make_agent_state(
                run_id=self._run_id,
                status=status,
                retries=retry_count,
                stop_reason=stop_reason,
            )
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


class _CyclingFakeLoop:
    """Loop stub that cycles through a list of (status, stop_reason) pairs."""

    def __init__(self, outcomes: list[tuple[RunStatus, StopReason | None]]) -> None:
        self._outcomes = outcomes
        self._index = 0
        self.run_store = _FakeRunStore()

    async def run_live_benchmark(
        self,
        intent: str,
        *,
        benchmark_url: str,
        max_steps: int,
    ) -> RunResponse:
        status, stop_reason = self._outcomes[self._index % len(self._outcomes)]
        run_id = f"run-{uuid4().hex[:8]}"
        self.run_store.put(
            _make_agent_state(
                run_id=run_id,
                status=status,
                retries=0,
                stop_reason=stop_reason,
            )
        )
        self._index += 1
        return RunResponse(
            run_id=run_id,
            status=status,
            intent=intent,
            step_count=1,
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
        failure_distribution={"no_meaningful_progress_across_steps": 1},
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
    assert summary.failure_distribution["no_meaningful_progress_across_steps"] == 1


@pytest.mark.asyncio
async def test_run_benchmark_success(tmp_path: Path) -> None:
    """All-success loop produces success_rate=1.0 and avg_retries=0."""
    fake = _FakeLoop(status=RunStatus.SUCCEEDED, retry_count=0, stop_reason=None)

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
        stop_reason=StopReason.RETRY_LIMIT_REACHED,
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
    """Mixed outcomes with distinct stop reasons are tallied correctly."""
    outcomes = [
        (RunStatus.FAILED, StopReason.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS),
        (RunStatus.FAILED, StopReason.MAX_STEP_LIMIT_REACHED),
        (RunStatus.SUCCEEDED, None),
        (RunStatus.FAILED, StopReason.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS),
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
    assert summary.failure_distribution.get("no_meaningful_progress_across_steps") == 2
    assert summary.failure_distribution.get("max_step_limit_reached") == 1


@pytest.mark.asyncio
async def test_per_run_summary_json_written(tmp_path: Path) -> None:
    """run_benchmark writes a summary.json under runs/<run_id>/ for each run."""
    fake = _FakeLoop(status=RunStatus.SUCCEEDED, retry_count=0, stop_reason=None)

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
