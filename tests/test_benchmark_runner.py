"""Focused tests for the bounded live benchmark runner."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agent.loop import AgentLoop
from src.models.benchmark import (
    BenchmarkSuiteSpec,
    BenchmarkTaskSpec,
    BenchmarkTaskType,
)
from src.models.common import FailureCategory, RunResponse, RunStatus, StopReason
from src.models.policy import ActionType
from src.models.state import AgentState


class StubRunStore:
    """Minimal in-memory run store for benchmark runner tests."""

    def __init__(self) -> None:
        self.state: AgentState | None = None

    def create_run(self, intent: str, *, start_url: str | None = None, headless: bool | None = None) -> AgentState:
        self.state = AgentState(run_id="run-benchmark", intent=intent, start_url=start_url, status=RunStatus.PENDING)
        return self.state

    async def get_run(self, run_id: str) -> AgentState | None:
        return self.state

    async def update_state(self, run_id: str, perception):
        raise NotImplementedError

    async def set_status(self, run_id: str, status: RunStatus) -> AgentState:
        assert self.state is not None
        self.state.status = status
        return self.state


@pytest.mark.asyncio
async def test_live_runner_stops_on_form_success() -> None:
    run_store = StubRunStore()
    executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    async def _step_run(request):
        assert run_store.state is not None
        run_store.state.step_count = 1
        run_store.state.status = RunStatus.SUCCEEDED
        run_store.state.stop_reason = StopReason.FORM_SUBMITTED_SUCCESS
        return RunResponse(
            run_id=run_store.state.run_id,
            status=run_store.state.status,
            intent=run_store.state.intent,
            step_count=run_store.state.step_count,
        )

    loop.step_run = AsyncMock(side_effect=_step_run)

    result = await loop.run_live_benchmark(max_steps=5)

    assert result.status is RunStatus.SUCCEEDED
    assert loop.step_run.await_count == 1
    bootstrap_action = executor.execute.await_args.args[0]
    assert bootstrap_action.action_type is ActionType.NAVIGATE


@pytest.mark.asyncio
async def test_live_runner_stops_on_retry_limit_reached() -> None:
    run_store = StubRunStore()
    executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    async def _step_run(request):
        assert run_store.state is not None
        run_store.state.step_count = 2
        run_store.state.status = RunStatus.FAILED
        return RunResponse(
            run_id=run_store.state.run_id,
            status=run_store.state.status,
            intent=run_store.state.intent,
            step_count=run_store.state.step_count,
        )

    loop.step_run = AsyncMock(side_effect=_step_run)

    result = await loop.run_live_benchmark(max_steps=5)

    assert result.status is RunStatus.FAILED
    assert loop.step_run.await_count == 1


@pytest.mark.asyncio
async def test_live_runner_stops_on_max_step_limit() -> None:
    run_store = StubRunStore()
    executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )

    async def _step_run(request):
        assert run_store.state is not None
        run_store.state.step_count += 1
        run_store.state.status = RunStatus.RUNNING
        return RunResponse(
            run_id=run_store.state.run_id,
            status=run_store.state.status,
            intent=run_store.state.intent,
            step_count=run_store.state.step_count,
        )

    loop.step_run = AsyncMock(side_effect=_step_run)

    result = await loop.run_live_benchmark(max_steps=2)

    assert result.status is RunStatus.FAILED
    assert loop.step_run.await_count == 2
    assert run_store.state is not None
    assert run_store.state.step_count == 2


@pytest.mark.asyncio
async def test_benchmark_startup_loads_gemini_api_key_from_dotenv(tmp_path, monkeypatch) -> None:
    from src.agent import benchmark

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=test-benchmark-key\n", encoding="utf-8")

    captured: dict[str, str | None] = {}

    class FakeGeminiClient:
        def __init__(self) -> None:
            import os

            captured["api_key"] = os.getenv("GEMINI_API_KEY")

    class FakeExecutor:
        async def close(self) -> None:
            return None

    class FakeRunStore:
        pass

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def run_live_benchmark(self, intent: str, benchmark_url: str, max_steps: int = 12):
            captured["intent"] = intent
            captured["benchmark_url"] = benchmark_url
            return SimpleNamespace(run_id="fake-run", model_dump_json=lambda indent=2: "{}")

    monkeypatch.setattr(benchmark, "GeminiHttpClient", FakeGeminiClient)
    monkeypatch.setattr(benchmark, "DesktopExecutor", FakeExecutor)
    monkeypatch.setattr(benchmark, "FileBackedRunStore", lambda **_kw: FakeRunStore())
    monkeypatch.setattr(benchmark, "FileBackedMemoryStore", lambda **_kw: type("FakeMem", (), {"get_hints": lambda *a, **k: [], "record_step": lambda *a, **k: []})())
    monkeypatch.setattr(benchmark, "AgentLoop", FakeLoop)
    monkeypatch.setattr(benchmark, "generate_run_metrics", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "write_run_metrics", lambda *a, **k: None)
    monkeypatch.setenv("FORM_BENCHMARK_URL", "https://example.test/form")

    await benchmark.run_form_benchmark(max_steps=1)

    assert captured["api_key"] == "test-benchmark-key"
    assert captured["benchmark_url"] == "https://example.test/form"
    assert captured["intent"] == benchmark.DEFAULT_FORM_BENCHMARK_INTENT


@pytest.mark.asyncio
async def test_benchmark_suite_runs_multiple_tasks_and_writes_metrics(tmp_path, monkeypatch) -> None:
    from src.agent import benchmark

    suite = BenchmarkSuiteSpec(
        suite_id="suite-test",
        tasks=[
            BenchmarkTaskSpec(
                task_id="task-1",
                page_url="https://example.test/form-a",
                task_type=BenchmarkTaskType.FORM_SUBMIT,
                intent="Task A",
                expected_completion_signal="done",
                difficulty_tags=["single_page"],
            ),
            BenchmarkTaskSpec(
                task_id="task-2",
                page_url="https://example.test/form-b",
                task_type=BenchmarkTaskType.MULTI_STEP_FORM,
                intent="Task B",
                expected_completion_signal="done",
                difficulty_tags=["multi_step"],
            ),
        ],
    )

    run_ids = iter(["run-a", "run-b"])

    class FakeLoop:
        async def run_live_benchmark(self, intent: str, benchmark_url: str, max_steps: int = 12):
            return RunResponse(
                run_id=next(run_ids),
                status=RunStatus.SUCCEEDED if "A" in intent else RunStatus.FAILED,
                intent=intent,
                step_count=2,
            )

    class FakeExecutor:
        async def close(self) -> None:
            return None

    def fake_build_loop(*, root_dir="runs"):
        return FakeLoop(), FakeExecutor()

    def fake_generate_run_metrics(run_id, *, root_dir="runs", task_spec=None):
        from src.models.benchmark import RunMetrics
        return RunMetrics(
            run_id=run_id,
            task_id=task_spec.task_id,
            page_url=task_spec.page_url,
            task_type=task_spec.task_type,
            tags=task_spec.difficulty_tags,
            status=RunStatus.SUCCEEDED if run_id == "run-a" else RunStatus.FAILED,
            success=run_id == "run-a",
            final_stop_reason=StopReason.FORM_SUBMITTED_SUCCESS if run_id == "run-a" else StopReason.CLICK_NO_EFFECT,
            failure_category=None if run_id == "run-a" else FailureCategory.CLICK_NO_EFFECT,
            step_count=2,
            perception_retry_count=1,
            selector_recovery_count=0,
            execution_retry_count=1,
            no_progress_events=0 if run_id == "run-a" else 1,
            loop_detected=run_id != "run-a",
            average_top_selector_score=90.0,
            average_selector_margin=10.0,
            selector_failure_count=0 if run_id == "run-a" else 1,
            average_total_elements=5.0,
            average_labeled_elements=4.0,
            average_unlabeled_elements=1.0,
            average_usable_elements=3.0,
            stale_target_events=0,
            focus_failures=0,
            click_no_effect_events=0 if run_id == "run-a" else 1,
            verification_failures=0 if run_id == "run-a" else 1,
        )

    monkeypatch.setattr(benchmark, "_build_loop", fake_build_loop)
    monkeypatch.setattr(benchmark, "generate_run_metrics", fake_generate_run_metrics)

    result = await benchmark.run_benchmark_suite(suite, root_dir=tmp_path / "runs")

    assert result.suite_summary.total_runs == 2
    assert result.suite_summary.success_count == 1
    assert len(result.run_metrics_paths) == 2
    assert (tmp_path / "runs" / "benchmark_suite_summary.json").exists()
    assert (tmp_path / "runs" / "benchmark_suite_summary.md").exists()
    payload = json.loads((tmp_path / "runs" / "benchmark_suite_summary.json").read_text(encoding="utf-8"))
    assert payload["suite_id"] == "suite-test"
    assert payload["success_count"] == 1
