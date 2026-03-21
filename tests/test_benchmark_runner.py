"""Focused tests for the bounded live benchmark runner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agent.loop import AgentLoop
from src.models.common import RunResponse, RunStatus, StopReason
from src.models.policy import ActionType
from src.models.state import AgentState


class StubRunStore:
    """Minimal in-memory run store for benchmark runner tests."""

    def __init__(self) -> None:
        self.state: AgentState | None = None

    def create_run(self, intent: str) -> AgentState:
        self.state = AgentState(run_id="run-benchmark", intent=intent, status=RunStatus.PENDING)
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
    browser_executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        browser_executor=browser_executor,
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
    bootstrap_action = browser_executor.execute.await_args.args[0]
    assert bootstrap_action.action_type is ActionType.NAVIGATE


@pytest.mark.asyncio
async def test_live_runner_stops_on_retry_limit_reached() -> None:
    run_store = StubRunStore()
    browser_executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        browser_executor=browser_executor,
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
    browser_executor = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(success=True)))
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=run_store,
        policy_service=Mock(),
        browser_executor=browser_executor,
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

    class FakeBrowserExecutor:
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
            return SimpleNamespace(model_dump_json=lambda indent=2: "{}")

    monkeypatch.setattr(benchmark, "GeminiHttpClient", FakeGeminiClient)
    monkeypatch.setattr(benchmark, "PlaywrightBrowserExecutor", FakeBrowserExecutor)
    monkeypatch.setattr(benchmark, "FileBackedRunStore", lambda: FakeRunStore())
    monkeypatch.setattr(benchmark, "AgentLoop", FakeLoop)
    monkeypatch.setenv("FORM_BENCHMARK_URL", "https://example.test/form")

    await benchmark.run_form_benchmark(max_steps=1)

    assert captured["api_key"] == "test-benchmark-key"
    assert captured["benchmark_url"] == "https://example.test/form"
    assert captured["intent"] == benchmark.DEFAULT_FORM_BENCHMARK_INTENT
