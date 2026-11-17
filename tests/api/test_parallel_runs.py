"""KAN-81: OPERON_PARALLEL_RUNS. Batch browser runs each get their own loop and
executor so their steps overlap; the cap refuses extra starts; desktop and
observable runs stay serialized. Fake executors only: no browser, no screen."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from operon.agent.backends.base import AgentBackend
from operon.agent.loop import AgentLoop, StartUrlBlockedError
from operon.api.routes import browser as browser_routes
from operon.api.runtime import loops
from operon.api.runtime.tasks import cleanup_cancelled_run
from operon.core.contracts.perception import Environment
from operon.core.runtime import current_runtime, reset_runtime
from operon.models.common import RunTaskRequest, StepRequest
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

STEP_SECONDS = 0.15


def _fake_executor():
    executor = Mock()
    executor.configure_run = Mock()
    executor.set_current_run_id = Mock()
    executor.start_run_recording = AsyncMock()
    executor.stop_run_recording = AsyncMock()
    executor.aclose_run = AsyncMock()
    executor.aclose = AsyncMock()
    return executor


def _layers(**override):
    layers = dict(
        capture_service=Mock(),
        perception_service=Mock(),
        policy_service=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    layers.update(override)
    return layers


@pytest.fixture
def parallel(tmp_path, monkeypatch):
    """N=2 with loops built from fake executors; ``built`` lists every executor made."""
    reset_runtime()
    monkeypatch.setenv("OPERON_PARALLEL_RUNS", "2")
    store = FileBackedRunStore(tmp_path)
    memory = FileBackedMemoryStore()
    built: list = []

    def fake_new(*, run_store=None, memory_store=None, loop_cls=AgentLoop):
        executor = _fake_executor()
        built.append(executor)
        loop = loop_cls(
            run_store=run_store or store,
            executor=executor,
            memory_store=memory_store or memory,
            environment=Environment.BROWSER,
            **_layers(),
        )
        return loop, executor

    monkeypatch.setattr(loops, "_new_browser_loop", fake_new)
    base, _ = fake_new()
    current_runtime().browser_loop = base
    built.clear()
    yield built
    reset_runtime()


@pytest.fixture
def timed_steps(monkeypatch):
    """Replace the step body with a sleep that records (loop id, start, end)."""
    spans: list[tuple[int, float, float]] = []

    async def fake_step(self, request):
        begin = time.monotonic()
        await asyncio.sleep(STEP_SECONDS)
        spans.append((id(self), begin, time.monotonic()))
        return Mock()

    monkeypatch.setattr(AgentLoop, "_step_run_locked", fake_step)
    return spans


def _request() -> RunTaskRequest:
    return RunTaskRequest(intent="Open the page and read it.", mode="batch")


def _overlap(spans) -> bool:
    (_, a0, a1), (_, b0, b1) = spans[:2]
    return a0 < b1 and b0 < a1


async def test_two_parallel_runs_step_at_the_same_time_on_their_own_executors(parallel, timed_steps):
    (loop_a, first), (loop_b, second) = [await loops.start_parallel_browser_run(_request()) for _ in range(2)]

    assert loop_a is not loop_b and loop_a.executor is not loop_b.executor
    assert len(parallel) == 2  # one fake executor per run
    assert loop_a._executor_lock is not loop_b._executor_lock
    await asyncio.gather(
        loop_a.step_run(StepRequest(run_id=first.run_id)),
        loop_b.step_run(StepRequest(run_id=second.run_id)),
    )
    assert len({span[0] for span in timed_steps}) == 2 and _overlap(timed_steps)
    assert loops.loop_for_run(first.run_id) is loop_a and loops.loop_for_run(second.run_id) is loop_b
    # Each run was set up on its own executor only.
    loop_a.executor.configure_run.assert_called_once()
    assert loop_a.executor.configure_run.call_args.args[0] == first.run_id
    assert loop_b.executor.configure_run.call_args.args[0] == second.run_id


async def test_runs_on_one_shared_loop_still_take_turns(tmp_path, timed_steps):
    loop = AgentLoop(run_store=FileBackedRunStore(tmp_path), executor=_fake_executor(), **_layers())
    first = await loop.start_run(_request())
    second = await loop.start_run(_request())
    await asyncio.gather(
        loop.step_run(StepRequest(run_id=first.run_id)),
        loop.step_run(StepRequest(run_id=second.run_id)),
    )
    assert not _overlap(timed_steps)


async def test_desktop_runs_stay_serialized_even_with_a_parallel_cap(tmp_path, monkeypatch, timed_steps):
    reset_runtime()
    monkeypatch.setenv("OPERON_PARALLEL_RUNS", "2")
    loop = AgentLoop(
        run_store=FileBackedRunStore(tmp_path),
        executor=_fake_executor(),
        environment=Environment.DESKTOP,
        **_layers(),
    )
    first = await loop.start_run(_request())
    second = await loop.start_run(_request())
    await asyncio.gather(
        loop.step_run(StepRequest(run_id=first.run_id)),
        loop.step_run(StepRequest(run_id=second.run_id)),
    )
    assert not _overlap(timed_steps)
    assert not loops.runs_in_parallel("observable")  # one CDP port: live-view runs take turns
    assert loops.runs_in_parallel("batch")


def test_the_default_keeps_todays_behaviour(monkeypatch):
    monkeypatch.delenv("OPERON_PARALLEL_RUNS", raising=False)
    assert loops.parallel_runs() == 1
    assert not loops.runs_in_parallel("batch")


async def test_the_cap_refuses_the_third_start_until_a_run_finishes(parallel):
    _, first = await loops.start_parallel_browser_run(_request())
    await loops.start_parallel_browser_run(_request())

    with pytest.raises(loops.RunSlotsFull):
        await loops.start_parallel_browser_run(_request())
    with pytest.raises(HTTPException) as refused:
        await browser_routes.run_task(_request())
    assert refused.value.status_code == 429 and refused.value.headers == {"Retry-After": "5"}

    await cleanup_cancelled_run(loops.loop_for_run(first.run_id), first.run_id)
    _, third = await loops.start_parallel_browser_run(_request())  # the slot came back
    assert loops.loop_for_run(third.run_id) is not None


async def test_executors_close_at_finish_and_cancel_but_not_before(parallel):
    loop_a, first = await loops.start_parallel_browser_run(_request())
    loop_b, second = await loops.start_parallel_browser_run(_request())
    exec_a, exec_b = loop_a.executor, loop_b.executor
    exec_a.aclose.assert_not_called()

    await loop_a._cleanup_completed_run(first.run_id)  # a run that finished
    exec_a.aclose_run.assert_awaited_once_with(first.run_id)
    exec_a.aclose.assert_awaited_once()
    exec_b.aclose.assert_not_called()
    assert loops.loop_for_run(first.run_id) is None

    await cleanup_cancelled_run(loop_b, second.run_id)  # a cancelled run
    exec_b.aclose.assert_awaited_once()
    assert loops.loop_for_run(second.run_id) is None

    await loop_a._cleanup_completed_run(first.run_id)  # idempotent: no double release
    exec_a.aclose.assert_awaited_once()
    assert current_runtime().run_slots._value == 2


async def test_a_failed_start_gives_the_slot_and_executor_back(parallel, monkeypatch):
    async def refuse(self, request):
        raise StartUrlBlockedError("blocked")

    monkeypatch.setattr(AgentLoop, "start_run", refuse)
    with pytest.raises(StartUrlBlockedError):
        await loops.start_parallel_browser_run(_request())
    parallel[0].aclose.assert_awaited_once()
    assert current_runtime().run_slots._value == 2


class _Backend(AgentBackend):
    __abstractmethods__ = frozenset()
    def __init__(self) -> None:
        self._conversations: dict[str, list[str]] = {}

    perceive = choose_action = latest_debug_artifacts = None  # type: ignore[assignment]


async def test_each_parallel_run_has_its_own_backend_so_conversations_are_distinct(parallel, monkeypatch):
    backends: list[_Backend] = []

    def fake_new(*, run_store=None, memory_store=None, loop_cls=AgentLoop):
        backend = _Backend()
        backends.append(backend)
        executor = _fake_executor()
        loop = loop_cls(
            run_store=run_store,
            executor=executor,
            memory_store=memory_store,
            environment=Environment.BROWSER,
            **_layers(perception_service=backend),
        )
        return loop, executor

    monkeypatch.setattr(loops, "_new_browser_loop", fake_new)
    _, first = await loops.start_parallel_browser_run(_request())
    _, second = await loops.start_parallel_browser_run(_request())
    backends[0]._conversations[first.run_id] = ["a"]
    backends[1]._conversations[second.run_id] = ["b"]
    assert backends[0] is not backends[1]
    assert second.run_id not in backends[0]._conversations
    backends[0].release_run(first.run_id)  # releasing one run leaves the other's alone
    assert backends[1]._conversations == {second.run_id: ["b"]}


def test_backend_per_run_caches_are_keyed_by_run_id():
    assert "_conversations" in AgentBackend._PER_RUN_CACHES
    backend = _Backend()
    backend._conversations = {"r1": ["x"], "r2": ["y"]}
    backend.release_run("r1")
    assert backend._conversations == {"r2": ["y"]}


async def test_observer_asks_a_parallel_runs_own_executor(parallel, tmp_path, monkeypatch):
    """A RUNNING parallel run is not reconciled to failed, and its live frame is served."""
    from operon.api.routes import observer
    from operon.models.common import RunStatus

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path))
    base = current_runtime().browser_loop
    base.executor.current_url_for_run = AsyncMock(return_value=None)  # shared executor: no session
    loop, response = await loops.start_parallel_browser_run(_request())
    loop.executor.current_url_for_run = AsyncMock(return_value="https://example.com")
    loop.executor.live_frame_png = AsyncMock(return_value=b"png-bytes")
    await loop.run_store.set_status(response.run_id, RunStatus.RUNNING)

    snapshot = await observer.observer_run(response.run_id)
    assert snapshot["run"]["current_url"] == "https://example.com"
    await observer.observer_runs(limit=20)
    assert (await loop.run_store.get_run(response.run_id)).status is RunStatus.RUNNING
    frame = await observer.observer_live_browser(response.run_id)
    assert frame.body == b"png-bytes"
    base.executor.current_url_for_run.assert_not_called()
