"""Process runtime with a context per run (KAN-156)."""

from __future__ import annotations

import asyncio
import time

import pytest

from operon.agent import control
from operon.core import runtime as runtime_mod
from operon.core.runtime import OperonRuntime, current_runtime, reset_runtime


@pytest.fixture
def fresh_runtime(monkeypatch):
    fresh = OperonRuntime()
    monkeypatch.setattr(runtime_mod, "_runtime", fresh)
    return fresh


def test_run_context_is_created_once_and_peek_never_creates(fresh_runtime) -> None:
    assert fresh_runtime.peek_run("r1") is None
    first = fresh_runtime.run("r1")
    assert fresh_runtime.run("r1") is first
    assert fresh_runtime.peek_run("r1") is first
    fresh_runtime.forget_run("r1")
    assert fresh_runtime.peek_run("r1") is None


def test_override_hints_live_on_the_run_context(fresh_runtime) -> None:
    control.set_override_hint("r1", "click Next")
    assert fresh_runtime.peek_run("r1").override_hint == "click Next"
    assert control.consume_override_hint("r2") is None
    assert fresh_runtime.peek_run("r2") is None  # consuming never creates a context
    assert control.consume_override_hint("r1") == "click Next"
    assert control.consume_override_hint("r1") is None


def test_control_settings_are_per_runtime(fresh_runtime) -> None:
    control.set_confidence_threshold(0.7)
    control.set_disabled_rules(["a"])
    assert fresh_runtime.control.confidence_threshold == 0.7
    assert fresh_runtime.control.disabled_rules == frozenset({"a"})
    other = OperonRuntime()
    assert other.control.confidence_threshold == 0.0


def test_finished_runs_are_pruned_oldest_first(fresh_runtime, monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "MAX_FINISHED_RUNS", 2)
    fresh_runtime.run("live").started_at = time.time()  # still going: never pruned
    for name in ("a", "b", "c"):
        fresh_runtime.run(name).finished_at = time.time()
    fresh_runtime.run("d")  # adding a run prunes
    assert list(fresh_runtime.runs) == ["live", "b", "c", "d"]


async def test_a_finished_run_with_a_task_still_running_is_kept(fresh_runtime, monkeypatch) -> None:
    monkeypatch.setattr(runtime_mod, "MAX_FINISHED_RUNS", 0)
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    context = fresh_runtime.run("a")
    context.finished_at = time.time()
    context.auto_run_task = task
    fresh_runtime.run("b")
    assert "a" in fresh_runtime.runs
    release.set()
    await task


async def test_auto_run_task_is_kept_on_the_run_context(fresh_runtime, monkeypatch) -> None:
    from operon.api.runtime import tasks

    release = asyncio.Event()

    async def fake_auto_run(_loop, _run_id: str, _max_steps: int) -> None:
        await release.wait()

    monkeypatch.setattr(tasks, "auto_run_loop", fake_auto_run)
    task = tasks.schedule_auto_run(object(), "r1", 5)
    assert fresh_runtime.peek_run("r1").auto_run_task is task
    await tasks.cancel_auto_run("r1")
    assert tasks.auto_run_task("r1") is None


def test_stream_and_loops_live_on_the_runtime(fresh_runtime) -> None:
    from operon.api import ws_stream
    from operon.api.runtime import loops
    from operon.browser.manager import get_active_manager, set_active_manager

    marker = object()
    ws_stream.set_active_run("r1", marker)
    assert fresh_runtime.stream.active_run_id == "r1" and fresh_runtime.stream.active_loop is marker
    ws_stream.set_agent_loop(marker)
    assert fresh_runtime.browser_loop is marker
    set_active_manager(marker)  # type: ignore[arg-type]
    assert get_active_manager() is marker
    fresh_runtime.browser_loop = None
    fresh_runtime.desktop_loop = type("L", (), {"executor": "desk"})()
    assert list(loops.iter_built_executors()) == ["desk"]


def test_reset_runtime_starts_empty(monkeypatch) -> None:
    previous = current_runtime()
    try:
        previous.run("kept")
        fresh = reset_runtime()
        assert fresh is current_runtime() and fresh is not previous
        assert fresh.peek_run("kept") is None and fresh.browser_loop is None
    finally:
        monkeypatch.setattr(runtime_mod, "_runtime", previous)
