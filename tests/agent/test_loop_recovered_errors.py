"""KAN-158: the loop's broad-except recoveries are logged, counted per run, and
re-raised for bugs in strict mode (OPERON_STRICT_ERRORS, on in this suite)."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from evals import live_bench
from operon.agent.loop import AgentLoop
from operon.models.common import RunTaskRequest
from operon.store.run_store import FileBackedRunStore


def _loop(tmp_path, recording_error: BaseException) -> tuple[AgentLoop, FileBackedRunStore]:
    store = FileBackedRunStore(tmp_path)
    executor = Mock()
    executor.reset_desktop = AsyncMock()
    executor.configure_run = Mock()
    executor.start_run_recording = AsyncMock(side_effect=recording_error)
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=store,
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    return loop, store


async def test_a_recovery_is_counted_in_the_run_state_and_logged(tmp_path, caplog):
    loop, store = _loop(tmp_path, RuntimeError("disk full"))

    with caplog.at_level(logging.WARNING, logger="operon.agent.loop"):
        response = await loop.start_run(RunTaskRequest(intent="Open the page and read it."))

    state = await store.get_run(response.run_id)
    assert state is not None and state.recovered_errors == {"start_run_recording": 1}
    saved = json.loads((tmp_path / response.run_id / "state.json").read_text(encoding="utf-8"))
    assert saved["recovered_errors"] == {"start_run_recording": 1}
    record = next(r for r in caplog.records if "recovered at start_run_recording" in r.getMessage())
    assert record.exc_info is not None  # the traceback is logged, not just the message


async def test_counts_add_up_across_flushes(tmp_path):
    loop, store = _loop(tmp_path, RuntimeError("disk full"))
    response = await loop.start_run(RunTaskRequest(intent="Open the page and read it."))

    loop._recovered(response.run_id, "start_run_recording", RuntimeError("again"))
    loop._recovered(response.run_id, "context_reset", RuntimeError("no page"))
    await loop._flush_recovered_errors(response.run_id)

    state = await store.get_run(response.run_id)
    assert state is not None
    assert state.recovered_errors == {"start_run_recording": 2, "context_reset": 1}


async def test_strict_mode_reraises_a_bug(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERON_STRICT_ERRORS", "true")
    loop, _ = _loop(tmp_path, AttributeError("'NoneType' object has no attribute 'stop'"))

    with pytest.raises(AttributeError):
        await loop.start_run(RunTaskRequest(intent="Open the page and read it."))


async def test_without_strict_mode_a_bug_is_counted_like_any_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERON_STRICT_ERRORS", "false")
    loop, store = _loop(tmp_path, AttributeError("'NoneType' object has no attribute 'stop'"))

    response = await loop.start_run(RunTaskRequest(intent="Open the page and read it."))

    state = await store.get_run(response.run_id)
    assert state is not None and state.recovered_errors == {"start_run_recording": 1}


async def test_strict_mode_still_recovers_from_ordinary_errors(tmp_path):
    # The suite runs with strict mode on: a runtime failure (not a bug) is recovered.
    loop, store = _loop(tmp_path, OSError("recorder unavailable"))

    response = await loop.start_run(RunTaskRequest(intent="Open the page and read it."))

    state = await store.get_run(response.run_id)
    assert state is not None and state.recovered_errors == {"start_run_recording": 1}


def test_live_bench_reports_recoveries(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps({"stop_reason": None, "recovered_errors": {"page_load_perceive": 2}}), encoding="utf-8"
    )
    monkeypatch.setattr(live_bench, "_run_dir", lambda run_id: tmp_path / run_id)
    assert live_bench._recovered_errors("run-1") == {"page_load_perceive": 2}
    assert live_bench._recovered_errors("missing") == {}

    def result(name, recovered):
        return live_bench.TaskResult(
            name, "c", "r", "succeeded", "success", "-", 1, 0, 5, 1, 0, "", "-", recovered_errors=recovered
        )

    summary = live_bench.summarize([result("a", {"page_load_perceive": 2}), result("b", {})])
    assert summary["recovered_errors"] == 2
    assert summary["recovered_tasks"] == 1
