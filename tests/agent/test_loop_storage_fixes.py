"""Loop and run storage fixes from the Greptile review (KAN-48)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.core.paths import prompts_dir
from operon.models.common import RunStatus, StepRequest
from operon.models.policy import ActionType, AgentAction
from operon.models.verification import VerificationResult, VerificationStatus
from operon.store import background_writer
from operon.store.background_writer import BackgroundWriter
from operon.store.run_store import FileBackedRunStore
from tests.agent.test_trust_gate_resume import _loop

SAFE = AgentAction(action_type=ActionType.TYPE, text="hello", target_element_id="box")


def _failing(stop: bool) -> AsyncMock:
    return AsyncMock(
        return_value=VerificationResult(
            status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=stop, reason="no"
        )
    )


# G8: no in-step retry after a terminal verdict, and never in the thin harness.


async def test_no_retry_after_a_terminal_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, _ = _loop(monkeypatch, [SAFE, SAFE, SAFE, SAFE])
    loop.verifier_service.verify = _failing(stop=True)

    await loop.step_run(StepRequest(run_id="run-1"))

    assert execute.call_count == 1
    assert loop.policy_service.choose_action.call_count == 1


async def test_the_thin_harness_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, _ = _loop(monkeypatch, [SAFE, SAFE, SAFE, SAFE])
    loop.verifier_service.verify = _failing(stop=False)
    loop.thin_harness = True

    await loop.step_run(StepRequest(run_id="run-1"))

    assert execute.call_count == 1


async def test_the_full_harness_still_retries_a_plain_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, execute, _ = _loop(monkeypatch, [SAFE, SAFE, SAFE, SAFE])
    loop.verifier_service.verify = _failing(stop=False)
    loop.thin_harness = False

    await loop.step_run(StepRequest(run_id="run-1"))

    assert execute.call_count > 1


# G9: background writes land in order and atomically.


async def test_background_writes_land_in_the_order_queued(tmp_path: Path) -> None:
    writer = BackgroundWriter()
    target = tmp_path / "state.json"
    for i in range(200):
        writer.enqueue(target, json.dumps({"i": i}) + " " * (i % 7) * 10_000)
    await writer.flush()

    assert json.loads(target.read_text())["i"] == 199
    assert list(tmp_path.glob(".*.tmp")) == []


def test_the_writer_pool_has_one_worker() -> None:
    assert background_writer._thread_pool._max_workers == 1


# G20: a finished run's conversation and frames are released.


def test_release_run_drops_the_per_run_caches() -> None:
    backend = BrowserComputerUseBackend(client=SimpleNamespace(), prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")
    backend._conversations["run-1"] = object()
    backend._last_frames["run-1"] = object()
    backend._advisory_hints["run-1"] = [("h", "s")]
    backend._conversations["run-2"] = object()

    backend.release_run("run-1")

    assert "run-1" not in backend._conversations
    assert "run-1" not in backend._last_frames
    assert "run-1" not in backend._advisory_hints
    assert "run-2" in backend._conversations


async def test_cleanup_releases_backend_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    loop, _, _ = _loop(monkeypatch, [SAFE])
    released: list[str] = []
    loop.perception_service.release_run = released.append

    await loop._cleanup_completed_run("run-1")

    assert released == ["run-1"]


# G21: every status change is written, so the observer is not a step behind.


async def test_set_status_running_writes_the_snapshot(tmp_path: Path) -> None:
    store = FileBackedRunStore(tmp_path)
    record = store.create_run("Open the page")
    record.current_subgoal = "after step 1"

    await store.set_status(record.run_id, RunStatus.RUNNING)
    await background_writer.bg_writer.flush()

    saved = json.loads((tmp_path / record.run_id / "state.json").read_text())
    assert saved["status"] == "running"
    assert saved["current_subgoal"] == "after step 1"


# G24: shutdown closes every run's browser session, not just the persistent one.


async def test_aclose_closes_open_batch_sessions(tmp_path: Path) -> None:
    from operon.executor.browser_native import NativeBrowserExecutor

    executor = NativeBrowserExecutor(artifact_dir=tmp_path)
    closed: list[str] = []

    async def aclose_run(run_id: str) -> int:
        closed.append(run_id)
        executor._sessions.pop(run_id, None)
        return 1

    executor._sessions = {"run-a": object(), "run-b": object()}
    executor.aclose_run = aclose_run
    executor.close_persistent_browser = AsyncMock()

    await executor.aclose()

    assert sorted(closed) == ["run-a", "run-b"]
    executor.close_persistent_browser.assert_awaited_once()



def test_a_write_is_not_lost_when_the_swap_is_refused(tmp_path, monkeypatch) -> None:
    # Windows refuses os.replace while a reader has the target open.
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    def refuse(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(background_writer.os, "replace", refuse)
    monkeypatch.setattr(background_writer.time, "sleep", lambda _s: None)
    background_writer._write_file(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
