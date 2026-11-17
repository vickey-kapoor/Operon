"""The exclusive cross-process claim a resume takes on a run (KAN-80)."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from operon.models.common import RunStatus, StopReason
from operon.store import owner as run_owner
from operon.store.checkpoints import mark_interrupted_runs
from operon.store.owner import RunOwnedError
from tests.agent.test_checkpoints import STOP, _crash, _on_disk, _use
from tests.fixtures.frontdoor import build_loop

OTHER_PID = 2_000_000_001


def _foreign(run_dir: Path, pid: int = OTHER_PID) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "owner.json").write_text(
        json.dumps({"pid": pid, "created": 7, "host": socket.gethostname()}), encoding="utf-8"
    )


@pytest.fixture
def alive(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    """The set of foreign pids treated as running (this process always is); creation time matches."""
    live: set[int] = set()
    monkeypatch.setattr(run_owner, "pid_alive", lambda pid: pid in live or pid == os.getpid())
    monkeypatch.setattr(run_owner, "process_create_time", lambda pid: 7)
    return live


def test_a_live_owner_blocks_a_second_claim(tmp_path: Path, alive: set[int]) -> None:
    alive.add(OTHER_PID)
    _foreign(tmp_path)

    with pytest.raises(RunOwnedError, match=f"already being resumed by pid {OTHER_PID}"):
        run_owner.acquire(tmp_path)

    assert run_owner.read(tmp_path)["pid"] == OTHER_PID


def test_a_stale_claim_is_taken_over(tmp_path: Path, alive: set[int]) -> None:
    _foreign(tmp_path)  # OTHER_PID is not alive

    run_owner.acquire(tmp_path)

    assert run_owner.read(tmp_path)["pid"] == os.getpid()
    assert not (tmp_path / "owner.takeover").exists()


def test_two_processes_that_both_see_it_stale_cannot_both_win(
    tmp_path: Path, alive: set[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """B reads the file as stale, then A takes over before B acts: B must be refused."""
    _foreign(tmp_path)
    real = run_owner._held_by_other
    calls = {"n": 0}

    def racing(run_dir: Path):
        result = real(run_dir)  # stale: None
        calls["n"] += 1
        if calls["n"] == 1:  # A wins in the window after B's look
            _foreign(run_dir, OTHER_PID + 1)
            alive.add(OTHER_PID + 1)
        return result

    monkeypatch.setattr(run_owner, "_held_by_other", racing)

    with pytest.raises(RunOwnedError):
        run_owner.acquire(tmp_path)

    assert run_owner.read(tmp_path)["pid"] == OTHER_PID + 1  # A's claim is intact


def test_exclusive_create_lets_exactly_one_writer_win(tmp_path: Path) -> None:
    path = tmp_path / "x"
    assert run_owner._create_exclusive(path, "a") is True
    assert run_owner._create_exclusive(path, "b") is False
    assert path.read_text(encoding="utf-8") == "a"


def test_this_process_is_not_blocked_by_its_own_claim(tmp_path: Path, alive: set[int]) -> None:
    run_owner.acquire(tmp_path)
    run_owner.acquire(tmp_path)

    assert run_owner.read(tmp_path)["pid"] == os.getpid()


async def test_a_refused_resume_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, alive: set[int]
) -> None:
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    alive.add(OTHER_PID)
    _foreign(tmp_path / run_id)
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)
    configured: list[str] = []
    loop.executor.configure_run = lambda rid, **kw: configured.append(rid)
    before = (tmp_path / run_id / "state.json").read_text(encoding="utf-8")

    with pytest.raises(RunOwnedError, match="already being resumed"):
        await loop.resume_interrupted(run_id)

    assert configured == []
    assert (tmp_path / run_id / "state.json").read_text(encoding="utf-8") == before
    assert _on_disk(tmp_path, run_id).stop_reason is StopReason.INTERRUPTED
    assert run_owner.read(tmp_path / run_id)["pid"] == OTHER_PID


async def test_the_same_process_can_resume_again_after_its_own_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    await loop.resume_interrupted(run_id)

    assert run_owner.read(tmp_path / run_id)["pid"] == os.getpid()
    loop.run_store.acquire_run(run_id)  # its own claim does not block it
