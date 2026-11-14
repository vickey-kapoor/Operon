"""KAN-152: runs stay with the loop that owns them; memory and cleanup share runs_dir()."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent import control
from operon.api import ws_stream
from operon.models.common import RunStatus
from operon.store.cleanup import cleanup_old_runs
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore, stored_environment


async def test_a_run_is_stamped_with_its_loop_and_the_other_loop_will_not_load_it(tmp_path: Path) -> None:
    browser = FileBackedRunStore(tmp_path, environment="browser")
    record = browser.create_run("Open the docs")

    assert record.environment == "browser"
    assert stored_environment(record.run_id, tmp_path) == "browser"
    assert await FileBackedRunStore(tmp_path, environment="desktop").get_run(record.run_id) is None
    assert await FileBackedRunStore(tmp_path, environment="browser").get_run(record.run_id) is not None


async def test_a_run_saved_before_the_field_existed_still_loads(tmp_path: Path) -> None:
    record = FileBackedRunStore(tmp_path).create_run("Open the docs")
    assert record.environment is None
    assert stored_environment(record.run_id, tmp_path) is None
    assert await FileBackedRunStore(tmp_path, environment="desktop").get_run(record.run_id) is not None


def test_the_front_door_routes_a_restarted_servers_desktop_run_to_the_desktop_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from operon.api import frontdoor

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    record = FileBackedRunStore(tmp_path, environment="desktop").create_run("Open Notepad")
    monkeypatch.setattr(frontdoor._loops, "desktop_loop_holds_run", lambda run_id: False)

    assert frontdoor._environment_of(record.run_id) == "desktop"
    assert frontdoor._environment_of("unknown-run") == "browser"


def test_memory_lives_under_the_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path / "runs-root"))
    monkeypatch.chdir(tmp_path)

    store = FileBackedMemoryStore()

    assert store.memory_path == tmp_path / "runs-root" / "memory" / "memory.jsonl"


def test_memory_kept_under_the_old_working_directory_path_is_carried_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path / "runs-root"))
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "runs" / "memory" / "memory.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"kept": true}\n', encoding="utf-8")

    store = FileBackedMemoryStore()

    assert store.memory_path.read_text(encoding="utf-8").startswith('{"kept": true}\n')


def test_cleanup_never_deletes_the_memory_folder(tmp_path: Path) -> None:
    old = time.time() - 30 * 86400
    run = tmp_path / "old-run"
    run.mkdir()
    (run / "state.json").write_text("{}", encoding="utf-8")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.jsonl").write_text("{}", encoding="utf-8")
    for path in (run / "state.json", run, memory / "memory.jsonl", memory):
        os.utime(path, (old, old))

    deleted, _ = cleanup_old_runs(root_dir=tmp_path, keep_days=14, delete=True)

    assert deleted == [run]
    assert (memory / "memory.jsonl").exists()


async def test_ws_control_goes_to_the_loop_that_owns_the_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    browser_loop = SimpleNamespace(pause_run=AsyncMock())
    desktop_loop = SimpleNamespace(pause_run=AsyncMock(return_value=SimpleNamespace(status=RunStatus.WAITING_FOR_USER)))
    monkeypatch.setattr(ws_stream, "_agent_loop", browser_loop)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)
    ws_stream.set_active_run("run-d", loop=desktop_loop)
    try:
        await ws_stream._handle_control({"type": "pause"})
    finally:
        ws_stream.set_active_run(None)

    desktop_loop.pause_run.assert_awaited_once_with("run-d")
    browser_loop.pause_run.assert_not_awaited()


async def test_ws_input_is_refused_while_a_run_drives_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    record = SimpleNamespace(in_takeover=False, status=RunStatus.RUNNING)
    loop = SimpleNamespace(run_store=SimpleNamespace(get_run=AsyncMock(return_value=record)))
    manager = SimpleNamespace(inject_input=AsyncMock())
    events: list[dict] = []
    monkeypatch.setattr(ws_stream, "_browser_manager", manager)
    monkeypatch.setattr(ws_stream, "publish_event", events.append)
    ws_stream.set_active_run("run-1", loop=loop)
    try:
        await ws_stream._handle_control({"type": "inject_input", "input_type": "click"})
        manager.inject_input.assert_not_awaited()
        assert events[-1]["error"] == "take control of the run first"

        record.in_takeover = True
        await ws_stream._handle_control({"type": "inject_input", "input_type": "click"})
        manager.inject_input.assert_awaited_once()
    finally:
        ws_stream.set_active_run(None)


async def test_a_nan_confidence_threshold_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "_confidence_threshold", 0.3)
    events: list[dict] = []
    monkeypatch.setattr(ws_stream, "publish_event", events.append)

    await ws_stream._handle_control({"type": "set_confidence_threshold", "threshold": "nan"})

    assert control.get_confidence_threshold() == 0.3
    assert "error" in events[-1]


def test_the_step_event_reads_a_computer_use_perception(tmp_path: Path) -> None:
    from operon.api.runtime import events as runtime_events

    path = tmp_path / "computer_use_parsed.json"
    path.write_text(
        json.dumps({"perception": {"page_hint": "form", "visible_elements": [{"element_id": "e1", "primary_name": "Email"}], "focused_element_id": "e1"}}),
        encoding="utf-8",
    )

    assert runtime_events._read_perception_summary(path) == "form . focus: Email"
