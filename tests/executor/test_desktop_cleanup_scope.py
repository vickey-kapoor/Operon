"""Desktop cleanup closes only what the run opened, never the user's own windows (G6)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from operon.executor import desktop
from operon.models.policy import ActionType, AgentAction
from tests.executor.test_desktop import _make_executor


def _tasklist(*pids: int, exe: str = "notepad.exe") -> SimpleNamespace:
    lines = [f'"{exe}","{pid}","Console","1","12,345 K"' for pid in pids]
    return SimpleNamespace(returncode=0, stdout="\n".join(lines) or "INFO: No tasks are running which match the specified criteria.")


def test_image_pids_parses_tasklist_csv() -> None:
    with patch("operon.executor.desktop.subprocess.run", return_value=_tasklist(10, 20)):
        assert desktop._image_pids("notepad.exe") == frozenset({10, 20})
    with patch("operon.executor.desktop.subprocess.run", return_value=_tasklist()):
        assert desktop._image_pids("notepad.exe") == frozenset()
    with patch("operon.executor.desktop.subprocess.run", side_effect=FileNotFoundError):
        assert desktop._image_pids("notepad.exe") is None


async def _launch(executor, pids_before, pids_after) -> None:
    with (
        patch("operon.executor.desktop._image_pids", side_effect=[frozenset(pids_before), frozenset(pids_after)]),
        patch("operon.executor.desktop.subprocess.Popen"),
    ):
        result = await executor._exec_launch_app(AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad"))
    assert result.success is True


def _cleanup(executor, pids_now) -> tuple[int, list[list[str]]]:
    executor._launched_processes.clear()  # only the image-name fallback is under test
    kills: list[list[str]] = []

    def fake_run(args, **kwargs):
        kills.append(args)
        return SimpleNamespace(returncode=0, stdout="")

    with (
        patch("operon.executor.desktop._image_pids", return_value=frozenset(pids_now)),
        patch("operon.executor.desktop.subprocess.run", side_effect=fake_run),
    ):
        closed = executor.cleanup_run("run-1")
    assert all("/IM" not in args for args in kills)
    return closed, kills


def _executor():
    executor = _make_executor()
    executor._post_launch_delay = 0
    executor._capture_after = AsyncMock(return_value="after.png")
    executor.set_current_run_id("run-1")
    return executor


@pytest.mark.asyncio
async def test_cleanup_kills_only_processes_the_launch_started() -> None:
    executor = _executor()
    # The user already has Notepad open as pid 100; the launch starts pid 200.
    await _launch(executor, {100}, {100, 200})

    closed, kills = _cleanup(executor, {100, 200})

    assert kills == [["taskkill", "/PID", "200", "/F"]]
    assert closed == 1


@pytest.mark.asyncio
async def test_cleanup_leaves_a_window_the_user_opened_during_the_run() -> None:
    executor = _executor()
    await _launch(executor, {100}, {100, 200})
    # Later in the run the user opens another Notepad (pid 300).

    closed, kills = _cleanup(executor, {100, 200, 300})

    assert kills == [["taskkill", "/PID", "200", "/F"]]
    assert closed == 1


@pytest.mark.asyncio
async def test_cleanup_covers_every_launch_of_the_same_app() -> None:
    executor = _executor()
    await _launch(executor, {100}, {100, 200})
    await _launch(executor, {100, 200, 300}, {100, 200, 300, 400})  # 300 is the user's

    closed, kills = _cleanup(executor, {100, 200, 300, 400})

    assert kills == [["taskkill", "/PID", "200", "/F"], ["taskkill", "/PID", "400", "/F"]]
    assert closed == 2


def test_cleanup_kills_nothing_by_name_when_processes_cannot_be_listed() -> None:
    executor = _make_executor()
    executor._launched_app_names["run-1"] = ["chrome.exe"]  # launched, but no ownership recorded

    with patch("operon.executor.desktop.subprocess.run") as run:
        closed = executor.cleanup_run("run-1")

    assert closed == 0
    run.assert_not_called()


def test_click_check_samples_a_monitor_left_of_the_primary() -> None:
    virtual = {"left": -1920, "top": -200, "width": 3840, "height": 1280}
    assert desktop._sample_box(virtual, -1000, 300, 30) == {"left": -1030, "top": 270, "width": 60, "height": 60}
    # Near the virtual screen's edges the box moves inside instead of crossing it.
    assert desktop._sample_box(virtual, -1915, -195, 30)["left"] == -1920
    assert desktop._sample_box(virtual, -1915, -195, 30)["top"] == -200
    assert desktop._sample_box(virtual, 1915, 1075, 30) == {"left": 1860, "top": 1020, "width": 60, "height": 60}


def test_region_check_reads_pixels_at_negative_coordinates() -> None:
    executor = _make_executor()
    grabbed: list[dict] = []

    class FakeMss:
        monitors = [{"left": -1920, "top": 0, "width": 3840, "height": 1080}]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, region):
            grabbed.append(region)
            return SimpleNamespace(rgb=b"\x00" * 60 * 60 * 3)

    with patch("operon.executor.desktop._mss", return_value=FakeMss()):
        executor._sample_region(-1000, 500)

    assert grabbed == [{"left": -1030, "top": 470, "width": 60, "height": 60}]


@pytest.mark.asyncio
async def test_a_run_ending_closes_its_apps_only_when_asked() -> None:
    executor = _executor()
    await _launch(executor, {100}, {100, 200})

    # A benchmark run: the loop's end-of-run hook leaves the windows for the grader.
    with patch("operon.executor.desktop.subprocess.run") as run:
        assert await executor.aclose_run("run-1") == 0
    run.assert_not_called()

    # A task-page run: its apps close when it ends, and only once.
    executor.close_apps_when_done("run-1")
    with patch.object(executor, "cleanup_run", return_value=1) as cleanup:
        assert await executor.aclose_run("run-1") == 1
        assert await executor.aclose_run("run-1") == 0
    cleanup.assert_called_once_with("run-1")
