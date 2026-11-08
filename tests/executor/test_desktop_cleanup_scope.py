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


@pytest.mark.asyncio
async def test_cleanup_kills_only_processes_started_after_the_launch() -> None:
    executor = _make_executor()
    executor._post_launch_delay = 0
    executor._capture_after = AsyncMock(return_value="after.png")
    executor.set_current_run_id("run-1")

    # The user already has Notepad open as pid 100.
    with (
        patch("operon.executor.desktop._image_pids", return_value=frozenset({100})),
        patch("operon.executor.desktop.subprocess.Popen"),
    ):
        result = await executor._exec_launch_app(AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad"))
    assert result.success is True
    executor._launched_processes.clear()  # only the image-name fallback is under test

    kills: list[list[str]] = []

    def fake_run(args, **kwargs):
        kills.append(args)
        return SimpleNamespace(returncode=0, stdout="")

    with (
        patch("operon.executor.desktop._image_pids", return_value=frozenset({100, 200})),
        patch("operon.executor.desktop.subprocess.run", side_effect=fake_run),
    ):
        closed = executor.cleanup_run("run-1")

    assert kills == [["taskkill", "/PID", "200", "/F"]]
    assert closed == 1
    assert all("/IM" not in args for args in kills)


def test_cleanup_kills_nothing_by_name_when_processes_cannot_be_listed() -> None:
    executor = _make_executor()
    executor._launched_app_names["run-1"] = ["chrome.exe"]
    executor._preexisting_pids["run-1"] = {"chrome.exe": None}

    with patch("operon.executor.desktop.subprocess.run") as run:
        closed = executor.cleanup_run("run-1")

    assert closed == 0
    run.assert_not_called()
