"""LAUNCH_APP succeeds only when the app shows up (a returned call is not proof)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from operon.models.common import FailureCategory
from operon.models.policy import ActionType, AgentAction
from tests.executor.test_desktop import _make_executor

D = "operon.executor.desktop"
ACTION = AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad")


def _executor():
    executor = _make_executor()
    executor._post_launch_delay = 0
    executor._capture_after = AsyncMock(return_value="after.png")
    return executor


def _proc(code):
    proc = MagicMock()
    proc.poll.return_value = code
    return proc


async def _run(executor, *, windows, foreground=(1,), proc=None, pids=frozenset()):
    """windows: successive _visible_windows results (first is "before")."""
    clock = SimpleNamespace(now=0.0)

    async def fake_sleep(seconds):
        clock.now += seconds

    with (
        patch(f"{D}._visible_windows", side_effect=list(windows) + [windows[-1]] * 200),
        patch(f"{D}._foreground_window", side_effect=list(foreground) + [foreground[-1]] * 200),
        patch(f"{D}._window_owner", return_value=(777, "Untitled - Notepad")),
        patch(f"{D}._image_pids", return_value=pids),
        patch(f"{D}.subprocess.Popen", return_value=proc or _proc(None)),
        patch(f"{D}.time.monotonic", side_effect=lambda: clock.now),
        patch(f"{D}.asyncio.sleep", side_effect=fake_sleep),
    ):
        result = await executor._exec_launch_app(ACTION)
    return result, clock.now


@pytest.mark.asyncio
async def test_new_window_means_success() -> None:
    result, _ = await _run(_executor(), windows=[frozenset({1}), frozenset({1}), frozenset({1, 2})])
    assert result.success is True


@pytest.mark.asyncio
async def test_exit_nonzero_with_no_window_fails() -> None:
    result, _ = await _run(_executor(), windows=[frozenset({1})], proc=_proc(1))
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert "exited with code 1" in result.detail and "notepad.exe" in result.detail


@pytest.mark.asyncio
async def test_exit_nonzero_but_window_appears_later_is_success() -> None:
    # explorer.exe exits 1 when the running shell takes over; the window comes later.
    result, _ = await _run(
        _executor(),
        windows=[frozenset({1}), frozenset({1}), frozenset({1}), frozenset({1}), frozenset({1, 2})],
        proc=_proc(1),
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_exit_nonzero_waits_the_grace_period_then_fails() -> None:
    result, elapsed = await _run(_executor(), windows=[frozenset({1})], proc=_proc(1))
    assert result.success is False
    assert 2.5 <= elapsed < 3.0


@pytest.mark.asyncio
async def test_no_window_with_launcher_not_failed_is_success_with_note_at_the_bound() -> None:
    result, elapsed = await _run(_executor(), windows=[frozenset({1})])
    assert result.success is True
    assert "no new window detected" in result.detail
    assert 8.0 <= elapsed < 8.5


@pytest.mark.asyncio
async def test_missing_executable_fails_before_launching() -> None:
    executor = _executor()
    with (
        patch(f"{D}._executable_missing", return_value=True),
        patch(f"{D}.subprocess.Popen") as popen,
    ):
        result = await executor._exec_launch_app(ACTION)
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert "'notepad.exe' is not installed" in result.detail
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_already_running_app_activated_is_success() -> None:
    # No new window, but the foreground moved to an existing window of the app.
    result, _ = await _run(
        _executor(), windows=[frozenset({1, 5})], foreground=(1, 5), pids=frozenset({777})
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_unlistable_windows_keeps_trusting_the_call() -> None:
    executor = _executor()
    with patch(f"{D}._visible_windows", return_value=None), patch(f"{D}.subprocess.Popen"):
        result = await executor._exec_launch_app(ACTION)
    assert result.success is True


@pytest.mark.asyncio
async def test_linux_process_exiting_with_error_fails() -> None:
    from operon.executor.linux_desktop import LinuxDesktopExecutor

    with patch(f"{D}._set_dpi_awareness"):
        executor = LinuxDesktopExecutor(artifact_dir=_make_executor()._artifact_dir)
    executor._post_launch_delay = 0
    executor._capture_after = AsyncMock(return_value="after.png")
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="terminal")
    with patch("operon.executor.linux_desktop.subprocess.Popen", return_value=_proc(127)):
        result = await executor._exec_launch_app(action)
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    with patch("operon.executor.linux_desktop.subprocess.Popen", return_value=_proc(None)):
        assert (await executor._exec_launch_app(action)).success is True


def _fake_winreg(entries: dict[str, str]):
    """A winreg stand-in: App Paths values by exe name, under HKLM only."""
    import types

    def open_key(hive, path):
        name = path.rsplit("\\", 1)[-1].lower()
        if hive == "HKLM" and name in entries:
            return MagicMock(__enter__=lambda self: name, __exit__=lambda *a: False)
        raise FileNotFoundError(path)

    return types.SimpleNamespace(
        HKEY_CURRENT_USER="HKCU",
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=open_key,
        QueryValueEx=lambda key, _name: (entries[key], 1),
    )


def test_app_path_lookup_and_missing_check() -> None:
    from operon.executor import desktop
    from tests.executor.conftest import REAL_EXECUTABLE_MISSING as missing

    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    fake = _fake_winreg({"chrome.exe": chrome})
    with (
        patch.object(desktop.sys, "platform", "win32"),
        patch.dict("sys.modules", {"winreg": fake}),
        patch.object(desktop.Path, "exists", return_value=True),
        patch(f"{D}.shutil.which", return_value=None),
    ):
        assert desktop._app_path("chrome") == chrome
        assert missing("chrome") is False  # found under App Paths
        assert desktop._app_path("nosuchapp") is None
        assert missing("nosuchapp") is True  # neither PATH nor App Paths
        assert desktop._resolve_launch_command("chrome") == f'"{chrome}"'
        assert desktop._resolve_launch_command("chrome --new-window") == f'"{chrome}" --new-window'


@pytest.mark.skipif(os.name != "nt", reason="%VAR% expansion is Windows ntpath behaviour")
def test_app_path_expands_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.executor import desktop

    monkeypatch.setenv("OPERON_TEST_PF", r"C:\Apps")
    fake = _fake_winreg({"tool.exe": r"%OPERON_TEST_PF%\tool.exe"})
    with (
        patch.object(desktop.sys, "platform", "win32"),
        patch.dict("sys.modules", {"winreg": fake}),
        patch.object(desktop.Path, "exists", return_value=True),
    ):
        assert desktop._app_path("tool") == r"C:\Apps\tool.exe"


def test_on_path_commands_are_left_alone() -> None:
    from operon.executor import desktop
    from tests.executor.conftest import REAL_EXECUTABLE_MISSING as missing

    with patch.object(desktop.sys, "platform", "win32"), patch(f"{D}.shutil.which", return_value="C:/w/notepad.exe"):
        assert desktop._resolve_launch_command("notepad.exe") == "notepad.exe"
        assert missing("notepad.exe") is False


@pytest.mark.asyncio
async def test_app_paths_program_is_launched_by_its_resolved_path() -> None:
    executor = _executor()
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="chrome")
    chrome = r"C:\Chrome\chrome.exe"
    with (
        patch(f"{D}._visible_windows", return_value=None),
        patch(f"{D}._resolve_launch_command", return_value=chrome),
        patch(f"{D}.subprocess.Popen") as popen,
    ):
        result = await executor._exec_launch_app(action)
    assert result.success is True
    assert popen.call_args[0][0] == chrome
