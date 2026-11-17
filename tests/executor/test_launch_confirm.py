"""LAUNCH_APP succeeds only when the app shows up (a returned call is not proof)."""

from __future__ import annotations

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
async def test_missing_executable_exits_nonzero_and_fails() -> None:
    result, _ = await _run(_executor(), windows=[frozenset({1})], proc=_proc(1))
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert "exited with code 1" in result.detail and "notepad.exe" in result.detail


@pytest.mark.asyncio
async def test_no_window_times_out_within_the_bound() -> None:
    result, elapsed = await _run(_executor(), windows=[frozenset({1})])
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    assert "no window appeared within 8 s" in result.detail
    assert 8.0 <= elapsed < 8.5


@pytest.mark.asyncio
async def test_already_running_app_activated_is_success() -> None:
    # No new window, but the foreground moved to an existing window of the app.
    result, _ = await _run(
        _executor(), windows=[frozenset({1, 5})], foreground=(1, 5), pids=frozenset({777})
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_launcher_exit_zero_without_window_still_waits_then_fails() -> None:
    result, elapsed = await _run(_executor(), windows=[frozenset({1})], proc=_proc(0))
    assert result.success is False
    assert elapsed >= 8.0


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
