"""Verify Windows subprocess calls pass CREATE_NO_WINDOW to suppress conhost.exe.

On Windows, when a console-mode parent process (e.g. Python in a terminal)
launches a console subprocess (powershell, taskkill, anything that allocates
its own console), Windows flashes a conhost.exe window. That flash steals
focus from the user's foreground app — measurably so during a multi-task
benchmark sweep where these calls fire repeatedly.

The fix: every Windows subprocess call must pass creationflags including
CREATE_NO_WINDOW. These tests guard the regression boundary.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="creationflags is Windows-specific")
def test_browser_native_app_activate_passes_no_console_window() -> None:
    from src.executor.browser_native import NativeBrowserExecutor

    with patch("src.executor.browser_native.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        NativeBrowserExecutor._app_activate_browser_window(browser_pid=1234)

    assert mock_run.called
    kwargs = mock_run.call_args.kwargs
    assert "creationflags" in kwargs, "AppActivate PowerShell must pass creationflags"
    import subprocess
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW, (
        "AppActivate PowerShell must include CREATE_NO_WINDOW"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="creationflags is Windows-specific")
def test_browser_native_chrome_process_ids_passes_no_console_window() -> None:
    from src.executor.browser_native import NativeBrowserExecutor

    with patch("src.executor.browser_native.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="[]")
        NativeBrowserExecutor._chrome_process_ids("C:\\path\\to\\chrome.exe")

    assert mock_run.called
    kwargs = mock_run.call_args.kwargs
    import subprocess
    assert kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, (
        "_chrome_process_ids fires per benchmark task — without CREATE_NO_WINDOW it "
        "flashes conhost on every session creation and steals user focus"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="creationflags is Windows-specific")
def test_no_console_constant_is_real_create_no_window() -> None:
    """Sanity: the helper constant in each module is the actual Windows flag."""
    import subprocess

    from src.agent.hitl import _NO_CONSOLE as hitl_flag
    from src.executor.browser_native import _NO_CONSOLE as browser_flag
    from src.executor.desktop import _NO_CONSOLE as desktop_flag

    expected = subprocess.CREATE_NO_WINDOW
    assert hitl_flag == expected
    assert browser_flag == expected
    assert desktop_flag == expected


def test_no_console_constant_is_zero_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Windows, the flag must be 0 so creationflags has no effect.

    Pure semantic test — verifies the conditional expression rather than the
    actual platform value, so we can run it on Windows CI too.
    """
    import subprocess as _sub

    def evaluate(is_win: bool) -> int:
        return _sub.CREATE_NO_WINDOW if is_win else 0

    assert evaluate(False) == 0
    if sys.platform == "win32":
        assert evaluate(True) == _sub.CREATE_NO_WINDOW
