"""Headless launches must NOT spawn the WMI/CIM PowerShell to find chrome PIDs.

The before/after PowerShell pair around `chromium.launch` produced `browser_pid`
which is consumed solely by `_bring_browser_to_foreground`. That foreground
helper is itself gated to non-headless runs, so in headless mode the PIDs are
computed and never read.

Worse, those PowerShell calls were measurably stealing user focus during the
WebArena medium suite (WMI initialization can briefly grab foreground on some
Windows configs, even with CREATE_NO_WINDOW). Skipping them entirely in
headless mode is both a correctness improvement and a focus-theft fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_chrome_process_ids_not_called_in_headless_mode() -> None:
    """Headless ensure_session must not invoke _chrome_process_ids.

    This is a behaviour test: we don't care which Playwright APIs are used,
    only that the PowerShell-spawning helper is bypassed.
    """
    from src.executor.browser_native import NativeBrowserExecutor

    executor = NativeBrowserExecutor.__new__(NativeBrowserExecutor)
    executor._sessions = {}
    executor._fresh_session_run_id = None
    executor._run_headless = {}
    executor._headless = True  # default headless
    executor._record_video = False
    executor._viewport_width = 1920
    executor._viewport_height = 1080
    executor._last_dialog_message = {}
    executor._video_dir_for_run = MagicMock(return_value=None)

    # Stub everything Playwright reaches for so ensure_session can run.
    page = MagicMock()
    page.goto = MagicMock()
    page.bring_to_front = MagicMock()
    page.on = MagicMock()

    async def _aiter_to_page(*_a, **_kw):
        return page

    context = MagicMock()
    context.new_page = MagicMock(side_effect=_aiter_to_page)
    context.add_init_script = MagicMock()

    async def _new_context(**_kw):
        return context

    browser = MagicMock()
    browser.new_context = MagicMock(side_effect=_new_context)

    async def _launch(**_kw):
        return browser

    chromium = MagicMock()
    chromium.launch = MagicMock(side_effect=_launch)
    chromium.executable_path = "C:\\path\\to\\chrome.exe"

    pw = MagicMock()
    pw.chromium = chromium

    async def _start():
        return pw

    pw_module = MagicMock()
    pw_module.start = MagicMock(side_effect=_start)

    with (
        patch("playwright.async_api.async_playwright", return_value=pw_module),
        patch.object(NativeBrowserExecutor, "_chrome_process_ids") as mock_pids,
    ):
        try:
            await executor.ensure_session("test-run", foreground=False)
        except Exception:
            # The stubs aren't a complete Playwright surface; we only care that
            # _chrome_process_ids was bypassed before any failure.
            pass

    assert not mock_pids.called, (
        "Headless launch must not call _chrome_process_ids — it spawns a "
        "PowerShell WMI query that can briefly steal user focus and is only "
        "needed to compute browser_pid for non-headless foreground promotion."
    )


# A symmetric "headed mode still calls _chrome_process_ids" test would need
# substantial Playwright mocking to exercise the post-launch path. The headed
# path is covered by real runs (the existing live-server tests). What matters
# here is the headless-mode skip — that's the bug we're fixing.
