"""Unit tests for NativeBrowserExecutor session routing and launch args.

Tests use mocked Playwright so no real browser is launched.
Safe to run in CI — no network, no subprocesses, no display required.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: executor with test-safe-mode on
# ---------------------------------------------------------------------------

@pytest.fixture()
def executor():
    os.environ["OPERON_TEST_SAFE_MODE"] = "true"
    from operon.executor.browser_native import NativeBrowserExecutor
    ex = NativeBrowserExecutor(headless=True)
    ex._sessions = {}
    return ex


def _fake_pw():
    """Minimal Playwright mock: playwright → chromium → browser → context → page."""
    page = AsyncMock()
    page.bring_to_front = AsyncMock()

    context = AsyncMock()
    context.new_page = AsyncMock(return_value=page)

    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)

    chromium = AsyncMock()
    chromium.launch = AsyncMock(return_value=browser)

    pw = AsyncMock()
    pw.chromium = chromium
    pw.start = AsyncMock(return_value=pw)
    pw.stop = AsyncMock()
    return pw


# ---------------------------------------------------------------------------
# _ensure_session routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_session_routes_observable(executor):
    """observable run must route to _ensure_session_observable."""
    executor._run_mode["obs"] = "observable"
    sentinel = MagicMock()

    with patch.object(executor, "_ensure_session_observable", new=AsyncMock(return_value=sentinel)) as obs, \
         patch.object(executor, "_ensure_session_batch", new=AsyncMock()) as batch:
        result = await executor._ensure_session("obs", foreground=False)

    obs.assert_awaited_once_with("obs", foreground=False)
    batch.assert_not_awaited()
    assert result is sentinel


@pytest.mark.asyncio
async def test_ensure_session_routes_batch_no_bm(executor):
    """batch run with no active BrowserManager uses _ensure_session_batch."""
    executor._run_mode["batch"] = "batch"
    sentinel = MagicMock()

    with patch.object(executor, "_ensure_session_observable", new=AsyncMock()) as obs, \
         patch.object(executor, "_ensure_session_batch", new=AsyncMock(return_value=sentinel)) as batch:
        result = await executor._ensure_session("batch", foreground=False)

    batch.assert_awaited_once_with("batch", foreground=False)
    obs.assert_not_awaited()
    assert result is sentinel


# ---------------------------------------------------------------------------
# Observable mode: Playwright launch args
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observable_launch_args_include_headless_new(executor):
    """observable mode must pass headless=False + --headless=new to chromium.launch."""
    executor._run_mode["obs"] = "observable"

    pw = _fake_pw()

    with patch("playwright.async_api.async_playwright", return_value=pw), \
         patch.object(executor, "_attach_page_listeners"), \
         patch.object(executor, "_close_other_sessions", new=AsyncMock()), \
         patch.object(executor, "_connect_browser_manager_to_port", new=AsyncMock()):
        await executor._ensure_session_observable("obs", foreground=False)

    pw.chromium.launch.assert_awaited_once()
    kwargs = pw.chromium.launch.call_args.kwargs
    assert kwargs.get("headless") is False, (
        "headless kwarg must be False to prevent Playwright injecting its own --headless flag"
    )
    args = kwargs.get("args", [])
    assert any("--headless=new" in a for a in args), (
        f"--headless=new missing from observable launch args: {args}"
    )
    assert any("--remote-debugging-port=9222" in a for a in args), (
        f"--remote-debugging-port=9222 missing from observable launch args: {args}"
    )


@pytest.mark.asyncio
async def test_observable_reuses_browser_on_second_run(executor):
    """Second observable run reuses _obs_browser without launching again."""
    executor._run_mode["obs1"] = "observable"
    executor._run_mode["obs2"] = "observable"

    pw = _fake_pw()

    with patch("playwright.async_api.async_playwright", return_value=pw), \
         patch.object(executor, "_attach_page_listeners"), \
         patch.object(executor, "_close_other_sessions", new=AsyncMock()), \
         patch.object(executor, "_connect_browser_manager_to_port", new=AsyncMock()):
        await executor._ensure_session_observable("obs1", foreground=False)
        await executor._ensure_session_observable("obs2", foreground=False)

    assert pw.chromium.launch.await_count == 1, (
        "chromium.launch must be called only once; second run reuses _obs_browser"
    )


# ---------------------------------------------------------------------------
# Batch mode: --headless=new must NOT appear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_launch_args_exclude_headless_new(executor):
    """batch mode must not inject --headless=new into Chromium launch args."""
    executor._run_mode["batch"] = "batch"

    pw = _fake_pw()

    with patch("playwright.async_api.async_playwright", return_value=pw), \
         patch.object(executor, "_attach_page_listeners"), \
         patch.object(executor, "_close_other_sessions", new=AsyncMock()), \
         patch.object(executor, "_video_dir_for_run", return_value=None), \
         patch.object(executor, "_chrome_process_ids", return_value=set()), \
         patch.object(executor, "_bring_browser_to_foreground", new=AsyncMock()):
        await executor._ensure_session_batch("batch", foreground=False)

    pw.chromium.launch.assert_awaited_once()
    args = pw.chromium.launch.call_args.kwargs.get("args", [])
    assert not any("--headless=new" in a for a in args), (
        f"--headless=new must not appear in batch launch args: {args}"
    )


# ---------------------------------------------------------------------------
# ensure_cdp_ready: batch early-return
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def testensure_cdp_ready_noop_for_batch():
    """ensure_cdp_ready(mode='batch') must return without touching BrowserManager.

    Batch tasks launch their own isolated Playwright Chromium and close it on
    completion, so they must not be routed through a shared CDP browser.
    """
    from operon.api.runtime import ensure_cdp_ready

    with patch("operon.browser.manager.BrowserManager") as mock_cls:
        await ensure_cdp_ready(mode="batch")

    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def testensure_cdp_ready_proceeds_for_observable():
    """ensure_cdp_ready(mode='observable') must not early-return."""
    from operon.api.runtime import ensure_cdp_ready

    # Fastest path through observable: already-connected manager → no Chrome launch
    mock_bm = MagicMock()
    mock_bm.is_connected = True

    with patch("operon.browser.manager.get_active_manager", return_value=mock_bm):
        await ensure_cdp_ready(mode="observable")
