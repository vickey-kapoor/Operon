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
         patch.object(executor, "_attach_live_view", new=AsyncMock()):
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
         patch.object(executor, "_attach_live_view", new=AsyncMock()):
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
async def test_ensure_cdp_ready_launches_no_browser_in_observable_mode():
    """Observable mode has one browser, the executor's: ensure_cdp_ready must not
    launch Chrome or attach the live view to anything (G5)."""
    from operon.api.runtime import ensure_cdp_ready

    with patch("subprocess.Popen") as popen, \
         patch("operon.browser.manager.BrowserManager") as mock_cls:
        await ensure_cdp_ready(mode="observable")

    popen.assert_not_called()
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_each_observable_run_points_the_live_view_at_its_own_page(executor):
    executor._run_mode["obs1"] = "observable"
    executor._run_mode["obs2"] = "observable"
    pw = _fake_pw()
    attached = []

    async def attach(page, port):
        attached.append((page, port))

    with patch("playwright.async_api.async_playwright", return_value=pw), \
         patch.object(executor, "_attach_page_listeners"), \
         patch.object(executor, "_close_other_sessions", new=AsyncMock()), \
         patch.object(executor, "_attach_live_view", new=attach):
        first = await executor._ensure_session_observable("obs1", foreground=False)
        second = await executor._ensure_session_observable("obs2", foreground=False)

    assert attached == [(first.page, 9222), (second.page, 9222)]


@pytest.mark.asyncio
async def test_the_live_view_follows_the_page_with_the_runs_target():
    from operon.browser import manager as manager_mod

    manager = manager_mod.BrowserManager()
    other, ours = MagicMock(name="other"), MagicMock(name="ours")
    ours.viewport_size = {"width": 1280, "height": 800}
    manager._browser = MagicMock()
    manager._browser.contexts = [MagicMock(pages=[other]), MagicMock(pages=[ours])]
    targets = {id(other): "T-other", id(ours): "T-ours"}

    async def target_id(page):
        return targets[id(page)]

    with patch.object(manager_mod, "_target_id", new=target_id), \
         patch.object(manager, "start_screencast", new=AsyncMock()), \
         patch.object(manager, "stop_screencast", new=AsyncMock()):
        assert await manager.follow_target("T-ours") is True
        assert manager._page is ours and manager._viewport_w == 1280
        assert await manager.follow_target("T-missing") is False
        assert manager._page is ours


def _session(page):
    from operon.executor.browser_native import _BrowserSession

    session = _BrowserSession.__new__(_BrowserSession)
    session.page = page
    session.context = MagicMock(pages=[page])
    return session


@pytest.mark.asyncio
async def test_the_live_view_follows_a_tab_the_click_opened(executor):
    first, new_tab = AsyncMock(name="first"), AsyncMock(name="new_tab")
    executor._run_mode["obs1"] = "observable"
    executor._sessions["obs1"] = _session(first)
    executor._sessions["obs1"].context.pages = [first, new_tab]
    executor.set_current_run_id("obs1")

    with patch.object(executor, "_attach_live_view", new=AsyncMock()) as attach:
        await executor._adopt_new_tab_if_opened()
        await executor._adopt_new_tab_if_opened()  # same tab again: no second switch

    assert executor._sessions["obs1"].page is new_tab
    attach.assert_awaited_once_with(new_tab, 9222)


@pytest.mark.asyncio
async def test_the_live_view_follows_a_late_popup(executor):
    import asyncio

    page, popup = MagicMock(name="page"), MagicMock(name="popup")
    handlers = {}
    page.on = lambda event, fn: handlers.setdefault(event, fn)
    popup.on = MagicMock()
    executor._run_mode["obs1"] = "observable"
    executor._sessions["obs1"] = _session(page)
    executor._attach_page_listeners(page, "obs1")

    with patch.object(executor, "_attach_live_view", new=AsyncMock()) as attach:
        handlers["popup"](popup)
        await asyncio.gather(*executor._live_view_tasks)

    assert executor._sessions["obs1"].page is popup
    attach.assert_awaited_once_with(popup, 9222)


@pytest.mark.asyncio
async def test_a_batch_run_never_touches_the_live_view(executor):
    first, new_tab = AsyncMock(name="first"), AsyncMock(name="new_tab")
    executor._run_mode["b1"] = "batch"
    executor._sessions["b1"] = _session(first)
    executor._sessions["b1"].context.pages = [first, new_tab]
    executor.set_current_run_id("b1")

    with patch.object(executor, "_attach_live_view", new=AsyncMock()) as attach:
        await executor._adopt_new_tab_if_opened()

    attach.assert_not_called()


@pytest.mark.asyncio
async def test_two_quick_popups_leave_the_live_view_on_the_newest_tab(executor):
    import asyncio

    page, older, newer = MagicMock(name="page"), MagicMock(name="older"), MagicMock(name="newer")
    handlers = {}
    page.on = lambda event, fn: handlers.setdefault(event, fn)
    older.on = lambda event, fn: handlers.setdefault(("older", event), fn)
    newer.on = MagicMock()
    executor._run_mode["obs1"] = "observable"
    executor._sessions["obs1"] = _session(page)
    executor._attach_page_listeners(page, "obs1")
    followed = []

    async def slow_attach(target, port):
        await asyncio.sleep(0.01)  # e.g. waiting in stop_screencast
        followed.append(target)

    with patch.object(executor, "_attach_live_view", new=slow_attach):
        handlers["popup"](older)
        handlers[("older", "popup")](newer)
        await asyncio.gather(*list(executor._live_view_tasks))

    assert executor._sessions["obs1"].page is newer
    assert followed[-1] is newer
    assert older not in followed


@pytest.mark.asyncio
async def test_an_observable_run_records_its_session_video(executor, tmp_path):
    """Streaming the browser must not cost the run its post-run recording."""
    executor._run_mode["obs"] = "observable"
    executor._artifact_dir = tmp_path
    pw = _fake_pw()

    with patch("playwright.async_api.async_playwright", return_value=pw), \
         patch.object(executor, "_attach_page_listeners"), \
         patch.object(executor, "_close_other_sessions", new=AsyncMock()), \
         patch.object(executor, "_attach_live_view", new=AsyncMock()):
        session = await executor._ensure_session_observable("obs", foreground=False)

    browser = pw.chromium.launch.return_value
    kwargs = browser.new_context.call_args.kwargs
    assert kwargs["record_video_dir"] == str(tmp_path / "obs" / "session_video")
    assert session.video_dir == tmp_path / "obs" / "session_video"


@pytest.mark.asyncio
async def test_closing_an_observable_run_stops_the_screencast(executor):
    """Its last frame must not reach the next viewer, e.g. a desktop run's page."""
    from operon.browser import manager as manager_mod

    manager = MagicMock()
    manager.stop_screencast = AsyncMock()
    session = _session(AsyncMock())
    session.owns_context = True
    session.context = AsyncMock()
    session.video_dir = None
    session.browser = None
    session.playwright = None

    with patch.object(manager_mod, "get_active_manager", return_value=manager):
        await executor._close_session(session, observable=True)
        manager.stop_screencast.assert_awaited_once()
        await executor._close_session(session, observable=False)
        manager.stop_screencast.assert_awaited_once()  # batch sessions leave it alone
