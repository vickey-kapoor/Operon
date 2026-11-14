"""Observable runs in the built-in browser (KAN-138): a tab of the saved-profile browser."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from operon.executor import browser_native
from operon.executor.browser_native import NativeBrowserExecutor


class FakeContext:
    pass


class FakePage:
    def __init__(self) -> None:
        self.context = FakeContext()
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def on(self, *_: object) -> None:
        return None


@pytest.fixture
def executor(tmp_path, monkeypatch):
    page = FakePage()
    fake_browser = type("B", (), {})()
    fake_browser.new_task_page = AsyncMock(return_value=page)
    fake_browser.release_run = AsyncMock()
    fake_browser.set_active = lambda p: None
    fake_browser.own_page = lambda p, run_id: None
    monkeypatch.setattr(browser_native, "get_builtin_browser", lambda: fake_browser)
    monkeypatch.setattr(browser_native, "builtin_browser_enabled", lambda: True)
    ex = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    ex._attach_page_listeners = lambda *_: None
    ex._attach_live_view = AsyncMock()
    ex._stop_live_view = AsyncMock()
    ex.fake_browser, ex.page = fake_browser, page
    return ex


async def test_an_observable_run_opens_a_tab_in_the_built_in_browser(executor) -> None:
    executor.configure_run("run-1", mode="observable", start_url="https://shop.example/start")

    session = await executor._ensure_session("run-1")

    executor.fake_browser.new_task_page.assert_awaited_once_with("run-1", "https://shop.example/start")
    assert session.page is executor.page and session.owns_context is False
    executor._attach_live_view.assert_awaited_once_with(executor.page, 9222)


async def test_ending_the_run_releases_its_tabs_and_launches_no_second_browser(executor) -> None:
    executor.configure_run("run-1", mode="observable")
    await executor._ensure_session("run-1")

    assert await executor.aclose_run("run-1") == 1

    executor.fake_browser.release_run.assert_awaited_once_with("run-1")
    assert executor._obs_browser is None


async def test_a_users_tab_is_never_adopted_as_the_runs_new_tab(executor) -> None:
    executor.configure_run("run-1", mode="observable")
    session = await executor._ensure_session("run-1")
    executor.set_current_run_id("run-1")
    users_tab = FakePage()
    session.context.pages = [executor.page, users_tab]  # the shared context holds both
    executor.fake_browser.pages_of_run = lambda run_id: [executor.page]

    await executor._adopt_new_tab_if_opened()

    assert session.page is executor.page  # only the run's own pages count


async def test_batch_runs_never_use_the_built_in_browser(executor, monkeypatch) -> None:
    opened = AsyncMock()
    monkeypatch.setattr(executor, "_ensure_session_batch", opened)
    executor.configure_run("run-1", mode="batch")
    await executor._open_session("run-1", foreground=True)
    opened.assert_awaited_once()
    executor.fake_browser.new_task_page.assert_not_awaited()
