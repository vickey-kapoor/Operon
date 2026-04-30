from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.executor.browser_native import NativeBrowserExecutor, _BrowserSession
from src.executor.browserbase_native import (
    BrowserbaseNativeBrowserExecutor,
    _is_session_closed,
)
from src.models.common import FailureCategory
from src.models.policy import ActionType, AgentAction


def _install_browserbase_modules(monkeypatch: pytest.MonkeyPatch):
    fake_page = SimpleNamespace()
    fake_context = SimpleNamespace(pages=[fake_page], new_page=AsyncMock(return_value=fake_page))
    fake_browser = SimpleNamespace(contexts=[fake_context], new_context=AsyncMock(return_value=fake_context))
    fake_playwright = SimpleNamespace(
        chromium=SimpleNamespace(connect_over_cdp=AsyncMock(return_value=fake_browser)),
        stop=AsyncMock(),
    )

    class FakeManager:
        async def start(self):
            return fake_playwright

    sessions = SimpleNamespace(
        create=MagicMock(return_value=SimpleNamespace(id="bb-session-1")),
        debug=MagicMock(return_value=SimpleNamespace(ws_url="wss://browserbase.test/devtools")),
        update=MagicMock(),
    )
    client = SimpleNamespace(sessions=sessions)

    browserbase_module = ModuleType("browserbase")
    browserbase_module.Browserbase = lambda api_key: client
    playwright_module = ModuleType("playwright.async_api")
    playwright_module.async_playwright = lambda: FakeManager()

    monkeypatch.setitem(sys.modules, "browserbase", browserbase_module)
    monkeypatch.setitem(sys.modules, "playwright.async_api", playwright_module)
    return sessions, fake_playwright, fake_browser, fake_context, fake_page


def test_is_session_closed_detects_known_signals() -> None:
    class TargetClosedError(Exception):
        pass

    assert _is_session_closed(TargetClosedError("boom")) is True
    assert _is_session_closed(RuntimeError("Target page, context or browser has been closed")) is True
    assert _is_session_closed(RuntimeError("socket timeout")) is False


@pytest.mark.asyncio
async def test_ensure_session_connects_and_tracks_browserbase_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, fake_playwright, fake_browser, fake_context, fake_page = _install_browserbase_modules(monkeypatch)

    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="bb-project",
        headless=True,
    )

    session = await executor._ensure_session("run-1")

    sessions.create.assert_called_once_with(project_id="bb-project")
    sessions.debug.assert_called_once_with("bb-session-1")
    fake_playwright.chromium.connect_over_cdp.assert_awaited_once_with("wss://browserbase.test/devtools")
    assert executor._bb_session_ids["run-1"] == "bb-session-1"
    assert session.browser is fake_browser
    assert session.context is fake_context
    assert session.page is fake_page


@pytest.mark.asyncio
async def test_ensure_session_requires_browserbase_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_browserbase_modules(monkeypatch)
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)

    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="",
        project_id="bb-project",
        headless=True,
    )

    with pytest.raises(RuntimeError, match="BROWSERBASE_API_KEY"):
        await executor._ensure_session("run-missing-key")


@pytest.mark.asyncio
async def test_ensure_session_requires_browserbase_project_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_browserbase_modules(monkeypatch)
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)

    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="",
        headless=True,
    )

    with pytest.raises(RuntimeError, match="BROWSERBASE_PROJECT_ID"):
        await executor._ensure_session("run-missing-project")


@pytest.mark.asyncio
async def test_capture_eviction_returns_blank_frame_on_closed_session(tmp_path: Path) -> None:
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="bb-project",
        headless=True,
    )
    executor._current_run_id = "run-closed"
    dead_session = _BrowserSession(
        playwright=SimpleNamespace(stop=AsyncMock()),
        browser=SimpleNamespace(close=AsyncMock()),
        context=SimpleNamespace(close=AsyncMock()),
        page=object(),
        video_dir=None,
        browser_pid=None,
    )
    executor._sessions["run-closed"] = dead_session
    executor._bb_session_ids["run-closed"] = "bb-session-closed"

    def fake_stop(run_id: str) -> None:
        executor._bb_session_ids.pop(run_id, None)

    executor._stop_bb_session = fake_stop  # type: ignore[method-assign]

    with patch.object(NativeBrowserExecutor, "capture", AsyncMock(side_effect=RuntimeError("Target closed"))):
        frame = await executor.capture()

    await asyncio.sleep(0)
    assert frame.artifact_path.endswith(".png")
    assert Path(frame.artifact_path).exists()
    assert "run-closed" not in executor._sessions
    dead_session.playwright.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_eviction_returns_failure_on_closed_session(tmp_path: Path) -> None:
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="bb-project",
        headless=True,
    )
    executor._current_run_id = "run-execute"
    dead_session = _BrowserSession(
        playwright=SimpleNamespace(stop=AsyncMock()),
        browser=SimpleNamespace(close=AsyncMock()),
        context=SimpleNamespace(close=AsyncMock()),
        page=object(),
        video_dir=None,
        browser_pid=None,
    )
    executor._sessions["run-execute"] = dead_session
    executor._bb_session_ids["run-execute"] = "bb-session-execute"

    def fake_stop(run_id: str) -> None:
        executor._bb_session_ids.pop(run_id, None)

    executor._stop_bb_session = fake_stop  # type: ignore[method-assign]

    with patch.object(
        NativeBrowserExecutor,
        "execute",
        AsyncMock(side_effect=RuntimeError("Target page, context or browser has been closed")),
    ):
        result = await executor.execute(AgentAction(action_type=ActionType.CLICK, x=10, y=20))

    await asyncio.sleep(0)
    assert result.success is False
    assert result.failure_category is FailureCategory.EXECUTION_ERROR
    assert "reconnecting on next step" in result.detail
    assert "run-execute" not in executor._sessions
    dead_session.playwright.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_run_releases_remote_session(tmp_path: Path) -> None:
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="bb-project",
        headless=True,
    )
    released: list[str] = []

    def fake_stop(run_id: str) -> None:
        released.append(run_id)

    executor._stop_bb_session = fake_stop  # type: ignore[method-assign]

    with patch.object(NativeBrowserExecutor, "aclose_run", AsyncMock(return_value=1)) as mock_close:
        result = await executor.aclose_run("run-close")

    assert result == 1
    mock_close.assert_awaited_once_with("run-close")
    assert released == ["run-close"]


def test_cleanup_run_releases_remote_session(tmp_path: Path) -> None:
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key="bb-key",
        project_id="bb-project",
        headless=True,
    )
    released: list[str] = []

    def fake_stop(run_id: str) -> None:
        released.append(run_id)

    executor._stop_bb_session = fake_stop  # type: ignore[method-assign]

    with patch.object(NativeBrowserExecutor, "cleanup_run", return_value=1) as mock_cleanup:
        result = executor.cleanup_run("run-cleanup")

    assert result == 1
    mock_cleanup.assert_called_once_with("run-cleanup")
    assert released == ["run-cleanup"]
