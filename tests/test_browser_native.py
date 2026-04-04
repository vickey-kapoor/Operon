from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.executor.browser_native import NativeBrowserExecutor, _BrowserSession


@pytest.mark.asyncio
async def test_close_session_finalizes_single_recorded_video(tmp_path: Path) -> None:
    video_dir = tmp_path / "run-1" / "session_video"
    video_dir.mkdir(parents=True)
    raw_video = video_dir / "playwright-video.webm"
    raw_video.write_bytes(b"video")

    session = _BrowserSession(
        playwright=SimpleNamespace(stop=AsyncMock()),
        browser=SimpleNamespace(close=AsyncMock()),
        context=SimpleNamespace(close=AsyncMock()),
        page=object(),
        video_dir=video_dir,
    )
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True, record_video=True)

    await executor._close_session(session)

    assert not raw_video.exists()
    assert (video_dir / "session.webm").exists()
    session.context.close.assert_awaited_once()
    session.browser.close.assert_awaited_once()
    session.playwright.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_enables_recording_for_all_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        url = "about:blank"

    class FakeContext:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self):
            return self.page

    class FakeBrowser:
        def __init__(self) -> None:
            self.new_context = AsyncMock(side_effect=self._new_context)
            self.context_kwargs = None

        async def _new_context(self, **kwargs):
            self.context_kwargs = kwargs
            return FakeContext()

    class FakePlaywright:
        def __init__(self) -> None:
            self.browser = FakeBrowser()
            self.chromium = SimpleNamespace(launch=AsyncMock(return_value=self.browser))

    class FakeManager:
        def __init__(self) -> None:
            self.playwright = FakePlaywright()

        async def start(self):
            return self.playwright

    manager = FakeManager()
    fake_module = SimpleNamespace(async_playwright=lambda: manager)
    monkeypatch.setitem(__import__("sys").modules, "playwright.async_api", fake_module)

    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path,
        viewport_width=1440,
        viewport_height=900,
        headless=False,
    )

    session = await executor._ensure_session("run-123")

    assert session.video_dir == tmp_path / "run-123" / "session_video"
    assert session.video_dir.exists()
    assert manager.playwright.browser.context_kwargs["record_video_dir"] == str(session.video_dir)
    assert manager.playwright.browser.context_kwargs["record_video_size"] == {"width": 1440, "height": 900}


def test_recorded_video_path_prefers_finalized_session_name(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    video_dir = tmp_path / "run-xyz" / "session_video"
    video_dir.mkdir(parents=True)
    final_path = video_dir / "session.webm"
    final_path.write_bytes(b"video")

    assert executor.recorded_video_path_for_run("run-xyz") == final_path
