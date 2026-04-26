from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.executor.browser_native import NativeBrowserExecutor, _BrowserSession
from src.models.policy import ActionType, AgentAction


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
    monkeypatch.delenv("OPERON_TEST_SAFE_MODE", raising=False)

    class FakePage:
        url = "about:blank"

        def __init__(self) -> None:
            self.bring_to_front = AsyncMock()
            self.goto = AsyncMock()
            self.evaluate = AsyncMock()
            self.keyboard = SimpleNamespace(press=AsyncMock())
            self.wait_for_load_state = AsyncMock()
            self.on = lambda *_args, **_kwargs: None

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
            self.chromium = SimpleNamespace(
                launch=AsyncMock(return_value=self.browser),
                executable_path="C:\\playwright\\chrome.exe",
            )

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
        viewport_width=1920,
        viewport_height=1080,
        headless=False,
    )
    executor._bring_browser_to_foreground = AsyncMock()
    executor._detect_browser_pid = lambda *_args: 4321

    session = await executor._ensure_session("run-123")

    assert session.video_dir == tmp_path / "run-123" / "session_video"
    assert session.browser_pid == 4321
    assert session.video_dir.exists()
    assert manager.playwright.browser.context_kwargs["viewport"] == {"width": 1920, "height": 1080}
    assert manager.playwright.browser.context_kwargs["record_video_dir"] == str(session.video_dir)
    assert manager.playwright.browser.context_kwargs["record_video_size"] == {"width": 1920, "height": 1080}
    manager.playwright.chromium.launch.assert_awaited_once_with(
        headless=False,
        args=["--window-size=1920,1080", "--window-position=0,0"],
    )
    session.page.bring_to_front.assert_awaited_once()
    session.page.keyboard.press.assert_awaited_once_with("Control+0")
    executor._bring_browser_to_foreground.assert_awaited_once_with(4321)


@pytest.mark.asyncio
async def test_ensure_session_uses_per_run_headless_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        url = "about:blank"

        def __init__(self) -> None:
            self.bring_to_front = AsyncMock()
            self.goto = AsyncMock()
            self.evaluate = AsyncMock()
            self.keyboard = SimpleNamespace(press=AsyncMock())
            self.wait_for_load_state = AsyncMock()
            self.on = lambda *_args, **_kwargs: None

    class FakeContext:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self):
            return self.page

    class FakeBrowser:
        def __init__(self) -> None:
            self.new_context = AsyncMock(return_value=FakeContext())

    class FakePlaywright:
        def __init__(self) -> None:
            self.browser = FakeBrowser()
            self.chromium = SimpleNamespace(
                launch=AsyncMock(return_value=self.browser),
                executable_path="C:\\playwright\\chrome.exe",
            )

    class FakeManager:
        def __init__(self) -> None:
            self.playwright = FakePlaywright()

        async def start(self):
            return self.playwright

    manager = FakeManager()
    fake_module = SimpleNamespace(async_playwright=lambda: manager)
    monkeypatch.setitem(__import__("sys").modules, "playwright.async_api", fake_module)

    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False, viewport_width=1440, viewport_height=900)
    executor.configure_run("run-override", headless=True)

    await executor._ensure_session("run-override")

    manager.playwright.chromium.launch.assert_awaited_once_with(
        headless=True,
        args=["--window-size=1440,900", "--window-position=0,0"],
    )


@pytest.mark.asyncio
async def test_ensure_session_closes_stale_sessions_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPERON_TEST_SAFE_MODE", raising=False)

    class FakePage:
        url = "about:blank"

        def __init__(self) -> None:
            self.bring_to_front = AsyncMock()
            self.goto = AsyncMock()
            self.evaluate = AsyncMock()
            self.keyboard = SimpleNamespace(press=AsyncMock())
            self.wait_for_load_state = AsyncMock()
            self.on = lambda *_args, **_kwargs: None

    class FakeContext:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self):
            return self.page

    class FakeBrowser:
        def __init__(self) -> None:
            self.new_context = AsyncMock(return_value=FakeContext())

    class FakePlaywright:
        def __init__(self) -> None:
            self.browser = FakeBrowser()
            self.chromium = SimpleNamespace(
                launch=AsyncMock(return_value=self.browser),
                executable_path="C:\\playwright\\chrome.exe",
            )

    class FakeManager:
        def __init__(self) -> None:
            self.playwright = FakePlaywright()

        async def start(self):
            return self.playwright

    manager = FakeManager()
    fake_module = SimpleNamespace(async_playwright=lambda: manager)
    monkeypatch.setitem(__import__("sys").modules, "playwright.async_api", fake_module)

    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False)
    stale_session = _BrowserSession(
        playwright=SimpleNamespace(stop=AsyncMock()),
        browser=SimpleNamespace(close=AsyncMock()),
        context=SimpleNamespace(close=AsyncMock()),
        page=object(),
        video_dir=None,
        browser_pid=None,
    )
    executor._sessions["run-old"] = stale_session
    executor.configure_run("run-old", headless=False)
    executor._bring_browser_to_foreground = AsyncMock()
    executor._detect_browser_pid = lambda *_args: 99

    await executor._ensure_session("run-new")

    assert "run-old" not in executor._sessions
    stale_session.context.close.assert_awaited_once()
    stale_session.browser.close.assert_awaited_once()
    stale_session.playwright.stop.assert_awaited_once()
    assert "run-new" in executor._sessions
    executor._bring_browser_to_foreground.assert_awaited_once_with(99)


@pytest.mark.asyncio
async def test_ensure_session_forces_headless_in_test_safe_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        url = "about:blank"

        def __init__(self) -> None:
            self.bring_to_front = AsyncMock()
            self.goto = AsyncMock()
            self.evaluate = AsyncMock()
            self.keyboard = SimpleNamespace(press=AsyncMock())
            self.wait_for_load_state = AsyncMock()
            self.on = lambda *_args, **_kwargs: None

    class FakeContext:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self):
            return self.page

    class FakeBrowser:
        def __init__(self) -> None:
            self.new_context = AsyncMock(return_value=FakeContext())
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
    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")

    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False, viewport_width=1440, viewport_height=900)

    await executor._ensure_session("run-safe")

    manager.playwright.chromium.launch.assert_awaited_once_with(
        headless=True,
        args=["--window-size=1440,900", "--window-position=0,0"],
    )


@pytest.mark.asyncio
async def test_bring_browser_to_foreground_retries_until_success(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False)
    attempts = {"count": 0}

    def fake_focus() -> bool:
        attempts["count"] += 1
        return attempts["count"] >= 3

    executor._focus_browser_window = lambda _pid=None: fake_focus()

    await executor._bring_browser_to_foreground()

    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_bring_browser_to_foreground_uses_app_activate_fallback(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False)
    executor._focus_browser_window = lambda _pid=None: False
    executor._app_activate_browser_window = Mock(return_value=True)  # type: ignore[method-assign]

    await executor._bring_browser_to_foreground()

    executor._app_activate_browser_window.assert_called_once()  # type: ignore[attr-defined]


def test_recorded_video_path_prefers_finalized_session_name(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    video_dir = tmp_path / "run-xyz" / "session_video"
    video_dir.mkdir(parents=True)
    final_path = video_dir / "session.webm"
    final_path.write_bytes(b"video")

    assert executor.recorded_video_path_for_run("run-xyz") == final_path


@pytest.mark.asyncio
async def test_launch_app_reports_opened_browser_session_for_fresh_run(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=False)
    executor.set_current_run_id("run-fresh")
    executor._fresh_session_run_id = "run-fresh"
    executor._current_page = AsyncMock(return_value=SimpleNamespace())
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))

    result = await executor.execute(AgentAction(action_type=ActionType.LAUNCH_APP, text="browser"))

    assert result.success is True
    assert result.detail == "Opened browser session"


@pytest.mark.asyncio
async def test_click_uses_target_context_center_when_coordinates_missing(tmp_path: Path) -> None:
    mouse = SimpleNamespace(click=AsyncMock())
    page = SimpleNamespace(mouse=mouse, wait_for_load_state=AsyncMock())
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))

    result = await executor.execute(
        AgentAction(
            action_type=ActionType.CLICK,
            target_element_id="search_input",
            target_context={
                "intent": {"action": "click", "target_text": "Search", "expected_element_types": ["input"]},
                "original_target": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "x": 100,
                    "y": 200,
                    "width": 80,
                    "height": 20,
                },
                "selected_candidate_evidence": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "total_score": 1.0,
                    "matched_signals": [],
                    "rejected_by": [],
                    "action_compatible": True,
                    "exact_semantic_match": True,
                    "uses_unlabeled_fallback": False,
                    "nearest_matched_text_candidate_id": None,
                    "spatial_grounding_contributed": False,
                    "confidence_band": "high",
                },
            },
        )
    )

    assert result.success is True
    mouse.click.assert_awaited_once_with(140, 210)


@pytest.mark.asyncio
async def test_press_key_focuses_target_context_before_keypress(tmp_path: Path) -> None:
    mouse = SimpleNamespace(click=AsyncMock())
    keyboard = SimpleNamespace(press=AsyncMock())
    page = SimpleNamespace(mouse=mouse, keyboard=keyboard, wait_for_load_state=AsyncMock())
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))

    result = await executor.execute(
        AgentAction(
            action_type=ActionType.PRESS_KEY,
            key="Enter",
            target_element_id="search_input",
            target_context={
                "intent": {"action": "click", "target_text": "Search", "expected_element_types": ["input"]},
                "original_target": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "x": 50,
                    "y": 60,
                    "width": 100,
                    "height": 40,
                },
                "selected_candidate_evidence": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "total_score": 1.0,
                    "matched_signals": [],
                    "rejected_by": [],
                    "action_compatible": True,
                    "exact_semantic_match": True,
                    "uses_unlabeled_fallback": False,
                    "nearest_matched_text_candidate_id": None,
                    "spatial_grounding_contributed": False,
                    "confidence_band": "high",
                },
            },
        )
    )

    assert result.success is True
    mouse.click.assert_awaited_once_with(100, 80)
    keyboard.press.assert_awaited_once_with("Enter")


@pytest.mark.asyncio
async def test_type_enters_text_without_clear_before_typing(tmp_path: Path) -> None:
    mouse = SimpleNamespace(click=AsyncMock())
    keyboard = SimpleNamespace(type=AsyncMock(), press=AsyncMock())
    page = SimpleNamespace(mouse=mouse, keyboard=keyboard, wait_for_load_state=AsyncMock())
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))

    result = await executor.execute(
        AgentAction(
            action_type=ActionType.TYPE,
            target_element_id="search_input",
            text="MacBook under $2000",
            press_enter=True,
            target_context={
                "intent": {"action": "type", "target_text": "Search", "expected_element_types": ["input"]},
                "original_target": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "x": 50,
                    "y": 60,
                    "width": 100,
                    "height": 40,
                },
                "selected_candidate_evidence": {
                    "element_id": "search_input",
                    "element_type": "input",
                    "primary_name": "Search",
                    "total_score": 1.0,
                    "matched_signals": [],
                    "rejected_by": [],
                    "action_compatible": True,
                    "exact_semantic_match": True,
                    "uses_unlabeled_fallback": False,
                    "nearest_matched_text_candidate_id": None,
                    "spatial_grounding_contributed": False,
                    "confidence_band": "high",
                },
            },
        )
    )

    assert result.success is True
    mouse.click.assert_awaited_once_with(100, 80)
    keyboard.type.assert_awaited_once_with("MacBook under $2000")
    keyboard.press.assert_awaited_once_with("Enter")


@pytest.mark.asyncio
async def test_type_clears_before_typing_when_requested(tmp_path: Path) -> None:
    keyboard = SimpleNamespace(type=AsyncMock(), press=AsyncMock())
    page = SimpleNamespace(mouse=SimpleNamespace(click=AsyncMock()), keyboard=keyboard, wait_for_load_state=AsyncMock())
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))

    result = await executor.execute(
        AgentAction(
            action_type=ActionType.TYPE,
            x=10,
            y=20,
            text="updated query",
            clear_before_typing=True,
        )
    )

    assert result.success is True
    assert keyboard.press.await_args_list[0].args == ("Control+A",)
    assert keyboard.press.await_args_list[1].args == ("Backspace",)
    keyboard.type.assert_awaited_once_with("updated query")
