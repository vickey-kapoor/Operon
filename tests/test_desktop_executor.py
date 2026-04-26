"""Tests for the desktop executor (pyautogui/mss based)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models.common import FailureCategory
from src.models.policy import ActionType, AgentAction


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_executor(artifact_dir: Path | None = None):
    """Import and create a DesktopExecutor with mocked DPI awareness."""
    with patch("src.executor.desktop._set_dpi_awareness"):
        from src.executor.desktop import DesktopExecutor

        return DesktopExecutor(artifact_dir=artifact_dir or _local_test_dir("desktop"))


# ── capture tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capture_returns_valid_frame() -> None:
    """capture() should return a CaptureFrame with correct dimensions."""
    art_dir = _local_test_dir("capture")

    fake_shot = SimpleNamespace(
        rgb=b"\x00" * (100 * 100 * 3),
        size=(100, 100),
        width=100,
        height=100,
    )
    # monitors[0] is the virtual combined screen; monitors[1] is the first physical monitor
    fake_monitor = [
        {"top": 0, "left": 0, "width": 100, "height": 100},
        {"top": 0, "left": 0, "width": 100, "height": 100},
    ]

    with (
        patch("src.executor.desktop.mss.mss") as mock_mss,
        patch("src.executor.desktop.mss.tools.to_png"),
    ):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.monitors = fake_monitor
        ctx.grab.return_value = fake_shot
        mock_mss.return_value = ctx

        executor = _make_executor(art_dir)
        frame = await executor.capture()

        assert frame.width == 100
        assert frame.height == 100
        assert frame.mime_type == "image/png"
        assert frame.artifact_path.endswith(".png")


# ── click tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_click_success() -> None:
    """Click at coordinates should succeed."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.CLICK, x=500, y=300)

    with (
        patch("src.executor.desktop.pyautogui.click") as mock_click,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_click.assert_called_once()
        assert "500" in result.detail and "300" in result.detail


@pytest.mark.asyncio
async def test_click_without_coords_fails() -> None:
    """Click without x,y should fail on desktop."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="e1")

    result = await executor.execute(action)

    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND


# ── type tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_type_with_coords() -> None:
    """Type at coordinates should click once when focus is detected, then paste via clipboard."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.TYPE, text="hello", x=100, y=200)

    with (
        patch("src.executor.desktop.pyautogui.click") as mock_click,
        patch("src.executor.desktop.pyautogui.hotkey") as mock_hotkey,
        patch("src.executor.desktop.pyperclip.copy") as mock_copy,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
        # Return different bytes pre/post click to simulate focus change (no retry needed)
        patch.object(executor, "_sample_region", side_effect=[b"\x00" * 100, b"\xff" * 100]),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_click.assert_called_once_with(100, 200)
        mock_copy.assert_called_once_with("hello")
        mock_hotkey.assert_any_call("ctrl", "v")


@pytest.mark.asyncio
async def test_type_with_coords_retries_click_when_focus_not_detected() -> None:
    """Type should retry click once when pre/post pixel region is unchanged (focus stolen)."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.TYPE, text="hello", x=100, y=200)

    with (
        patch("src.executor.desktop.pyautogui.click") as mock_click,
        patch("src.executor.desktop.pyautogui.hotkey"),
        patch("src.executor.desktop.pyperclip.copy"),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
        # Return identical bytes to simulate no visual change (focus stolen → retry)
        patch.object(executor, "_sample_region", return_value=b"\x00" * 100),
    ):
        result = await executor.execute(action)

        assert result.success is True
        assert mock_click.call_count == 2


# ── press_key tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_press_key() -> None:
    """press_key should call pyautogui.press."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.PRESS_KEY, key="enter")

    with (
        patch("src.executor.desktop.pyautogui.press") as mock_press,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_press.assert_called_once()


# ── hotkey tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hotkey_splits_keys() -> None:
    """hotkey should split key string and call pyautogui.hotkey."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.HOTKEY, key="ctrl+shift+esc")

    with (
        patch("src.executor.desktop.pyautogui.hotkey") as mock_hotkey,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_hotkey.assert_called_once_with("ctrl", "shift", "esc")


# ── launch_app tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_launch_app_known_alias() -> None:
    """launch_app should resolve known aliases."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="calculator")

    with (
        patch("src.executor.desktop.subprocess.Popen") as mock_popen,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        call_args = mock_popen.call_args
        assert call_args[0][0] == "calc.exe"


@pytest.mark.asyncio
async def test_launch_app_unknown_passthrough() -> None:
    """launch_app with unknown app name passes it through as-is."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="myapp.exe")

    with (
        patch("src.executor.desktop.subprocess.Popen") as mock_popen,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        call_args = mock_popen.call_args
        assert call_args[0][0] == "myapp.exe"


# ── unsupported action tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_navigate_unsupported() -> None:
    """NAVIGATE should fail on desktop."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com")

    result = await executor.execute(action)

    assert result.success is False
    assert "not supported" in result.detail


@pytest.mark.asyncio
async def test_select_unsupported() -> None:
    """SELECT should fail on desktop."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.SELECT, text="option1", target_element_id="e1")

    result = await executor.execute(action)

    assert result.success is False
    assert "not supported" in result.detail


# ── wait tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_action() -> None:
    """WAIT should sleep and return success."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=100)

    with patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"):
        result = await executor.execute(action)

    assert result.success is True
    assert "0.1s" in result.detail


# ── stop / wait_for_user tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_stop_action() -> None:
    """STOP should return success with no side effects."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.STOP)

    result = await executor.execute(action)

    assert result.success is True
    assert "stop" in result.detail.lower()


@pytest.mark.asyncio
async def test_wait_for_user_action() -> None:
    """WAIT_FOR_USER should return success."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.WAIT_FOR_USER, text="Please confirm")

    result = await executor.execute(action)

    assert result.success is True


# ── model validation edge cases ─────────────────────────────────


def test_hotkey_rejects_missing_key() -> None:
    """HOTKEY without key should be rejected by the model validator."""
    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.HOTKEY)


def test_launch_app_rejects_missing_text() -> None:
    """LAUNCH_APP without text should be rejected by the model validator."""
    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.LAUNCH_APP)


def test_type_rejects_missing_text() -> None:
    """TYPE without text should be rejected by the model validator."""
    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.TYPE, x=100, y=200)


def test_press_key_rejects_missing_key() -> None:
    """PRESS_KEY without key should be rejected by the model validator."""
    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.PRESS_KEY)


# ── launch_app ms-settings path ────────────────────────────────


@pytest.mark.asyncio
async def test_launch_app_ms_settings() -> None:
    """launch_app with 'settings' should use os.startfile for ms-settings: URI."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="settings")

    with (
        patch("src.executor.desktop.os.startfile") as mock_startfile,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_startfile.assert_called_once_with("ms-settings:")


# ── double_click tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_double_click_success() -> None:
    """Double-click at coordinates should succeed."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.DOUBLE_CLICK, x=500, y=300)

    with (
        patch("src.executor.desktop.pyautogui.doubleClick") as mock_dclick,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_dclick.assert_called_once()
        assert "500" in result.detail and "300" in result.detail


@pytest.mark.asyncio
async def test_double_click_without_coords_fails() -> None:
    """Double-click without x,y should fail on desktop."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.DOUBLE_CLICK, target_element_id="e1")

    result = await executor.execute(action)

    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND


# ── right_click tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_right_click_success() -> None:
    """Right-click at coordinates should succeed."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.RIGHT_CLICK, x=400, y=250)

    with (
        patch("src.executor.desktop.pyautogui.rightClick") as mock_rclick,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_rclick.assert_called_once()
        assert "400" in result.detail and "250" in result.detail


@pytest.mark.asyncio
async def test_right_click_without_coords_fails() -> None:
    """Right-click without x,y should fail on desktop."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.RIGHT_CLICK, target_element_id="e1")

    result = await executor.execute(action)

    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND


# ── drag tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drag_success() -> None:
    """Drag from (x,y) to (x_end,y_end) using mouseDown/moveTo/mouseUp."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.DRAG, x=100, y=200, x_end=300, y_end=400)

    with (
        patch("src.executor.desktop.pyautogui.moveTo") as mock_move,
        patch("src.executor.desktop.pyautogui.mouseDown") as mock_down,
        patch("src.executor.desktop.pyautogui.mouseUp") as mock_up,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        assert mock_move.call_count == 2
        mock_move.assert_any_call(100, 200)
        mock_move.assert_any_call(300, 400, duration=0.6)
        mock_down.assert_called_once()
        mock_up.assert_called_once()


@pytest.mark.asyncio
async def test_drag_missing_endpoints_fails() -> None:
    """Drag without x_end/y_end should be rejected by the model validator."""
    with pytest.raises((ValidationError, ValueError)):
        AgentAction(action_type=ActionType.DRAG, x=100, y=200)


# ── scroll tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scroll_success() -> None:
    """Scroll at coordinates should succeed."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.SCROLL, scroll_amount=3, x=100, y=200)

    with (
        patch("src.executor.desktop.pyautogui.scroll") as mock_scroll,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_scroll.assert_called_once_with(3, x=100, y=200)


@pytest.mark.asyncio
async def test_scroll_without_coords_fails() -> None:
    """Scroll without x,y should be rejected by the model validator."""
    with pytest.raises((ValidationError, ValueError)):
        AgentAction(action_type=ActionType.SCROLL, scroll_amount=3)


# ── hover tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hover_success() -> None:
    """Hover at coordinates should succeed."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.HOVER, x=150, y=250)

    with (
        patch("src.executor.desktop.pyautogui.moveTo") as mock_move,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

        assert result.success is True
        mock_move.assert_called_once_with(150, 250)


@pytest.mark.asyncio
async def test_hover_without_coords_fails() -> None:
    """Hover without x,y should be rejected by the model validator."""
    with pytest.raises((ValidationError, ValueError)):
        AgentAction(action_type=ActionType.HOVER)


# ── clipboard tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_clipboard_success() -> None:
    """read_clipboard should return clipboard content in detail."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.READ_CLIPBOARD)

    with patch("src.executor.desktop.pyperclip.paste", return_value="test content"):
        result = await executor.execute(action)

        assert result.success is True
        assert "test content" in result.detail


@pytest.mark.asyncio
async def test_write_clipboard_success() -> None:
    """write_clipboard should copy text to clipboard."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.WRITE_CLIPBOARD, text="hello world")

    with patch("src.executor.desktop.pyperclip.copy") as mock_copy:
        result = await executor.execute(action)

        assert result.success is True
        mock_copy.assert_called_once_with("hello world")


@pytest.mark.asyncio
async def test_write_clipboard_missing_text_fails() -> None:
    """write_clipboard without text should be rejected by the model validator."""
    with pytest.raises((ValidationError, ValueError)):
        AgentAction(action_type=ActionType.WRITE_CLIPBOARD)


# ── screenshot_region tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_screenshot_region_success() -> None:
    """screenshot_region should grab the specified region."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.SCREENSHOT_REGION, x=10, y=20, x_end=200, y_end=300)

    fake_shot = SimpleNamespace(
        rgb=b"\x00" * (190 * 280 * 3),
        size=(190, 280),
    )

    with (
        patch("src.executor.desktop.mss.mss") as mock_mss,
        patch("src.executor.desktop.mss.tools.to_png"),
    ):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.grab.return_value = fake_shot
        mock_mss.return_value = ctx

        result = await executor.execute(action)

        assert result.success is True
        # Verify the region dict passed to grab
        grab_call = ctx.grab.call_args[0][0]
        assert grab_call["left"] == 10
        assert grab_call["top"] == 20
        assert grab_call["width"] == 190
        assert grab_call["height"] == 280


@pytest.mark.asyncio
async def test_screenshot_region_invalid_bounds_fails() -> None:
    """screenshot_region with x_end <= x should fail."""
    executor = _make_executor()
    # x_end <= x: invalid bounds
    action = AgentAction(action_type=ActionType.SCREENSHOT_REGION, x=200, y=20, x_end=100, y_end=300)

    result = await executor.execute(action)

    assert result.success is False


# ── safety: alt+f4 blocked ──────────────────────────────────────


@pytest.mark.asyncio
async def test_hotkey_blocks_alt_f4() -> None:
    """alt+f4 hotkey should be blocked for safety."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.HOTKEY, key="alt+f4")

    result = await executor.execute(action)

    assert result.success is False
    assert "blocked" in result.detail.lower()


@pytest.mark.asyncio
async def test_hotkey_blocks_alt_f4_uppercase() -> None:
    """Alt+F4 (any case) should be blocked."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.HOTKEY, key="Alt+F4")

    result = await executor.execute(action)

    assert result.success is False
    assert "blocked" in result.detail.lower()


@pytest.mark.asyncio
async def test_hotkey_allows_other_hotkeys() -> None:
    """Non-dangerous hotkeys like ctrl+c should still work."""
    executor = _make_executor()
    action = AgentAction(action_type=ActionType.HOTKEY, key="ctrl+c")

    with (
        patch("src.executor.desktop.pyautogui.hotkey"),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

    assert result.success is True


# ── process tracking and cleanup ────────────────────────────────


@pytest.mark.asyncio
async def test_launch_app_tracks_process() -> None:
    """launch_app should track the Popen handle when run_id is set."""
    executor = _make_executor()
    executor.set_current_run_id("run-123")
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="calculator")

    mock_proc = MagicMock()
    mock_proc.args = "calc.exe"

    with (
        patch("src.executor.desktop.subprocess.Popen", return_value=mock_proc),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)

    assert result.success is True
    assert "run-123" in executor._launched_processes
    assert mock_proc in executor._launched_processes["run-123"]


@pytest.mark.asyncio
async def test_cleanup_terminates_tracked_processes() -> None:
    """cleanup_run should terminate non-protected tracked processes."""
    executor = _make_executor()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    mock_proc.args = "calc.exe"
    mock_proc.pid = 12345

    executor._launched_processes["run-456"] = [mock_proc]

    closed = executor.cleanup_run("run-456")

    assert closed == 1
    mock_proc.terminate.assert_called_once()
    assert "run-456" not in executor._launched_processes


@pytest.mark.asyncio
async def test_cleanup_skips_protected_processes() -> None:
    """cleanup_run should skip protected processes like VS Code."""
    executor = _make_executor()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.args = "code.exe"
    mock_proc.pid = 99999

    executor._launched_processes["run-789"] = [mock_proc]

    closed = executor.cleanup_run("run-789")

    assert closed == 0
    mock_proc.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_skips_already_exited() -> None:
    """cleanup_run should skip processes that already exited."""
    executor = _make_executor()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # already exited
    mock_proc.args = "notepad.exe"
    mock_proc.pid = 11111

    executor._launched_processes["run-aaa"] = [mock_proc]

    closed = executor.cleanup_run("run-aaa")

    assert closed == 0
    mock_proc.terminate.assert_not_called()
