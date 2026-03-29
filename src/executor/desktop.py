"""Desktop executor for full-screen computer-use automation."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import mss
import mss.tools
import pyautogui
import pyperclip

from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage
from src.models.execution import (
    ExecutedAction,
    ExecutionAttemptTrace,
    ExecutionTrace,
)
from src.models.policy import ActionType, AgentAction

from .browser import BrowserExecutor

logger = logging.getLogger(__name__)

_APP_ALIASES: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe",
    "settings": "ms-settings:",
    "task manager": "taskmgr.exe",
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
}


def _set_dpi_awareness() -> None:
    """Set per-monitor DPI awareness on Windows for accurate coordinates."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


class DesktopExecutor(BrowserExecutor):
    """Desktop executor using pyautogui/mss for full-screen automation."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path = ".desktop-artifacts",
        type_interval: float = 0.03,
        post_action_delay: float = 0.3,
        post_launch_delay: float = 1.5,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._type_interval = type_interval
        self._post_action_delay = post_action_delay
        self._post_launch_delay = post_launch_delay

        _set_dpi_awareness()
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    async def capture(self) -> CaptureFrame:
        """Take a full-screen screenshot using mss."""
        return await asyncio.to_thread(self._capture_sync)

    def _capture_sync(self) -> CaptureFrame:
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # full virtual screen
            shot = sct.grab(monitor)
            filename = f"desktop_{uuid4().hex[:8]}.png"
            filepath = self._artifact_dir / filename
            mss.tools.to_png(shot.rgb, shot.size, output=str(filepath))
            return CaptureFrame(
                artifact_path=str(filepath),
                width=shot.width,
                height=shot.height,
                mime_type="image/png",
            )

    async def execute(self, action: AgentAction) -> ExecutedAction:
        """Execute a desktop action using pyautogui."""
        at = action.action_type
        dispatch = {
            ActionType.CLICK: self._exec_click,
            ActionType.DOUBLE_CLICK: self._exec_double_click,
            ActionType.RIGHT_CLICK: self._exec_right_click,
            ActionType.TYPE: self._exec_type,
            ActionType.PRESS_KEY: self._exec_press_key,
            ActionType.HOTKEY: self._exec_hotkey,
            ActionType.LAUNCH_APP: self._exec_launch_app,
            ActionType.DRAG: self._exec_drag,
            ActionType.SCROLL: self._exec_scroll,
            ActionType.HOVER: self._exec_hover,
            ActionType.READ_CLIPBOARD: self._exec_read_clipboard,
            ActionType.WRITE_CLIPBOARD: self._exec_write_clipboard,
            ActionType.SCREENSHOT_REGION: self._exec_screenshot_region,
            ActionType.WAIT: self._exec_wait,
            ActionType.STOP: self._exec_noop,
            ActionType.WAIT_FOR_USER: self._exec_noop,
            ActionType.NAVIGATE: self._exec_unsupported,
            ActionType.SELECT: self._exec_unsupported,
        }
        handler = dispatch.get(at, self._exec_unsupported)
        return await handler(action)

    # ── action handlers ──────────────────────────────────────────

    async def _exec_click(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None:
            return self._fail(action, "click requires x,y coordinates on desktop", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.click, action.x, action.y)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Clicked at ({action.x}, {action.y})", after_path)
        except Exception as exc:
            return self._fail(action, f"Click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_double_click(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None:
            return self._fail(action, "double_click requires x,y coordinates on desktop", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.doubleClick, action.x, action.y)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Double-clicked at ({action.x}, {action.y})", after_path)
        except Exception as exc:
            return self._fail(action, f"Double-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_right_click(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None:
            return self._fail(action, "right_click requires x,y coordinates on desktop", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.rightClick, action.x, action.y)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Right-clicked at ({action.x}, {action.y})", after_path)
        except Exception as exc:
            return self._fail(action, f"Right-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_type(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "type requires text", FailureCategory.EXECUTION_ERROR)
        try:
            if action.x is not None and action.y is not None:
                await asyncio.to_thread(pyautogui.click, action.x, action.y)
                await asyncio.sleep(0.15)
            await asyncio.to_thread(pyautogui.write, action.text, interval=self._type_interval)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Typed '{action.text}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Type failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_press_key(self, action: AgentAction) -> ExecutedAction:
        if action.key is None:
            return self._fail(action, "press_key requires key", FailureCategory.EXECUTION_ERROR)
        try:
            await asyncio.to_thread(pyautogui.press, action.key)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Pressed key '{action.key}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Press key failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_hotkey(self, action: AgentAction) -> ExecutedAction:
        if action.key is None:
            return self._fail(action, "hotkey requires key", FailureCategory.EXECUTION_ERROR)
        try:
            keys = [k.strip() for k in action.key.split("+")]
            await asyncio.to_thread(pyautogui.hotkey, *keys)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Pressed hotkey '{action.key}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Hotkey failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_launch_app(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "launch_app requires text (app name)", FailureCategory.EXECUTION_ERROR)
        app_key = action.text.strip().lower()
        command = _APP_ALIASES.get(app_key, action.text.strip())
        try:
            if command.startswith("ms-"):
                await asyncio.to_thread(os.startfile, command)  # type: ignore[attr-defined]
            else:
                await asyncio.to_thread(
                    subprocess.Popen,
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            await asyncio.sleep(self._post_launch_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Launched '{command}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Launch failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_drag(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None or action.x_end is None or action.y_end is None:
            return self._fail(action, "drag requires x, y, x_end, y_end", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.moveTo, action.x, action.y)
            await asyncio.sleep(0.1)
            await asyncio.to_thread(pyautogui.dragTo, action.x_end, action.y_end, duration=0.5)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(
                action,
                f"Dragged from ({action.x}, {action.y}) to ({action.x_end}, {action.y_end})",
                after_path,
            )
        except Exception as exc:
            return self._fail(action, f"Drag failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_scroll(self, action: AgentAction) -> ExecutedAction:
        if action.scroll_amount is None:
            return self._fail(action, "scroll requires scroll_amount", FailureCategory.EXECUTION_ERROR)
        if action.x is None or action.y is None:
            return self._fail(action, "scroll requires x, y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.scroll, action.scroll_amount, x=action.x, y=action.y)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            direction = "up" if action.scroll_amount > 0 else "down"
            return self._ok(
                action,
                f"Scrolled {direction} by {abs(action.scroll_amount)} at ({action.x}, {action.y})",
                after_path,
            )
        except Exception as exc:
            return self._fail(action, f"Scroll failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_hover(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None:
            return self._fail(action, "hover requires x, y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            await asyncio.to_thread(pyautogui.moveTo, action.x, action.y)
            await asyncio.sleep(0.15)
            after_path = await self._capture_after()
            return self._ok(action, f"Hovered at ({action.x}, {action.y})", after_path)
        except Exception as exc:
            return self._fail(action, f"Hover failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_read_clipboard(self, action: AgentAction) -> ExecutedAction:
        try:
            content = await asyncio.to_thread(pyperclip.paste)
            return self._ok(action, f"Clipboard content: {content}", None)
        except Exception as exc:
            return self._fail(action, f"Read clipboard failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_write_clipboard(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "write_clipboard requires text", FailureCategory.EXECUTION_ERROR)
        try:
            await asyncio.to_thread(pyperclip.copy, action.text)
            return self._ok(action, f"Copied to clipboard: '{action.text}'", None)
        except Exception as exc:
            return self._fail(action, f"Write clipboard failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_screenshot_region(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None or action.x_end is None or action.y_end is None:
            return self._fail(action, "screenshot_region requires x, y, x_end, y_end", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        if action.x_end <= action.x or action.y_end <= action.y:
            return self._fail(action, "screenshot_region requires x_end > x and y_end > y", FailureCategory.EXECUTION_ERROR)
        try:
            region = {"top": action.y, "left": action.x, "width": action.x_end - action.x, "height": action.y_end - action.y}
            filename = f"region_{uuid4().hex[:8]}.png"
            filepath = self._artifact_dir / filename

            def _grab_region() -> None:
                with mss.mss() as sct:
                    shot = sct.grab(region)
                    mss.tools.to_png(shot.rgb, shot.size, output=str(filepath))

            await asyncio.to_thread(_grab_region)
            return self._ok(
                action,
                f"Captured region ({action.x},{action.y})-({action.x_end},{action.y_end})",
                str(filepath),
            )
        except Exception as exc:
            return self._fail(action, f"Screenshot region failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_wait(self, action: AgentAction) -> ExecutedAction:
        wait_s = (action.wait_ms or 1000) / 1000.0
        await asyncio.sleep(wait_s)
        after_path = await self._capture_after()
        return self._ok(action, f"Waited {wait_s:.1f}s", after_path)

    async def _exec_noop(self, action: AgentAction) -> ExecutedAction:
        detail = "Acknowledged stop" if action.action_type is ActionType.STOP else f"Waiting for user: {action.text}"
        return self._ok(action, detail, None)

    async def _exec_unsupported(self, action: AgentAction) -> ExecutedAction:
        return self._fail(
            action,
            f"Action '{action.action_type}' is not supported on desktop",
            FailureCategory.EXECUTION_ERROR,
        )

    # ── helpers ──────────────────────────────────────────────────

    async def _capture_after(self) -> str:
        frame = await self.capture()
        return frame.artifact_path

    def _ok(self, action: AgentAction, detail: str, after_path: str | None) -> ExecutedAction:
        trace = ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="desktop_action_no_revalidation",
                    verification_result="success",
                )
            ],
            final_outcome="success",
        )
        return ExecutedAction(
            action=action,
            success=True,
            detail=detail,
            artifact_path=after_path,
            execution_trace=trace,
        )

    def _fail(
        self,
        action: AgentAction,
        detail: str,
        failure_category: FailureCategory,
    ) -> ExecutedAction:
        trace = ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="desktop_action_no_revalidation",
                    verification_result="failure",
                    failure_category=failure_category,
                )
            ],
            final_outcome="failure",
            final_failure_category=failure_category,
        )
        return ExecutedAction(
            action=action,
            success=False,
            detail=detail,
            execution_trace=trace,
            failure_category=failure_category,
            failure_stage=LoopStage.EXECUTE,
        )
