"""
DesktopExecutor — captures OS screenshots with mss and dispatches
input events with pyautogui.

Thread safety: pyautogui is NOT thread-safe. All execute() calls MUST
be serialized — the FastAPI semaphore ensures at most one desktop session
runs at a time, making this safe without additional locking.
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from src.executor.actions import ActionResult

from PIL import Image

logger = logging.getLogger(__name__)

# Check mock mode once at import time
_DESKTOP_MOCK = os.environ.get("DESKTOP_MOCK", "").lower() == "true"


class DesktopExecutorError(Exception):
    """Raised when the desktop executor cannot be initialized."""


class DesktopExecutor:
    """
    Captures full-screen screenshots via mss and dispatches actions
    via pyautogui. Coordinates are expected in logical screen pixels.

    Lifecycle:
        executor = DesktopExecutor()
        await executor.start()   # detects screen size, warms mss
        ...
        await executor.stop()    # releases mss resources

    All public methods are async for interface compatibility with
    PlaywrightBrowserExecutor, even though the underlying calls are
    synchronous. Blocking calls run in a thread executor.

    Mock mode: set DESKTOP_MOCK=true env var to stub all actions and
    return blank 1920x1080 screenshots. For CI testing without a display.
    """

    # pyautogui inter-action pause — 50ms prevents input floods
    _PYAUTOGUI_PAUSE = 0.05
    # Typing delay per character in seconds (human-like speed)
    _TYPE_INTERVAL = 0.02

    def __init__(self) -> None:
        self._mss = None            # mss.mss() instance
        self._screen_width: int = 0
        self._screen_height: int = 0
        # HiDPI scale factors: pyautogui coords / mss coords
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mock = _DESKTOP_MOCK
        # Dedicated single-thread executor: mss stores Windows GDI handles
        # in thread-local storage, so all mss + pyautogui calls must run on
        # the same thread where mss.mss() was created.
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize mss, detect screen dimensions and DPI scale."""
        if self._started:
            return

        self._loop = asyncio.get_running_loop()

        # Create a dedicated single thread — mss and pyautogui must always
        # run on the thread that created the mss.mss() GDI context.
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="desktop-exec",
        )

        if self._mock:
            self._screen_width = 1920
            self._screen_height = 1080
            self._scale_x = 1.0
            self._scale_y = 1.0
            self._started = True
            logger.info("DesktopExecutor started in MOCK mode (1920x1080)")
            return

        await self._loop.run_in_executor(self._thread_pool, self._init_sync)
        self._started = True
        logger.info(
            "DesktopExecutor started: screen=%dx%d scale=%.2fx%.2f",
            self._screen_width, self._screen_height,
            self._scale_x, self._scale_y,
        )

    def _init_sync(self) -> None:
        """Synchronous initialization — runs in thread executor."""
        try:
            import mss
            import pyautogui
        except ImportError as exc:
            raise DesktopExecutorError(
                f"Desktop mode requires 'mss' and 'pyautogui': {exc}"
            ) from exc

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = self._PYAUTOGUI_PAUSE

        self._mss = mss.mss()
        monitor = self._mss.monitors[1]  # index 1 = primary monitor

        # mss returns logical resolution (what the OS reports)
        mss_w = monitor["width"]
        mss_h = monitor["height"]

        # pyautogui.size() returns screen resolution in pyautogui's
        # coordinate space — may differ on HiDPI displays.
        pyag_w, pyag_h = pyautogui.size()

        self._screen_width = mss_w
        self._screen_height = mss_h

        # Scale factor: convert screenshot pixel → pyautogui pixel
        self._scale_x = pyag_w / mss_w if mss_w > 0 else 1.0
        self._scale_y = pyag_h / mss_h if mss_h > 0 else 1.0

        logger.debug(
            "Screen: mss=%dx%d pyautogui=%dx%d scale=%.3fx%.3f",
            mss_w, mss_h, pyag_w, pyag_h, self._scale_x, self._scale_y,
        )

    async def stop(self) -> None:
        """Release mss resources."""
        if not self._started:
            return
        if self._mss:
            try:
                self._mss.close()
            except Exception as exc:
                logger.debug("mss close error (non-fatal): %s", exc)
            self._mss = None
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
        self._started = False
        logger.info("DesktopExecutor stopped")

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def screenshot(self) -> Image.Image:
        """Capture the primary monitor and return a PIL Image."""
        if self._mock:
            return Image.new("RGB", (self._screen_width, self._screen_height), (0, 0, 0))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._screenshot_sync)

    def _screenshot_sync(self) -> Image.Image:
        monitor = self._mss.monitors[1]
        sct = self._mss.grab(monitor)
        # mss returns BGRA; convert to RGB for PIL / Gemini
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return img

    async def screenshot_base64(self) -> str:
        """Capture and return base64-encoded PNG."""
        img = await self.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) in mss/screenshot coordinates."""
        return (self._screen_width, self._screen_height)

    # ------------------------------------------------------------------
    # Coordinate scaling
    # ------------------------------------------------------------------

    def _scale_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Scale coordinates from screenshot space (mss) to input space (pyautogui).

        On standard displays this is 1:1.
        On HiDPI/Retina displays this converts logical → physical pixels.
        Clamps to screen bounds after scaling.
        """
        sx = int(x * self._scale_x)
        sy = int(y * self._scale_y)
        pyag_w, pyag_h = int(self._screen_width * self._scale_x), int(self._screen_height * self._scale_y)
        sx = max(0, min(sx, pyag_w - 1))
        sy = max(0, min(sy, pyag_h - 1))
        return sx, sy

    # ------------------------------------------------------------------
    # Action helpers (all synchronous — run via thread executor)
    # ------------------------------------------------------------------

    def _click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        logger.debug("click (%d,%d) → scaled (%d,%d)", x, y, sx, sy)
        pyautogui.click(sx, sy)

    def _right_click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.rightClick(sx, sy)

    def _double_click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.doubleClick(sx, sy)

    def _move_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.moveTo(sx, sy, duration=0.1)

    def _type_sync(self, text: str) -> None:
        import subprocess

        import pyautogui
        # Clipboard paste is more reliable than typewrite() for Windows
        # Search, UWP apps, and non-ASCII text.
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "echo|set /p=" + text + "|clip"],
                shell=False, capture_output=True, timeout=5,
            )
            pyautogui.hotkey("ctrl", "v")
            import time
            time.sleep(0.1)
        else:
            pyautogui.typewrite(text, interval=self._TYPE_INTERVAL)

    def _key_sync(self, key_combo: str) -> None:
        """
        Press a key or chord.

        Single key: "enter", "escape", "tab", "f5"
        Chord: "ctrl+c", "alt+f4", "ctrl+shift+t"
        Win key: "win" or "super"

        pyautogui.hotkey() accepts comma-separated key names.
        We split on '+' to handle chords.
        """
        import pyautogui
        # Normalize: lowercase, strip spaces
        normalized = key_combo.strip().lower()
        # Map common aliases
        aliases = {
            "win": "winleft",
            "super": "winleft",
            "cmd": "command",
            "return": "enter",
            "del": "delete",
            "esc": "escape",
        }
        parts = [aliases.get(k.strip(), k.strip()) for k in normalized.split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)

    def _scroll_sync(self, x: int, y: int, direction: str, amount: int = 3) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.moveTo(sx, sy)
        clicks = amount if direction in ("down", "right") else -amount
        if direction in ("up", "down"):
            pyautogui.scroll(clicks)
        else:
            pyautogui.hscroll(clicks)

    # ------------------------------------------------------------------
    # Main execute method
    # ------------------------------------------------------------------

    async def execute(self, action) -> "ActionResult":
        """
        Execute a single DesktopAction and return an ActionResult.

        Mirrors PlaywrightBrowserExecutor.execute() interface.
        All blocking pyautogui calls run in the thread executor.
        """
        from src.executor.actions import ActionResult
        loop = asyncio.get_running_loop()

        if not self._started:
            return ActionResult(
                success=False,
                error="DesktopExecutor not started",
                action_type=action.action,
            )

        act = action.action

        try:
            # Validate required fields before mock short-circuit
            if act in ("click", "right_click", "double_click", "move"):
                if action.x is None or action.y is None:
                    return ActionResult(
                        success=False,
                        error=f"{act} requires x and y coordinates",
                        action_type=act,
                    )
            elif act == "type":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="type action requires text",
                        action_type=act,
                    )
            elif act == "key":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="key action requires text (key name / chord)",
                        action_type=act,
                    )

            if self._mock:
                # Mock mode: no-op all actions, return success
                screenshot_b64 = None
                if act in ("click", "right_click", "double_click", "move", "key"):
                    screenshot_b64 = await self.screenshot_base64()
                elif act == "wait":
                    duration_s = (action.duration or 1000) / 1000.0
                    await asyncio.sleep(duration_s)
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=act,
                )

            if act in ("click", "right_click", "double_click", "move"):
                if action.x is None or action.y is None:
                    return ActionResult(
                        success=False,
                        error=f"{act} requires x and y coordinates",
                        action_type=act,
                    )
                fn_map = {
                    "click": self._click_sync,
                    "right_click": self._right_click_sync,
                    "double_click": self._double_click_sync,
                    "move": self._move_sync,
                }
                await loop.run_in_executor(self._thread_pool, fn_map[act], action.x, action.y)
                # Take a screenshot after pointer actions so caller has fresh state
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=act,
                )

            elif act == "type":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="type action requires text",
                        action_type=act,
                    )
                await loop.run_in_executor(self._thread_pool, self._type_sync, action.text)
                return ActionResult(success=True, action_type=act)

            elif act == "key":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="key action requires text (key name / chord)",
                        action_type=act,
                    )
                await loop.run_in_executor(self._thread_pool, self._key_sync, action.text)
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=act,
                )

            elif act == "scroll":
                x = action.x or (self._screen_width // 2)
                y = action.y or (self._screen_height // 2)
                direction = action.direction or "down"
                amount = 3  # fixed; DesktopAction doesn't have scroll_amount
                await loop.run_in_executor(self._thread_pool, self._scroll_sync, x, y, direction, amount)
                return ActionResult(success=True, action_type=act)

            elif act == "wait":
                duration_s = (action.duration or 1000) / 1000.0
                await asyncio.sleep(duration_s)
                return ActionResult(success=True, action_type=act)

            elif act in ("done", "confirm_required"):
                # No physical action — signal handled by the caller
                return ActionResult(success=True, action_type=act)

            else:
                return ActionResult(
                    success=False,
                    error=f"Unknown desktop action type: {act!r}",
                    action_type=act,
                )

        except Exception as exc:
            logger.exception("DesktopExecutor error executing %s: %s", act, exc)
            return ActionResult(success=False, error=str(exc), action_type=act)
