"""The task page's live view of a desktop the person is not sitting at.

Shared by the Linux cloud computer (KAN-108) and the guest agent inside a run
environment (KAN-80): JPEG frames of the screen for ``/ws/stream``
(``live_frame_png``) and the person's own clicks and keys during a takeover
(``user_input``). Mixed into a ``DesktopExecutor`` subclass, before it.
"""

from __future__ import annotations

import asyncio
import io
import time

from operon.executor import desktop as _desktop

# Live view: at most this many frames a second, at most this wide, this JPEG quality.
_LIVE_FPS = 4.0
_LIVE_MAX_WIDTH = 1280
_LIVE_JPEG_QUALITY = 60


class LiveViewMixin:
    """``live_frame_png`` and ``user_input`` for a ``DesktopExecutor``."""

    _current_run_id: str | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._live_frame: bytes | None = None
        self._live_frame_at = 0.0
        self._live_lock = asyncio.Lock()

    def _screen(self) -> dict:
        with _desktop._mss() as sct:
            return dict(sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])

    def _grab_jpeg(self) -> bytes:
        from PIL import Image

        with _desktop._mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        if image.width > _LIVE_MAX_WIDTH:
            image = image.resize((_LIVE_MAX_WIDTH, round(image.height * _LIVE_MAX_WIDTH / image.width)))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=_LIVE_JPEG_QUALITY)
        return out.getvalue()

    async def live_frame_png(self, run_id: str) -> bytes | None:
        """A JPEG of the screen for the run on this desktop, or None for any other run.

        (The name matches the browser executor's hook that ``/ws/stream`` polls.)
        Every viewer polls, so frames are shared and taken at most ``_LIVE_FPS`` a second.
        """
        if run_id is None or run_id != self._current_run_id:
            return None
        async with self._live_lock:
            if self._live_frame is None or time.monotonic() - self._live_frame_at >= 1.0 / _LIVE_FPS:
                self._live_frame = await asyncio.to_thread(self._grab_jpeg)
                self._live_frame_at = time.monotonic()
            return self._live_frame

    async def user_input(
        self,
        run_id: str,
        *,
        input_type: str,
        x_ratio: float = 0.0,
        y_ratio: float = 0.0,
        text: str = "",
        key: str = "",
        delta_y: int = 300,
    ) -> bool:
        """The person's own input during a takeover, sent from the live view.

        Coordinates are fractions of the screen. This is the person, not the agent,
        so it skips the policy path and the visual servo, like the browser's takeover.
        Returns False when ``run_id`` is not the run on this desktop.
        """
        if run_id != self._current_run_id:
            return False
        gui = _desktop.pyautogui
        screen = await asyncio.to_thread(self._screen)
        x = screen["left"] + round(max(0.0, min(1.0, x_ratio)) * (screen["width"] - 1))
        y = screen["top"] + round(max(0.0, min(1.0, y_ratio)) * (screen["height"] - 1))
        if input_type == "click":
            await asyncio.to_thread(gui.click, x, y)
        elif input_type == "dblclick":
            await asyncio.to_thread(gui.doubleClick, x, y)
        elif input_type == "type":
            # Paste, like the agent's TYPE: key-by-key typing drops non-ASCII characters.
            await asyncio.to_thread(_desktop.pyperclip.copy, text)
            await asyncio.to_thread(gui.hotkey, "ctrl", "v")
        elif input_type == "key":
            keys = [_desktop._normalize_key_pyautogui(k.strip()) for k in key.split("+") if k.strip()]
            unknown = _desktop._unknown_keys(keys)
            if not keys or unknown:
                raise ValueError(f"unknown key {key!r}")
            await asyncio.to_thread(gui.hotkey, *keys)
        elif input_type == "scroll":
            # pyautogui scrolls in wheel clicks, up is positive; the page sends pixels, down positive.
            clicks = -round(delta_y / 100) or (-1 if delta_y > 0 else 1 if delta_y < 0 else 0)
            if clicks:
                await asyncio.to_thread(gui.scroll, clicks, x=x, y=y)
        else:
            raise ValueError(f"unsupported input_type {input_type!r}")
        self._live_frame = None  # show the result on the next frame, not a stale one
        return True
