"""Linux desktop executor (KAN-108): the cloud computer on an Azure Linux VM.

Same pyautogui/mss executor as Windows (both work on an X11 display); this
subclass replaces only what is Windows-specific:

* apps come from ``LINUX_APP_LAUNCH_TARGETS`` and start as plain argv lists in
  their own process group, so cleanup can end the app and its children;
* "show desktop" is XFCE's ctrl+alt+d instead of win+d;
* the task page's live view gets JPEG frames of the screen (``live_frame_png``) and
  the person's clicks and keys during a takeover (``user_input``), because the
  person is not sitting at this machine.

Opt in with ``OPERON_DESKTOP_PLATFORM=linux``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import subprocess
import time

from operon.agent.app_catalog import LINUX_APP_LAUNCH_TARGETS
from operon.executor import desktop as _desktop
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import AgentAction

logger = logging.getLogger(__name__)

# Live view: at most this many frames a second, at most this wide, this JPEG quality.
_LIVE_FPS = 4.0
_LIVE_MAX_WIDTH = 1280
_LIVE_JPEG_QUALITY = 60


class LinuxDesktopExecutor(_desktop.DesktopExecutor):
    """Desktop executor for an X11 Linux desktop (XFCE on the Azure VM)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._live_frame: bytes | None = None
        self._live_frame_at = 0.0
        self._live_lock = asyncio.Lock()

    # ── resets ───────────────────────────────────────────────────

    async def reset_desktop(self) -> None:
        """Minimise all windows (XFCE's show-desktop shortcut) for a clean start."""
        try:
            await asyncio.to_thread(_desktop.pyautogui.hotkey, "ctrl", "alt", "d")
            await asyncio.sleep(1.0)
        except Exception:
            logger.debug("reset_desktop: ctrl+alt+d failed, continuing anyway", exc_info=True)
            await asyncio.sleep(0.5)

    # ── apps ─────────────────────────────────────────────────────

    async def _exec_launch_app(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "launch_app requires text (app name)", FailureCategory.EXECUTION_ERROR)
        # SECURITY: only fixed catalog commands, run as an argv list (no shell).
        command = LINUX_APP_LAUNCH_TARGETS.get(action.text.strip().lower())
        if command is None:
            return self._fail(
                action,
                f"launch_app: '{action.text.strip()}' is not a recognized application; "
                "only catalog apps can be launched.",
                FailureCategory.EXECUTION_ERROR,
            )
        try:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                command.split(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=_desktop._launch_env(),
            )
            if self._current_run_id is not None:
                self._launched_processes.setdefault(self._current_run_id, []).append(proc)
            await asyncio.sleep(self._post_launch_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Launched '{command}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Launch failed: {exc}", FailureCategory.EXECUTION_ERROR)

    def cleanup_run(self, run_id: str) -> int:
        """End the apps this run launched: each one's whole process group."""
        procs = self._launched_processes.pop(run_id, [])
        self._launched_app_names.pop(run_id, None)
        self._owned_pids.pop(run_id, None)
        closed = 0
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue  # the app and its children are gone already
            except Exception:
                logger.debug("cleanup: SIGTERM failed for pid %d", proc.pid, exc_info=True)
                continue
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    logger.debug("cleanup: SIGKILL failed for pid %d", proc.pid, exc_info=True)
            closed += 1
        return closed

    # ── the task page's live view ────────────────────────────────

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
            await asyncio.to_thread(gui.write, text, interval=0.01)
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
