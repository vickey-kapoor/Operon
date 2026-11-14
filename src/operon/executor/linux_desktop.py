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
import logging
import os
import signal
import subprocess

from operon.agent.app_catalog import LINUX_APP_LAUNCH_TARGETS
from operon.executor import desktop as _desktop
from operon.executor.live_view import LiveViewMixin
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import AgentAction

logger = logging.getLogger(__name__)


class LinuxDesktopExecutor(LiveViewMixin, _desktop.DesktopExecutor):
    """Desktop executor for an X11 Linux desktop (XFCE on the Azure VM)."""

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
