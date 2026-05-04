"""Desktop executor for full-screen computer-use automation."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import mss
import mss.tools
import pyperclip

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

from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage
from src.models.execution import (
    ExecutedAction,
    ExecutionAttemptTrace,
    ExecutionTrace,
)
from src.models.policy import ActionType, AgentAction

from .browser import Executor

logger = logging.getLogger(__name__)

# Windows-only: pass to subprocess.run/Popen so the spawned process does NOT
# allocate a conhost.exe console window — that flash steals focus from the
# user's foreground app on every taskkill / app-launch invocation. 0 on
# non-Windows is a no-op (creationflags is ignored outside win32).
_NO_CONSOLE = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

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


class HardwareBaselineError(RuntimeError):
    """Raised when the display hardware baseline check fails.

    Coordinate targeting is geometry-sensitive: a sub-minimum resolution or
    severely mismatched DPI scaling produces systematic drift that cannot be
    corrected at runtime.  Stop early rather than execute hallucinated clicks.
    """


def validate_display_baseline(*, require_min_resolution: tuple[int, int] = (1280, 720)) -> None:
    """Check DPI scaling and primary-monitor resolution against automation baseline.

    DPI at 100% (96 DPI):  coordinates map 1:1 between logical and physical pixels.
    DPI != 100%:            logs a CoordDriftWarning — many systems legitimately run
                            at 125 % or 150 % with per-monitor DPI awareness enabled,
                            so this is a warning, not a hard stop.
    Resolution < minimum:   raises HardwareBaselineError — too little screen space for
                            reliable element targeting.
    """
    if sys.platform != "win32":
        return

    # --- DPI scaling ---
    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()  # type: ignore[attr-defined]
        scale_pct = round(dpi / 96 * 100)
        if scale_pct != 100:
            logger.warning(
                "CoordDriftWarning: DPI scaling is %d%% (expected 100%% / 96 DPI). "
                "Logical↔physical pixel ratio is %.2fx — coordinate drift is possible "
                "if per-monitor DPI awareness is not fully applied.",
                scale_pct,
                dpi / 96,
            )
    except (AttributeError, OSError) as exc:
        logger.debug("validate_display_baseline: DPI check skipped: %s", exc)

    # --- Resolution ---
    try:
        with mss.mss() as sct:
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
        logger.debug("Display baseline OK: %dx%d @ %d%% DPI", w, h, scale_pct if "scale_pct" in dir() else "?")
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
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # type: ignore[attr-defined]
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
        with mss.mss() as sct:
            for mon in sct.monitors[1:]:  # skip monitors[0] (virtual combined)
                # mss monitor dicts use left/top/width/height; cross-reference via
                # the monitor's origin to match the HMONITOR returned by Win32.
                info = ctypes.create_string_buffer(40)
                ctypes.c_uint32.from_buffer(info, 0).value = 40  # cbSize
                if user32.GetMonitorInfoA(hmon, info):
                    # rcMonitor occupies bytes 4-20: left, top, right, bottom (4×int32)
                    left, top, right, bottom = (
                        ctypes.c_int32.from_buffer(info, 4 + i * 4).value for i in range(4)
                    )
                    if mon["left"] == left and mon["top"] == top:
                        return mon
    except Exception as exc:
        logger.debug("_foreground_monitor: %s", exc)
    return None


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
}


def _normalize_key_pyautogui(key: str) -> str:
    return _PYAUTOGUI_KEY_MAP.get(key.lower(), key.lower())


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
        artifact_dir: str | Path = ".desktop-artifacts",
        type_interval: float = 0.03,
        post_action_delay: float = 0.15,
        post_launch_delay: float = 0.8,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._type_interval = type_interval
        self._post_action_delay = post_action_delay
        self._post_launch_delay = post_launch_delay
        self._current_run_id: str | None = None
        self._launched_processes: dict[str, list[subprocess.Popen]] = {}
        self._launched_app_names: dict[str, list[str]] = {}  # run_id → exe names for taskkill fallback
        self._run_recorders: dict[str, object] = {}   # run_id → ScreenRecorder
        self._run_video_paths: dict[str, Path] = {}   # run_id → final video path

        _set_dpi_awareness()
        if os.getenv("OPERON_TEST_SAFE_MODE", "false").lower() != "true":
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
            with mss.mss() as sct:
                if len(sct.monitors) < 2:
                    return 0.0
                mon = sct.monitors[1]
                mon_w, mon_h = mon["width"], mon["height"]
                mon_l, mon_t = mon["left"], mon["top"]

            rng = random.Random(42)  # reproducible seed for calibration
            margin = crop_radius + 2
            variances: list[float] = []
            with mss.mss() as sct:
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
            with mss.mss() as sct:
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
            with mss.mss() as sct:
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

    def cleanup_run(self, run_id: str) -> int:
        """Terminate all non-protected processes launched during *run_id*.

        Returns the count of processes successfully terminated.
        Uses taskkill as a fallback for apps where the launcher process
        exited immediately (e.g. Chrome, Settings via ms-* URIs).
        """
        procs = self._launched_processes.pop(run_id, [])
        app_names = self._launched_app_names.pop(run_id, [])
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
        # (Chrome, Edge, etc. spawn child processes and the parent exits)
        for app_exe in app_names:
            if app_exe in _PROTECTED_PROCESSES:
                continue
            try:
                result = subprocess.run(
                    ["taskkill", "/IM", app_exe, "/F"],
                    capture_output=True, timeout=5,
                    creationflags=_NO_CONSOLE,
                )
                if result.returncode == 0:
                    closed += 1
                    logger.info("cleanup: taskkill terminated %s", app_exe)
            except Exception:
                logger.debug("cleanup: taskkill failed for %s", app_exe, exc_info=True)

        return closed

    async def capture(self) -> CaptureFrame:
        """Take a 3-frame burst screenshot and return the last frame with visual velocity."""
        frame, velocity = await asyncio.to_thread(self._capture_burst_sync)
        if velocity > 0.02:
            # Screen is still animating — wait and re-capture once before perceiving
            logger.debug("High visual velocity (%.3f) detected; waiting 300ms before re-capture", velocity)
            await asyncio.sleep(0.3)
            frame, _ = await asyncio.to_thread(self._capture_burst_sync)
        return frame

    def _capture_burst_sync(self) -> tuple[CaptureFrame, float]:
        """Capture t0, t+100ms, t+200ms frames; return (last_frame, visual_velocity)."""
        import time as _time
        with mss.mss() as sct:
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
            ActionType.FILE_PORTER: self._exec_file_porter,
        }
        handler = dispatch.get(at, self._exec_unsupported)
        return await handler(action)

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
            await asyncio.sleep(self._post_action_delay)
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
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Double-clicked at ({action.x}, {action.y})", after_path).model_copy(update={"visual_variance": variance})
        except Exception as exc:
            return self._fail(action, f"Double-click failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _exec_right_click(self, action: AgentAction) -> ExecutedAction:
        gate, variance = await self._click_preamble(action, "right_click")
        if gate is not None:
            return gate
        try:
            await asyncio.to_thread(pyautogui.rightClick, action.x, action.y)
            await asyncio.sleep(self._post_action_delay)
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
            # Clipboard-paste avoids character-ordering issues with pyautogui.write()
            # when typing long strings or strings containing special characters like $
            await asyncio.to_thread(pyperclip.copy, action.text)
            await asyncio.to_thread(pyautogui.hotkey, 'ctrl', 'v')
            if action.press_enter:
                await asyncio.sleep(0.05)
                await asyncio.to_thread(pyautogui.press, 'enter')
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Typed '{action.text}'", after_path)
        except Exception as exc:
            return self._fail(action, f"Type failed: {exc}", FailureCategory.EXECUTION_ERROR)

    def _sample_region(self, x: int, y: int, radius: int = 30) -> bytes:
        """Capture a small pixel region around (x, y) for focus-change detection."""
        with mss.mss() as sct:
            region = {
                "left": max(0, x - radius),
                "top": max(0, y - radius),
                "width": radius * 2,
                "height": radius * 2,
            }
            return bytes(sct.grab(region).rgb)

    async def _exec_press_key(self, action: AgentAction) -> ExecutedAction:
        if action.key is None:
            return self._fail(action, "press_key requires key", FailureCategory.EXECUTION_ERROR)
        key = _normalize_key_pyautogui(action.key)
        try:
            await asyncio.to_thread(pyautogui.press, key)
            await asyncio.sleep(self._post_action_delay)
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
                proc = await asyncio.to_thread(
                    subprocess.Popen,
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_NO_CONSOLE,
                )
                # Track process for cleanup
                if self._current_run_id is not None:
                    self._launched_processes.setdefault(self._current_run_id, []).append(proc)
            # Also track the exe name for taskkill fallback (handles Chrome, etc.
            # where the launcher process exits but the app keeps running)
            if self._current_run_id is not None:
                exe_name = Path(command.split()[0]).name.lower()
                if not exe_name.endswith(".exe"):
                    exe_name += ".exe"
                self._launched_app_names.setdefault(self._current_run_id, []).append(exe_name)
            await asyncio.sleep(self._post_launch_delay)
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

    async def _exec_file_porter(self, action: AgentAction) -> ExecutedAction:
        if action.url is None or action.text is None:
            return self._fail(
                action,
                "file_porter requires url and text (folder ID)",
                FailureCategory.EXECUTION_ERROR,
            )
        try:
            from src.tools.file_porter import run_porter
            result = await asyncio.to_thread(run_porter, action.url, action.text)
            if result.success:
                return self._ok(action, result.detail, None)
            return self._fail(action, result.detail, FailureCategory.EXECUTION_ERROR)
        except Exception as exc:
            return self._fail(action, f"FilePorter raised: {exc}", FailureCategory.EXECUTION_ERROR)

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

    async def start_run_recording(self, run_id: str, root_dir: str | Path = "runs") -> None:
        """Start a full-run screen recording for *run_id*.

        Uses streaming mode so frames are written directly to disk — no in-memory
        buffer regardless of run length.  The video is saved to
        ``runs/<run_id>/run_recording.mp4`` at 8 fps, half native resolution.
        Safe to call multiple times for the same run_id (no-op if already recording).
        """
        if run_id in self._run_recorders:
            return
        from src.agent.screen_recorder import ScreenRecorder

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

    async def execute_with_recording(
        self, action: AgentAction, step_dir: Path,
    ) -> tuple[ExecutedAction, Path | None]:
        """Re-execute an action while recording the screen for video verification."""
        from src.agent.screen_recorder import ScreenRecorder

        video_path = step_dir / "verification_recording.mp4"
        recorder = ScreenRecorder(output_path=video_path)
        await recorder.start()
        result = await self.execute(action)
        await asyncio.sleep(2.0)  # let UI settle after action
        final_path = await recorder.stop()
        return result, final_path
