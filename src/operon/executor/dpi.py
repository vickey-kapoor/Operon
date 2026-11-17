"""DPI helpers for the Windows desktop executor (KAN-90).

Coordinate pipeline, end to end:

1. ``mss`` grabs the screen. Its pixels, ``EnumDisplayMonitors`` rects and the click API
   (``SetCursorPos``/``SendInput`` through pyautogui) all live in the *process's own*
   coordinate space, so they agree with one another whatever the awareness is.
2. The frame is downscaled for the model; the loop scales the model's point back up and adds
   the monitor origin (``agent/loop.py``, tests in ``tests/models/test_monitor_offset.py``).

Where it can break: pyautogui calls ``SetProcessDPIAware()`` when it is imported, which locks
the process at *system-aware*. On a monitor whose scale differs from the system's (mixed DPI),
Windows then bitmap-stretches that monitor: the pixels the model sees are not the real ones,
so targets blur and drift. ``OPERON_DPI_AWARE=on`` sets per-monitor-v2 *before* pyautogui is
imported (``apply_awareness_early``, called at the top of ``desktop.py``). Then capture,
monitor rects and clicks are all physical pixels and nothing is converted.

``GetDpiForMonitor`` and ``EnumDisplayMonitors`` are themselves virtualized for an unaware or
system-aware process, so true values are read under a temporary per-monitor-v2 thread context.

Win32 sits in small helpers so tests patch them; off Windows they report nothing.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PER_MONITOR = "per_monitor"
SYSTEM = "system"
UNAWARE = "unaware"
UNKNOWN = "unknown"

_AWARENESS_NAMES = {0: UNAWARE, 1: SYSTEM, 2: PER_MONITOR}
_PMV2 = -4  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2

_warned: set[str] = set()


@dataclass(frozen=True)
class MonitorInfo:
    left: int
    top: int
    right: int
    bottom: int
    dpi: int = 96

    @property
    def scale_pct(self) -> int:
        return round(self.dpi / 96 * 100)


def get_awareness() -> str:
    """Effective DPI awareness of the calling thread: per_monitor, system, unaware or unknown."""
    if sys.platform != "win32":
        return UNKNOWN
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        ctx = user32.GetThreadDpiAwarenessContext()
        return _AWARENESS_NAMES.get(int(user32.GetAwarenessFromDpiAwarenessContext(ctx)), UNKNOWN)
    except (AttributeError, OSError):
        return UNKNOWN


def set_process_per_monitor_v2() -> bool:
    """Ask Windows for per-monitor-v2 for the whole process. False when it did not take."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        return bool(user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(_PMV2)))
    except (AttributeError, OSError):
        return False


def get_system_dpi() -> int:
    if sys.platform != "win32":
        return 96
    try:
        return int(ctypes.windll.user32.GetDpiForSystem())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return 96


def enum_monitors() -> list[MonitorInfo]:
    """Every display's TRUE physical rect and effective DPI, whatever the process awareness.

    Read under a temporary per-monitor-v2 thread context (restored afterwards), because the
    plain calls return virtualized values to an unaware or system-aware process.
    """
    if sys.platform != "win32":
        return []
    found: list[MonitorInfo] = []
    user32 = None
    old_ctx = None
    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        shcore = ctypes.windll.shcore  # type: ignore[attr-defined]
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        old_ctx = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PMV2))
        proc_t = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT), ctypes.c_void_p,
        )

        def _cb(hmon, _hdc, rect, _data) -> int:
            dpi_x, dpi_y = ctypes.c_uint(96), ctypes.c_uint(96)
            # MDT_EFFECTIVE_DPI = 0
            if shcore.GetDpiForMonitor(ctypes.c_void_p(hmon), 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)) != 0:
                dpi_x.value = 96
            r = rect.contents
            found.append(MonitorInfo(r.left, r.top, r.right, r.bottom, int(dpi_x.value)))
            return 1

        user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, proc_t, ctypes.c_void_p]
        user32.EnumDisplayMonitors(None, None, proc_t(_cb), None)
    except Exception as exc:
        logger.debug("enum_monitors: %s", exc)
    finally:
        if user32 is not None and old_ctx:
            try:
                user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(old_ctx))
            except Exception:
                pass
    return found


def match_monitor(mss_monitors: list[dict], rect: tuple[int, int, int, int]) -> dict | None:
    """The mss monitor whose full rect equals ``rect`` (left, top, right, bottom)."""
    for mon in mss_monitors:
        if (mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"]) == rect:
            return mon
    return None


def drift_expected(awareness: str, system_dpi: int, monitors: list[MonitorInfo]) -> bool:
    """True when clicks can really drift: unaware, or system-aware with a monitor off the system DPI."""
    if awareness == UNAWARE:
        return True
    if awareness == SYSTEM:
        return any(m.dpi != system_dpi for m in monitors)
    return False


def display_report() -> dict:
    """Process awareness plus every monitor's true DPI and scale, for logs and displays.json."""
    return {
        "awareness": get_awareness(),
        "system_dpi": get_system_dpi(),
        "monitors": [
            {"rect": [m.left, m.top, m.right, m.bottom], "dpi": m.dpi, "scale_pct": m.scale_pct}
            for m in enum_monitors()
        ],
    }


def _warn_once(key: str, message: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def warn_if_drifting() -> str:
    """Warn once, only when it matters (see ``drift_expected``). Returns the awareness."""
    awareness = get_awareness()
    if awareness in (SYSTEM, UNAWARE) and drift_expected(awareness, get_system_dpi(), enum_monitors()):
        _warn_once(
            "drift",
            "CoordDriftWarning: process DPI awareness is %s and a display's scale differs from what the "
            "process sees, so clicks can drift. Fix: set OPERON_DPI_AWARE=on (per-monitor-v2 is then set "
            "before pyautogui loads).",
            awareness,
        )
    return awareness


def apply_awareness_early() -> str:
    """OPERON_DPI_AWARE: set per-monitor-v2 now, before pyautogui is imported. Returns the awareness.

    Does nothing when the flag is off or off Windows. If the awareness is still not per-monitor
    afterwards, something else already fixed it: warn once and name the likely culprit.
    """
    if sys.platform != "win32":
        return UNKNOWN
    try:
        from operon.core.settings import get_settings

        if not get_settings().dpi_aware:
            return UNKNOWN
    except Exception as exc:  # a broken config is reported at startup, not here
        logger.debug("apply_awareness_early: settings unavailable: %s", exc)
        return UNKNOWN
    set_process_per_monitor_v2()
    awareness = get_awareness()
    if awareness != PER_MONITOR:
        early = sorted(m for m in ("pyautogui", "pyscreeze", "mss", "tkinter", "PIL.ImageGrab") if m in sys.modules)
        _warn_once(
            "aware",
            "OPERON_DPI_AWARE is on but the process is still %s: something fixed the DPI awareness before "
            "Operon (already imported: %s; or an application manifest). Import operon.executor.desktop "
            "first, or start Operon from a clean interpreter.",
            awareness,
            ", ".join(early) or "none of the known suspects",
        )
    return awareness
