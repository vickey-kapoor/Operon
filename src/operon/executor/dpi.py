"""DPI helpers for the Windows desktop executor (KAN-90).

Coordinate pipeline, end to end:

1. ``mss`` grabs the screen. A per-monitor-v2 aware process gets *physical* pixels
   for every monitor, in one virtual-desktop space (negative origins are normal).
2. The frame is downscaled for the model, and the loop scales the model's point
   back up to those physical screenshot pixels and adds the monitor origin
   (``agent/loop.py``, tests in ``tests/models/test_monitor_offset.py``).
3. ``pyautogui`` clicks through ``SetCursorPos``/``SendInput``. In a per-monitor
   aware process that API also takes physical pixels, so steps 1-3 agree and
   nothing needs converting.

Where it can break:

* (a) Awareness never took effect. ``SetProcessDpiAwarenessContext`` fails when
  something else (python, a manifest, another library) already fixed the
  awareness. The process stays DPI-unaware or system-aware, Windows virtualizes
  the click API into logical pixels, and the click lands at ``x * 96 / dpi`` of
  where the screenshot pixel is: 125%, 150%, 175% drift. ``get_awareness`` reads
  the real state so this is detected, not assumed.
* (b) Mixed-DPI monitors. A virtualized process sees each monitor at a
  different factor, so the point must be converted with the scale of the
  monitor it lies on (``to_click_coords``).
* (c) Picking the capture monitor from the foreground window by left/top alone
  confuses monitors that share a left edge (``match_monitor`` uses the full rect).

Everything that touches Win32 sits in the small helpers below so tests patch
them; off Windows they report nothing.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PER_MONITOR = "per_monitor"
SYSTEM = "system"
UNAWARE = "unaware"
UNKNOWN = "unknown"

_AWARENESS_NAMES = {0: UNAWARE, 1: SYSTEM, 2: PER_MONITOR}


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

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


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


def get_system_dpi() -> int:
    if sys.platform != "win32":
        return 96
    try:
        return int(ctypes.windll.user32.GetDpiForSystem())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return 96


def enum_monitors() -> list[MonitorInfo]:
    """Every display with its rect and effective DPI (EnumDisplayMonitors + GetDpiForMonitor)."""
    if sys.platform != "win32":
        return []
    found: list[MonitorInfo] = []
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        shcore = ctypes.windll.shcore  # type: ignore[attr-defined]
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
    return found


def monitor_from_point(x: int, y: int, monitors: list[MonitorInfo] | None = None) -> MonitorInfo | None:
    """The monitor whose rect contains the point (physical virtual-desktop pixels)."""
    for mon in monitors if monitors is not None else enum_monitors():
        if mon.contains(x, y):
            return mon
    return None


def match_monitor(mss_monitors: list[dict], rect: tuple[int, int, int, int]) -> dict | None:
    """The mss monitor whose full rect equals ``rect`` (left, top, right, bottom)."""
    for mon in mss_monitors:
        if (mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"]) == rect:
            return mon
    return None


def to_click_coords(
    x: int,
    y: int,
    *,
    awareness: str,
    monitors: list[MonitorInfo],
    system_dpi: int = 96,
) -> tuple[int, int]:
    """Convert a physical screenshot coordinate to what the click API expects.

    Per-monitor aware (or unknown): identity, the click API is physical too.
    System-aware: the click space is the system DPI, so the factor is
    ``system_dpi / monitor_dpi``. Unaware: the click space is 96 DPI. The point is
    scaled around its monitor's origin, which keeps negative-origin monitors in place.
    """
    if awareness not in (SYSTEM, UNAWARE):
        return x, y
    mon = monitor_from_point(x, y, monitors)
    if mon is None:
        return x, y
    target = system_dpi if awareness == SYSTEM else 96
    factor = target / mon.dpi
    return round(mon.left + (x - mon.left) * factor), round(mon.top + (y - mon.top) * factor)


def display_report() -> dict:
    """Awareness plus per-monitor DPI and scale, for logs and the run's host.json."""
    return {
        "awareness": get_awareness(),
        "system_dpi": get_system_dpi(),
        "monitors": [
            {"rect": [m.left, m.top, m.right, m.bottom], "dpi": m.dpi, "scale_pct": m.scale_pct}
            for m in enum_monitors()
        ],
    }


def log_awareness() -> str:
    """Warn, with the fix, when the process is not per-monitor aware. Cheap: always on."""
    awareness = get_awareness()
    if awareness in (SYSTEM, UNAWARE):
        logger.warning(
            "CoordDriftWarning: process DPI awareness is %s, not per-monitor. Another library likely set it "
            "before Operon. Clicks drift on any display not at 100%%. Fix: launch Operon from a clean "
            "interpreter (import nothing that sets DPI awareness first), or set OPERON_DPI_CHECK=on so "
            "click coordinates are converted per monitor.",
            awareness,
        )
    return awareness
