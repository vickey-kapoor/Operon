"""Pixel-level screen change detection between before/after screenshots."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Minimum fraction of pixels that must change to count as meaningful screen change.
# Set low enough to detect text appearing in Notepad and keyboard shortcut effects.
SCREEN_CHANGE_THRESHOLD = 0.002  # 0.2%

# Below this, treat as cursor blink or compression artifact — no real change.
CURSOR_ONLY_THRESHOLD = 0.0005  # 0.05%

# Internal comparison resolution (fast, sufficient for change detection).
_COMPARE_WIDTH = 480
_COMPARE_HEIGHT = 270

# Per-pixel intensity difference threshold to count as "changed".
_PIXEL_DIFF_THRESHOLD = 30


def compute_screen_change_ratio(before_path: str, after_path: str) -> float:
    """Return 0.0–1.0 ratio of pixels that changed between two screenshots.

    Both images are resized to a small resolution and converted to grayscale
    before comparison.  Returns 0.0 if either file is missing or unreadable.
    """
    try:
        if not Path(before_path).exists() or not Path(after_path).exists():
            return 0.0

        with Image.open(before_path) as _b:
            before_data = _b.convert("L").resize((_COMPARE_WIDTH, _COMPARE_HEIGHT), Image.LANCZOS).tobytes()
        with Image.open(after_path) as _a:
            after_data = _a.convert("L").resize((_COMPARE_WIDTH, _COMPARE_HEIGHT), Image.LANCZOS).tobytes()

        total = len(before_data)
        if total == 0:
            return 0.0

        changed = sum(
            1 for a, b in zip(before_data, after_data)
            if abs(a - b) > _PIXEL_DIFF_THRESHOLD
        )
        return changed / total

    except Exception:
        logger.debug("screen_diff: comparison failed", exc_info=True)
        return 0.0



# ── blank-region test (visual servo, shared) ────────────────────────────────
# A region is "blank" when its pixel variance is at or below this (pixel^2). The desktop
# executor calibrates its own threshold from this default; the browser executor uses it as is.
BLANK_VARIANCE_THRESHOLD = 20.0
BLANK_REGION_RADIUS = 50

# Where a recovery reset may click to put focus on the page/window background, as (x, y)
# fractions of the screen: the old fixed point first, then the margins and middle.
_RESET_POINT_FRACTIONS = (
    (0.5, 1 / 3), (0.5, 0.5), (0.85, 1 / 3), (0.15, 1 / 3), (0.5, 0.15),
    (0.5, 0.85), (0.9, 0.5), (0.1, 0.5), (0.9, 0.9), (0.1, 0.9),
)


def region_variance(raw: bytes) -> float:
    """Pixel variance of raw channel bytes (0.0 for an empty sample)."""
    if not raw:
        return 0.0
    mean = sum(raw) / len(raw)
    return sum((b - mean) ** 2 for b in raw) / len(raw)


def png_region_rgb(png: bytes, x: int, y: int, radius: int = BLANK_REGION_RADIUS) -> bytes:
    """RGB bytes of the square around (x, y) in a PNG/JPEG screenshot, clamped to the image."""
    import io

    with Image.open(io.BytesIO(png)) as img:
        rgb = img.convert("RGB")
        size = radius * 2
        left = min(max(0, x - radius), max(0, rgb.width - size))
        top = min(max(0, y - radius), max(0, rgb.height - size))
        return rgb.crop((left, top, left + size, top + size)).tobytes()


def reset_click_candidates(width: int, height: int, left: int = 0, top: int = 0) -> list[tuple[int, int]]:
    """Points to try for a focus-restoring click, most conventional first."""
    return [(left + int(width * fx), top + int(height * fy)) for fx, fy in _RESET_POINT_FRACTIONS]


def first_blank_point(
    candidates: list[tuple[int, int]], has_content: Callable[[int, int], bool]
) -> tuple[int, int] | None:
    """The first candidate whose region has no content, or None when every one has some."""
    for x, y in candidates:
        if not has_content(x, y):
            return (x, y)
    return None
