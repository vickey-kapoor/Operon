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



def count_changed_pixels(before_path: str, after_path: str, *, diff_threshold: int = _PIXEL_DIFF_THRESHOLD) -> int | None:
    """Number of pixels that differ by more than ``diff_threshold`` in any channel,
    at native resolution (no downscale, so a 4 px checkbox tick still counts).

    Returns None when either file is missing or cannot be decoded, so a caller can
    fail open instead of reading an error as "nothing changed". Differently sized
    frames count every pixel as changed.
    """
    from PIL import ImageChops

    try:
        with Image.open(before_path) as b_img, Image.open(after_path) as a_img:
            before = b_img.convert("RGB")
            after = a_img.convert("RGB")
            if before.size != after.size:
                return before.size[0] * before.size[1]
            mask = None
            for channel in ImageChops.difference(before, after).split():
                over = channel.point(lambda v: 255 if v > diff_threshold else 0)
                mask = over if mask is None else ImageChops.lighter(mask, over)
            return int(mask.histogram()[255]) if mask is not None else 0
    except Exception:
        logger.debug("screen_diff: native comparison failed", exc_info=True)
        return None


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


# ── stronger test for a focus-click spot: plain *background*, not just uniform ──
# A 100px square can be uniform and still be a big neutral button, a card or a canvas. The
# click spot must also sit in a wider ring that is uniform, and match the screen's own
# dominant (background) colour.
WIDE_REGION_RADIUS = 100
BACKGROUND_COLOR_TOLERANCE = 18  # per channel, 0-255


def dominant_color(rgb: bytes, stride: int = 16) -> tuple[int, int, int]:
    """The most common colour in raw RGB bytes (histogram mode over 5-bit quantised pixels)."""
    import numpy as np

    px = np.frombuffer(rgb, dtype=np.uint8)
    px = px[: len(px) // 3 * 3].reshape(-1, 3)[::stride]
    if len(px) == 0:
        return (255, 255, 255)
    quant = px >> 3
    keys = (quant[:, 0].astype(np.int32) << 10) | (quant[:, 1].astype(np.int32) << 5) | quant[:, 2]
    mode = int(np.bincount(keys).argmax())
    members = px[keys == mode]
    r, g, b = (int(v) for v in members.mean(axis=0))
    return (r, g, b)


def png_dominant_color(png: bytes) -> tuple[int, int, int]:
    import io

    with Image.open(io.BytesIO(png)) as img:
        return dominant_color(img.convert("RGB").tobytes())


def channel_variance(rgb: bytes) -> float:
    """Largest per-channel variance of raw RGB bytes (a flat colour of any hue is 0)."""
    import numpy as np

    px = np.frombuffer(rgb, dtype=np.uint8)
    px = px[: len(px) // 3 * 3].reshape(-1, 3)
    if len(px) == 0:
        return 0.0
    return float(px.astype(np.float64).var(axis=0).max())


def _mean_color(rgb: bytes) -> tuple[float, float, float]:
    import numpy as np

    px = np.frombuffer(rgb, dtype=np.uint8)
    px = px[: len(px) // 3 * 3].reshape(-1, 3)
    m = px.mean(axis=0) if len(px) else np.array([255.0, 255.0, 255.0])
    return (float(m[0]), float(m[1]), float(m[2]))


def is_plain_background(
    inner: bytes,
    wide: bytes,
    background: tuple[int, int, int],
    threshold: float = BLANK_VARIANCE_THRESHOLD,
    tolerance: int = BACKGROUND_COLOR_TOLERANCE,
) -> bool:
    """True when both the inner sample and the wider ring are uniform and near ``background``."""
    if not inner or not wide:
        return False
    if channel_variance(inner) > threshold or channel_variance(wide) > threshold:
        return False
    return all(
        abs(c - b) <= tolerance
        for sample in (inner, wide)
        for c, b in zip(_mean_color(sample), background, strict=True)
    )
