"""Pixel-level screen change detection between before/after screenshots."""

from __future__ import annotations

import logging
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
