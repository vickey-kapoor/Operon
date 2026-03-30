"""Pixel-level screen change detection between before/after screenshots."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Minimum fraction of pixels that must change to count as meaningful screen change.
SCREEN_CHANGE_THRESHOLD = 0.005  # 0.5%

# Below this, treat as cursor blink or compression artifact — no real change.
CURSOR_ONLY_THRESHOLD = 0.001  # 0.1%

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

        before = Image.open(before_path).convert("L").resize(
            (_COMPARE_WIDTH, _COMPARE_HEIGHT), Image.LANCZOS,
        )
        after = Image.open(after_path).convert("L").resize(
            (_COMPARE_WIDTH, _COMPARE_HEIGHT), Image.LANCZOS,
        )

        before_data = before.tobytes()
        after_data = after.tobytes()

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
