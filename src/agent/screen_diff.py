"""Pixel-level screen change detection between before/after screenshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class TemporalSaliencyResult:
    """Pixel-velocity analysis of a sequence of frames from a recorded clip."""

    confidence: float        # 0.0–1.0: how likely the action had a real effect
    motion_class: str        # "hung" | "spinner" | "progressing"
    velocity_mean: float     # mean fraction of pixels changing per frame pair
    velocity_variance: float # variance of per-frame-pair velocities


# Screen is effectively frozen below this velocity.
_VELOCITY_HUNG = 0.0005
# Below this, motion is too faint to distinguish from a spinner.
_VELOCITY_SPINNER = 0.003
# Coefficient-of-variation below this → periodic (spinner-like) motion.
_CV_SPINNER_THRESHOLD = 0.5


def compute_temporal_saliency(frames: list) -> TemporalSaliencyResult:
    """Classify screen motion from a sequence of numpy BGR frames.

    Computes per-frame-pair pixel velocity and uses the coefficient of
    variation to distinguish periodic spinner motion from directed progress.
    Returns a TemporalSaliencyResult with a 0–1 confidence score.
    """
    import numpy as np

    if len(frames) < 2:
        return TemporalSaliencyResult(0.0, "hung", 0.0, 0.0)

    velocities: list[float] = []
    for prev, curr in zip(frames, frames[1:]):
        prev_g = prev.astype(np.float32).mean(axis=2)
        curr_g = curr.astype(np.float32).mean(axis=2)
        diff = np.abs(curr_g - prev_g)
        velocities.append(float((diff > _PIXEL_DIFF_THRESHOLD).sum()) / diff.size)

    mean_v = float(np.mean(velocities))
    var_v = float(np.var(velocities))

    if mean_v < _VELOCITY_HUNG:
        return TemporalSaliencyResult(0.0, "hung", mean_v, var_v)

    # Coefficient of variation: low → consistent periodic motion (spinner).
    cv = (var_v ** 0.5) / mean_v if mean_v > 0 else 0.0
    if cv < _CV_SPINNER_THRESHOLD or mean_v < _VELOCITY_SPINNER:
        return TemporalSaliencyResult(0.35, "spinner", mean_v, var_v)

    # Aperiodic motion above the spinner band → real directed progress.
    confidence = min(0.9, 0.55 + mean_v * 50.0)
    return TemporalSaliencyResult(confidence, "progressing", mean_v, var_v)
