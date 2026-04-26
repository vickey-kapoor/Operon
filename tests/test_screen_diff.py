"""Tests for pixel-level screen change detection (screen_diff.py).

Tier: SIMPLE — pure pixel math, no mocks or async.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.agent.screen_diff import (
    CURSOR_ONLY_THRESHOLD,
    SCREEN_CHANGE_THRESHOLD,
    compute_screen_change_ratio,
)


def _solid_png(path: Path, color: int, width: int = 64, height: int = 64) -> str:
    """Write a solid-grey PNG and return the path string."""
    img = Image.new("L", (width, height), color=color)
    img.save(path)
    return str(path)


def _rgb_png(path: Path, rgb: tuple[int, int, int], width: int = 64, height: int = 64) -> str:
    """Write a solid-colour RGB PNG and return the path string."""
    img = Image.new("RGB", (width, height), color=rgb)
    img.save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Simple: file-handling edge cases
# ---------------------------------------------------------------------------

def test_missing_before_returns_zero(tmp_path: Path) -> None:
    after = _solid_png(tmp_path / "after.png", color=200)
    ratio = compute_screen_change_ratio(str(tmp_path / "missing.png"), after)
    assert ratio == 0.0


def test_missing_after_returns_zero(tmp_path: Path) -> None:
    before = _solid_png(tmp_path / "before.png", color=100)
    ratio = compute_screen_change_ratio(before, str(tmp_path / "missing.png"))
    assert ratio == 0.0


def test_both_missing_returns_zero(tmp_path: Path) -> None:
    ratio = compute_screen_change_ratio(
        str(tmp_path / "a.png"), str(tmp_path / "b.png")
    )
    assert ratio == 0.0


def test_identical_images_return_zero(tmp_path: Path) -> None:
    before = _solid_png(tmp_path / "before.png", color=128)
    after = _solid_png(tmp_path / "after.png", color=128)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# Simple: ratio magnitude checks
# ---------------------------------------------------------------------------

def test_completely_different_images_high_ratio(tmp_path: Path) -> None:
    """Black → white should produce a ratio well above SCREEN_CHANGE_THRESHOLD."""
    before = _solid_png(tmp_path / "before.png", color=0)
    after = _solid_png(tmp_path / "after.png", color=255)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio > SCREEN_CHANGE_THRESHOLD


def test_tiny_brightness_change_below_cursor_threshold(tmp_path: Path) -> None:
    """A 1-unit brightness shift should fall below CURSOR_ONLY_THRESHOLD."""
    before = _solid_png(tmp_path / "before.png", color=100)
    after = _solid_png(tmp_path / "after.png", color=101)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio < CURSOR_ONLY_THRESHOLD


def test_moderate_change_between_thresholds(tmp_path: Path) -> None:
    """A solid-colour change of 35 intensity units should sit above CURSOR_ONLY_THRESHOLD."""
    before = _solid_png(tmp_path / "before.png", color=50)
    after = _solid_png(tmp_path / "after.png", color=85)  # diff=35 > _PIXEL_DIFF_THRESHOLD=30
    ratio = compute_screen_change_ratio(before, after)
    assert ratio > CURSOR_ONLY_THRESHOLD


def test_ratio_bounded_zero_to_one(tmp_path: Path) -> None:
    before = _solid_png(tmp_path / "before.png", color=0)
    after = _solid_png(tmp_path / "after.png", color=255)
    ratio = compute_screen_change_ratio(before, after)
    assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# Moderate: format handling
# ---------------------------------------------------------------------------

def test_rgb_images_compared_correctly(tmp_path: Path) -> None:
    """RGB images should be accepted (converted to greyscale internally)."""
    before = _rgb_png(tmp_path / "before.png", (0, 0, 0))
    after = _rgb_png(tmp_path / "after.png", (255, 255, 255))
    ratio = compute_screen_change_ratio(before, after)
    assert ratio > SCREEN_CHANGE_THRESHOLD


def test_different_size_images_still_work(tmp_path: Path) -> None:
    """Internal resizing should allow mismatched input dimensions."""
    before = _solid_png(tmp_path / "before.png", color=0, width=100, height=200)
    after = _solid_png(tmp_path / "after.png", color=255, width=800, height=600)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio > SCREEN_CHANGE_THRESHOLD


def test_corrupted_file_returns_zero(tmp_path: Path) -> None:
    """Unreadable image file should not raise — returns 0.0."""
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")
    before = _solid_png(tmp_path / "before.png", color=128)
    ratio = compute_screen_change_ratio(before, str(bad))
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# Complex: threshold semantics (meaningful for the video verifier decision gate)
# ---------------------------------------------------------------------------

def test_screen_change_threshold_classifies_real_change(tmp_path: Path) -> None:
    """A solid content change (colour shift > 30) classifies as a real change."""
    before = _solid_png(tmp_path / "before.png", color=0)
    after = _solid_png(tmp_path / "after.png", color=200)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio >= SCREEN_CHANGE_THRESHOLD, (
        f"Expected ratio >= {SCREEN_CHANGE_THRESHOLD}, got {ratio}"
    )


def test_cursor_only_threshold_classifies_non_change(tmp_path: Path) -> None:
    """A <1-unit brightness shift is below the cursor-only noise floor."""
    before = _solid_png(tmp_path / "before.png", color=150)
    after = _solid_png(tmp_path / "after.png", color=150)
    ratio = compute_screen_change_ratio(before, after)
    assert ratio < CURSOR_ONLY_THRESHOLD
