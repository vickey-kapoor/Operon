"""Pure geometry helpers used to ground labels to targets (perception.geometry)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.agent.perception.geometry import (
    bbox_center,
    bbox_distance,
    horizontal_overlap,
    is_above,
    is_left_of,
    same_column,
    same_row,
    vertical_overlap,
)


def _box(x: int, y: int, width: int = 100, height: int = 20):
    return SimpleNamespace(x=x, y=y, width=width, height=height)


def test_bbox_center_is_the_middle_of_the_box() -> None:
    assert bbox_center(_box(10, 20, 40, 10)) == (30.0, 25.0)


def test_bbox_distance_is_euclidean_between_centers() -> None:
    assert bbox_distance(_box(0, 0, 10, 10), _box(30, 40, 10, 10)) == pytest.approx(50.0)


def test_overlap_is_relative_to_the_smaller_box() -> None:
    small = _box(0, 0, 10, 10)
    big = _box(0, 5, 100, 100)
    assert vertical_overlap(small, big) == pytest.approx(0.5)
    assert horizontal_overlap(small, big) == pytest.approx(1.0)


def test_disjoint_boxes_have_zero_overlap() -> None:
    assert vertical_overlap(_box(0, 0), _box(0, 500)) == 0
    assert horizontal_overlap(_box(0, 0), _box(500, 0)) == 0


def test_zero_size_box_does_not_divide_by_zero() -> None:
    assert vertical_overlap(_box(0, 0, 0, 0), _box(0, 0)) == 0
    assert horizontal_overlap(_box(0, 0, 0, 0), _box(0, 0)) == 0


def test_same_row_by_center_tolerance_or_overlap() -> None:
    assert same_row(_box(0, 100), _box(300, 115))  # centers 15 px apart
    assert not same_row(_box(0, 100), _box(300, 140))  # 40 px apart, no overlap
    tall = _box(300, 60, 100, 100)  # center 50 px away, but overlaps the row fully
    assert same_row(_box(0, 100), tall)


def test_same_column_by_center_tolerance_or_overlap() -> None:
    assert same_column(_box(100, 0, 40), _box(120, 300, 40))
    assert not same_column(_box(100, 0, 40), _box(200, 300, 40))


def test_is_above_needs_a_small_gap_and_horizontal_alignment() -> None:
    label = _box(100, 100, 100, 20)  # bottom at 120
    assert is_above(label, _box(100, 130))
    assert not is_above(label, _box(100, 200))  # 80 px gap: too far
    assert not is_above(label, _box(300, 130))  # 200 px off to the side
    assert not is_above(label, _box(100, 110))  # overlapping: negative gap


def test_is_left_of_needs_a_small_gap_and_vertical_alignment() -> None:
    label = _box(0, 100, 100, 20)  # right edge at 100
    assert is_left_of(label, _box(150, 100))
    assert not is_left_of(label, _box(400, 100))  # 300 px gap: too far
    assert not is_left_of(label, _box(150, 200))  # 100 px lower
    assert not is_left_of(label, _box(50, 100))  # overlapping: negative gap
