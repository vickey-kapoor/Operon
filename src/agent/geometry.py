"""Pure geometric helpers for vision-first selector grounding."""

from __future__ import annotations

from math import hypot

from src.models.perception import UIElement

ROW_TOLERANCE_PX = 22
COLUMN_TOLERANCE_PX = 28
LABEL_ABOVE_MAX_VERTICAL_GAP_PX = 56
LABEL_ABOVE_MAX_HORIZONTAL_OFFSET_PX = 84
LABEL_LEFT_MAX_HORIZONTAL_GAP_PX = 180
LABEL_LEFT_MAX_VERTICAL_OFFSET_PX = 34
NEARBY_LABEL_MAX_DISTANCE_PX = 220.0
GROUP_VERTICAL_GAP_PX = 40


def bbox_center(element: UIElement) -> tuple[float, float]:
    return (element.x + (element.width / 2.0), element.y + (element.height / 2.0))


def bbox_distance(a: UIElement, b: UIElement) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return hypot(ax - bx, ay - by)


def vertical_overlap(a: UIElement, b: UIElement) -> float:
    top = max(a.y, b.y)
    bottom = min(a.y + a.height, b.y + b.height)
    overlap = max(0, bottom - top)
    return overlap / max(1, min(a.height, b.height))


def horizontal_overlap(a: UIElement, b: UIElement) -> float:
    left = max(a.x, b.x)
    right = min(a.x + a.width, b.x + b.width)
    overlap = max(0, right - left)
    return overlap / max(1, min(a.width, b.width))


def same_row(a: UIElement, b: UIElement, tolerance: int = ROW_TOLERANCE_PX) -> bool:
    return abs(bbox_center(a)[1] - bbox_center(b)[1]) <= tolerance or vertical_overlap(a, b) >= 0.5


def same_column(a: UIElement, b: UIElement, tolerance: int = COLUMN_TOLERANCE_PX) -> bool:
    return abs(bbox_center(a)[0] - bbox_center(b)[0]) <= tolerance or horizontal_overlap(a, b) >= 0.5


def is_above(
    label_bbox: UIElement,
    target_bbox: UIElement,
    max_vertical_gap: int = LABEL_ABOVE_MAX_VERTICAL_GAP_PX,
    max_horizontal_offset: int = LABEL_ABOVE_MAX_HORIZONTAL_OFFSET_PX,
) -> bool:
    vertical_gap = target_bbox.y - (label_bbox.y + label_bbox.height)
    label_cx, _ = bbox_center(label_bbox)
    target_cx, _ = bbox_center(target_bbox)
    return 0 <= vertical_gap <= max_vertical_gap and abs(label_cx - target_cx) <= max_horizontal_offset


def is_left_of(
    label_bbox: UIElement,
    target_bbox: UIElement,
    max_horizontal_gap: int = LABEL_LEFT_MAX_HORIZONTAL_GAP_PX,
    max_vertical_offset: int = LABEL_LEFT_MAX_VERTICAL_OFFSET_PX,
) -> bool:
    horizontal_gap = target_bbox.x - (label_bbox.x + label_bbox.width)
    _, label_cy = bbox_center(label_bbox)
    _, target_cy = bbox_center(target_bbox)
    return 0 <= horizontal_gap <= max_horizontal_gap and abs(label_cy - target_cy) <= max_vertical_offset
