"""Tests for the grounding helper (element_center)."""

from __future__ import annotations

from types import SimpleNamespace

from operon.agent.actions.grounding import element_center


def _el(x: int = 100, y: int = 200, width: int = 40, height: int = 20):
    return SimpleNamespace(x=x, y=y, width=width, height=height)


def test_element_center_is_geometric_center() -> None:
    assert element_center(_el(x=100, y=200, width=40, height=20)) == (120, 210)


def test_element_center_floors_offset_to_one_px() -> None:
    # Zero-size box → max(1, 0) keeps the point inside the element, not at its corner.
    assert element_center(_el(x=10, y=20, width=0, height=0)) == (11, 21)
