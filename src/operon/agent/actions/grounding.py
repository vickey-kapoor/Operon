"""Grounding helper: the shared "bbox -> click point" formula."""

from __future__ import annotations


def element_center(element: object) -> tuple[int, int]:
    """Return the click center for a perceived element / target signature.

    The shared formula used everywhere a bbox is turned into a click point:
    the geometric center, with a 1px floor so zero-width/height boxes still
    yield a point inside the element rather than its top-left corner.
    """
    return (
        element.x + max(1, element.width // 2),  # type: ignore[attr-defined]
        element.y + max(1, element.height // 2),  # type: ignore[attr-defined]
    )
