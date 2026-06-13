"""Grounding seam — RFC 0001, Move 3's companion (Move 2, increment 1).

"Grounding" is the mapping from an intended target (an element id / label /
intent) to the concrete (x, y) pixel coordinate that gets clicked. Today the
selection logic lives in :class:`DeterministicTargetSelector`, and every caller
that turns a selected element into coordinates repeats the same center formula.

This module introduces a single seam for that mapping:

- :func:`element_center` — the one shared "bbox -> center" formula (previously
  duplicated across the loop and retry-hardening paths).
- :class:`GroundingResult` — a typed result carrying the coordinate, the matched
  element id, and the selector trace.
- :class:`Grounder` — the protocol an alternative grounder (set-of-marks, a
  learned grounding model, ...) can implement to be swapped in.
- :class:`DeterministicGrounder` — the baseline, wrapping the existing
  ``DeterministicTargetSelector`` with identical behavior.

This increment introduces the abstraction and routes the re-resolution path
through it; folding the primary loop selection path onto the same interface is a
follow-up increment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from operon.models.perception import ScreenPerception
from operon.models.selector import TargetIntent, TargetSelectionContext


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


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of grounding an intent to a coordinate.

    ``grounded`` is False when no safe deterministic match was found; in that
    case ``x``/``y``/``element_id``/``element`` are None and the caller should
    fall back (e.g. keep the prior coordinates or escalate).
    """

    grounded: bool
    element_id: str | None = None
    x: int | None = None
    y: int | None = None
    element: object | None = None
    trace: object | None = None

    @classmethod
    def miss(cls, trace: object | None = None) -> GroundingResult:
        return cls(grounded=False, trace=trace)


@runtime_checkable
class Grounder(Protocol):
    """Maps an intent (optionally with a prior context for re-resolution) to a
    coordinate. Implementations must be deterministic for a given input or
    document their nondeterminism."""

    def ground(
        self,
        intent: TargetIntent,
        perception: ScreenPerception,
        *,
        prior_context: TargetSelectionContext | None = None,
    ) -> GroundingResult: ...

    def refine(self, x: int, y: int, perception: ScreenPerception) -> tuple[int, int]:
        """Optionally adjust an already-chosen raw coordinate (e.g. a CUA's
        ``click_at(x, y)``) using perception. The baseline returns it unchanged."""
        ...


class DeterministicGrounder:
    """Baseline grounder backed by ``DeterministicTargetSelector``.

    Encapsulates the existing select / reresolve + center-computation logic with
    identical behavior. ``prior_context`` selects the re-resolution path (used
    after a stale/shifted/lost target failure); without it, the initial
    selection path runs.
    """

    def __init__(self, target_selector) -> None:
        self._selector = target_selector

    def ground(
        self,
        intent: TargetIntent,
        perception: ScreenPerception,
        *,
        prior_context: TargetSelectionContext | None = None,
    ) -> GroundingResult:
        if prior_context is not None:
            result = self._selector.reresolve(perception, prior_context)
        else:
            result = self._selector.select(perception, intent)

        selected = result.selected
        if selected is None:
            return GroundingResult.miss(trace=result.trace)

        x, y = element_center(selected)
        return GroundingResult(
            grounded=True,
            element_id=selected.element_id,
            x=x,
            y=y,
            element=selected,
            trace=result.trace,
        )

    def refine(self, x: int, y: int, perception: ScreenPerception) -> tuple[int, int]:
        """Baseline: trust the coordinate as given."""
        return (x, y)


class SnapToInteractableGrounder(DeterministicGrounder):
    """Conservatively snap a raw click coordinate onto the nearest interactable
    element's center, but only within ``snap_radius`` pixels.

    Targets the failure mode where a vision model (e.g. Gemini Computer Use)
    emits a ``click_at(x, y)`` that is a few pixels off the real control — a
    near-miss snaps onto it, while an intentional precise click far from any
    control is left untouched. ``ground`` is inherited unchanged; only the
    raw-coordinate refinement differs. Opt-in via ``OPERON_GROUNDER=snap``.
    """

    def __init__(self, target_selector, *, snap_radius: int = 24) -> None:
        super().__init__(target_selector)
        self._snap_radius = snap_radius

    def refine(self, x: int, y: int, perception: ScreenPerception) -> tuple[int, int]:
        elements = getattr(perception, "visible_elements", None) or []
        best_center: tuple[int, int] | None = None
        best_dist: int | None = None
        for el in elements:
            if not getattr(el, "is_interactable", False):
                continue
            cx, cy = element_center(el)
            dist = abs(cx - x) + abs(cy - y)  # Manhattan; cheap and adequate here
            if dist > self._snap_radius:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best_center = dist, (cx, cy)
        return best_center if best_center is not None else (x, y)


def make_grounder(target_selector) -> Grounder:
    """Build the configured grounder. ``OPERON_GROUNDER`` selects the backend
    (default ``deterministic``); ``snap`` enables :class:`SnapToInteractableGrounder`."""
    name = os.getenv("OPERON_GROUNDER", "deterministic").strip().lower()
    if name == "snap":
        return SnapToInteractableGrounder(target_selector)
    return DeterministicGrounder(target_selector)
