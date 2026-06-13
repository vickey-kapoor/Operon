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
