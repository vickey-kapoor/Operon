"""Tests for the grounding seam (element_center + DeterministicGrounder)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.agent.actions.grounding import (
    DeterministicGrounder,
    Grounder,
    GroundingResult,
    SnapToInteractableGrounder,
    element_center,
    make_grounder,
)


def _el(
    eid: str = "e1", x: int = 100, y: int = 200, width: int = 40, height: int = 20,
    *, is_interactable: bool = True,
):
    return SimpleNamespace(
        element_id=eid, x=x, y=y, width=width, height=height, is_interactable=is_interactable,
    )


def _perception(elements):
    return SimpleNamespace(visible_elements=elements)


class _FakeSelector:
    """Records whether select() or reresolve() was called and returns a fixed result."""

    def __init__(self, selected, trace: str = "trace") -> None:
        self._selected = selected
        self._trace = trace
        self.calls: list[str] = []

    def select(self, perception, intent):
        self.calls.append("select")
        return SimpleNamespace(selected=self._selected, trace=self._trace)

    def reresolve(self, perception, context):
        self.calls.append("reresolve")
        return SimpleNamespace(selected=self._selected, trace=self._trace)


# ── element_center ────────────────────────────────────────────────────────

def test_element_center_is_geometric_center() -> None:
    assert element_center(_el(x=100, y=200, width=40, height=20)) == (120, 210)


def test_element_center_floors_offset_to_one_px() -> None:
    # Zero-size box → max(1, 0) keeps the point inside the element, not at its corner.
    assert element_center(_el(x=10, y=20, width=0, height=0)) == (11, 21)


# ── DeterministicGrounder ─────────────────────────────────────────────────

def test_grounder_select_path_returns_center_and_element() -> None:
    el = _el()
    selector = _FakeSelector(el)
    grounder = DeterministicGrounder(selector)

    result = grounder.ground(intent=object(), perception=object())

    assert selector.calls == ["select"]  # no prior_context → initial selection
    assert result.grounded is True
    assert result.element_id == "e1"
    assert (result.x, result.y) == element_center(el)
    assert result.element is el
    assert result.trace == "trace"


def test_grounder_uses_reresolve_when_prior_context_given() -> None:
    selector = _FakeSelector(_el())
    grounder = DeterministicGrounder(selector)

    grounder.ground(intent=object(), perception=object(), prior_context=object())

    assert selector.calls == ["reresolve"]


def test_grounder_miss_returns_empty_result_with_trace() -> None:
    selector = _FakeSelector(None, trace="why-it-missed")
    grounder = DeterministicGrounder(selector)

    result = grounder.ground(intent=object(), perception=object())

    assert result.grounded is False
    assert result.x is None and result.y is None
    assert result.element_id is None and result.element is None
    assert result.trace == "why-it-missed"


def test_grounding_result_miss_factory() -> None:
    miss = GroundingResult.miss(trace="t")
    assert miss.grounded is False
    assert miss.trace == "t"
    assert miss.x is None and miss.element_id is None


def test_deterministic_grounder_satisfies_protocol() -> None:
    assert isinstance(DeterministicGrounder(_FakeSelector(None)), Grounder)


def test_deterministic_refine_is_identity() -> None:
    grounder = DeterministicGrounder(_FakeSelector(None))
    assert grounder.refine(137, 246, _perception([_el()])) == (137, 246)


# ── SnapToInteractableGrounder ────────────────────────────────────────────

def _snap(snap_radius: int = 24) -> SnapToInteractableGrounder:
    return SnapToInteractableGrounder(_FakeSelector(None), snap_radius=snap_radius)


def test_snap_corrects_a_near_miss() -> None:
    # Element centered at (120, 210); a click 4px away (Manhattan) snaps onto it.
    perception = _perception([_el(x=100, y=200, width=40, height=20)])
    assert _snap().refine(118, 208, perception) == (120, 210)


def test_snap_leaves_far_clicks_untouched() -> None:
    perception = _perception([_el(x=100, y=200, width=40, height=20)])
    # Far from any interactable center → intentional click is respected.
    assert _snap().refine(500, 500, perception) == (500, 500)


def test_snap_ignores_non_interactable_elements() -> None:
    perception = _perception([_el(x=100, y=200, width=40, height=20, is_interactable=False)])
    # A near-miss on a decorative element is NOT snapped.
    assert _snap().refine(118, 208, perception) == (118, 208)


def test_snap_picks_the_nearest_interactable() -> None:
    perception = _perception([
        _el("near", x=100, y=200, width=40, height=20),   # center (120, 210)
        _el("far", x=280, y=300, width=40, height=20),    # center (300, 310)
    ])
    assert _snap().refine(122, 212, perception) == (120, 210)


def test_snap_inherits_deterministic_ground() -> None:
    el = _el()
    grounder = SnapToInteractableGrounder(_FakeSelector(el))
    result = grounder.ground(intent=object(), perception=_perception([el]))
    assert result.grounded is True
    assert (result.x, result.y) == element_center(el)


# ── make_grounder factory ─────────────────────────────────────────────────

def test_make_grounder_defaults_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_GROUNDER", raising=False)
    assert type(make_grounder(_FakeSelector(None))) is DeterministicGrounder


def test_make_grounder_selects_snap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_GROUNDER", "snap")
    assert type(make_grounder(_FakeSelector(None))) is SnapToInteractableGrounder


def test_make_grounder_unknown_falls_back_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_GROUNDER", "nonsense")
    assert type(make_grounder(_FakeSelector(None))) is DeterministicGrounder
