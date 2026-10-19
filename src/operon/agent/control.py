"""Operator control state the agent reads each step.

The UI sets these over ``/ws/stream`` (``api/ws_stream.py`` writes them); the
loop and rule engine only read them. Living in the agent layer keeps that
dependency pointing api -> agent, so agent code never imports the API.

- confidence threshold: pause for approval when a decision's confidence is below
  it (0.0 = off). A session-wide setting.
- disabled rules: rule-engine primitives the operator switched off. Session-wide.
- override hints: a correction typed during a HITL pause, consumed once by the
  next step of the run it was sent for.
"""

from __future__ import annotations

_confidence_threshold: float = 0.0
_disabled_rules: frozenset[str] = frozenset()
_override_hints: dict[str, str] = {}


def get_confidence_threshold() -> float:
    return _confidence_threshold


def set_confidence_threshold(value: float) -> float:
    """Clamp to [0, 1], store, and return the stored value."""
    global _confidence_threshold
    _confidence_threshold = max(0.0, min(1.0, float(value)))
    return _confidence_threshold


def get_disabled_rules() -> frozenset[str]:
    return _disabled_rules


def set_disabled_rules(names: list[str]) -> None:
    global _disabled_rules
    _disabled_rules = frozenset(names)


def set_override_hint(run_id: str, hint: str | None) -> None:
    """Store ``hint`` for ``run_id``'s next step; an empty hint clears a pending one."""
    if hint:
        _override_hints[run_id] = hint
    else:
        _override_hints.pop(run_id, None)


def consume_override_hint(run_id: str) -> str | None:
    """Pop and return the pending override hint for ``run_id`` (one-shot)."""
    return _override_hints.pop(run_id, None)


def reset() -> None:
    """Restore defaults (tests)."""
    global _confidence_threshold, _disabled_rules
    _confidence_threshold = 0.0
    _disabled_rules = frozenset()
    _override_hints.clear()
