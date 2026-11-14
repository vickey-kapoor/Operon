"""Operator control state the agent reads each step.

A ``/ws/stream`` client sets these (``api/ws_stream.py`` writes them); the
loop and rule engine only read them. Living in the agent layer keeps that
dependency pointing api -> agent, so agent code never imports the API.

- confidence threshold: pause for approval when a decision's confidence is below
  it (0.0 = off). A session-wide setting.
- disabled rules: rule-engine primitives the operator switched off. Session-wide.
- override hints: a correction typed during a HITL pause, consumed once by the
  next step of the run it was sent for.

The values live on the process runtime (``operon.core.runtime``, KAN-156): the
session settings on ``current_runtime().control``, a run's hint on its ``RunContext``.
"""

from __future__ import annotations

from operon.core.runtime import current_runtime


def get_confidence_threshold() -> float:
    return current_runtime().control.confidence_threshold


def set_confidence_threshold(value: float) -> float:
    """Clamp to [0, 1], store, and return the stored value."""
    control = current_runtime().control
    value = float(value)
    if value != value:  # NaN: clamping would turn it into 1.0 (every action gated)
        return control.confidence_threshold
    control.confidence_threshold = max(0.0, min(1.0, value))
    return control.confidence_threshold


def get_disabled_rules() -> frozenset[str]:
    return current_runtime().control.disabled_rules


def set_disabled_rules(names: list[str]) -> None:
    current_runtime().control.disabled_rules = frozenset(names)


def set_override_hint(run_id: str, hint: str | None) -> None:
    """Store ``hint`` for ``run_id``'s next step; an empty hint clears a pending one."""
    runtime = current_runtime()
    if hint:
        runtime.run(run_id).override_hint = hint
    elif (context := runtime.peek_run(run_id)) is not None:
        context.override_hint = None


def consume_override_hint(run_id: str) -> str | None:
    """Pop and return the pending override hint for ``run_id`` (one-shot)."""
    context = current_runtime().peek_run(run_id)
    if context is None:
        return None
    hint, context.override_hint = context.override_hint, None
    return hint


def reset() -> None:
    """Restore defaults (tests)."""
    runtime = current_runtime()
    runtime.control.confidence_threshold = 0.0
    runtime.control.disabled_rules = frozenset()
    for context in runtime.runs.values():
        context.override_hint = None
