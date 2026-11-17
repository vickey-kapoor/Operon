"""Step-efficiency score for one run (KAN-85). Measurement only: it changes no behaviour.

It is a pure function of what a run already records on disk: the step records in
``run.jsonl`` (``StepLog`` rows, plus ``pre_step_failure`` rows) and each step's
``step_timing.json``. Anything missing gives ``None`` for the numbers that need it,
never an error, so it also works on runs saved before a field existed.

Fields
------
total_steps          step records in the run log (completed steps and pre-step failures).
model_calls          steps decided by the model: ``decision_source`` is set and is not a
                     ``[RULE]`` decision. None if no step has a ``decision_source``.
actions_per_model_call
                     actions executed (a batch counts as its children) / model_calls:
                     above 1.0 means batching saved model calls.
wall_seconds         sum of the per-step ``total_ms`` in ``step_timing.json``; None if absent.
seconds_per_step     wall_seconds / total_steps.
wasted_steps         steps counted once if ANY of these signals holds:
  blocked            the action was suppressed as redundant: ``executed_action.failure_category``
                     is one of the four redundancy categories (repeated action, repeated target,
                     subgoal already completed, no meaningful progress across steps).
  failed             ``executed_action.success`` is false (this includes blocked steps), or the
                     step is a ``pre_step_failure`` record.
  repeat             the executed action equals the previous step's executed action and the
                     step's verification status is not ``success`` / ``progressing_stable``
                     (an identical repeat with no screen change confirmed).
  noop_wait          a ``wait`` action whose verification status is not ``success`` /
                     ``progressing_stable`` (waited, nothing changed).
wasted_ratio         wasted_steps / total_steps (None for an empty run).

``blocked``, ``failed``, ``repeat`` and ``noop_wait`` are also reported as counts; they
can overlap, so they may sum to more than wasted_steps.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REDUNDANT_CATEGORIES = frozenset({
    "repeated_action_without_progress",
    "repeated_target_without_progress",
    "subgoal_already_completed",
    "no_meaningful_progress_across_steps",
})
_PROGRESS_STATUSES = frozenset({"success", "progressing_stable"})


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else None


def _leaf_count(action: Any) -> int:
    children = _get(action, "actions")
    if _get(action, "action_type") == "batch" and isinstance(children, list) and children:
        return len(children)
    return 1


def compute_efficiency(
    entries: Iterable[Mapping[str, Any]],
    step_ms: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    """Score a run from its run-log rows (as dicts) and optional per-step durations in ms."""
    steps = [e for e in entries if isinstance(e, Mapping)]
    total = len(steps)
    blocked = failed = repeat = noop = wasted = 0
    model_calls: int | None = None
    executed = 0
    previous_action: Any = None
    for entry in steps:
        executed_action = _get(entry, "executed_action")
        action = _get(executed_action, "action")
        is_failed = entry.get("record_type") == "pre_step_failure" or _get(executed_action, "success") is False
        status = _get(_get(entry, "verification_result"), "status")
        progressed = status in _PROGRESS_STATUSES

        is_blocked = _get(executed_action, "failure_category") in REDUNDANT_CATEGORIES
        is_repeat = action is not None and action == previous_action and not progressed
        is_noop = _get(action, "action_type") == "wait" and not progressed
        blocked += is_blocked
        failed += is_failed
        repeat += is_repeat
        noop += is_noop
        wasted += is_blocked or is_failed or is_repeat or is_noop
        previous_action = action

        source = entry.get("decision_source")
        if isinstance(source, str):
            model_calls = (model_calls or 0) + (0 if source.startswith("[RULE]") else 1)
        if action is not None:
            executed += _leaf_count(action)

    timings = [ms for ms in (step_ms or []) if isinstance(ms, (int, float))]
    wall = round(sum(timings) / 1000, 3) if timings else None
    return {
        "total_steps": total,
        "model_calls": model_calls,
        "wall_seconds": wall,
        "seconds_per_step": round(wall / total, 3) if wall is not None and total else None,
        "wasted_steps": wasted,
        "wasted_ratio": round(wasted / total, 3) if total else None,
        "wasted_blocked": blocked,
        "wasted_failed": failed,
        "wasted_repeat": repeat,
        "wasted_noop_wait": noop,
        "actions_per_model_call": round(executed / model_calls, 3) if model_calls else None,
    }


def efficiency_for_response(run: Any, run_store: Any) -> dict[str, Any] | None:
    """The score for a GET run response: the one saved at the end of the run, else (a run
    saved before KAN-85, or a restart) computed from its files; None while it is running."""
    saved = getattr(run, "efficiency", None)
    if saved is not None:
        return saved if isinstance(saved, dict) else None
    status = getattr(getattr(run, "status", None), "value", None)
    root = getattr(run_store, "root_dir", None)
    if status not in {"succeeded", "failed", "cancelled"} or root is None:
        return None
    try:
        return run_efficiency(Path(root) / run.run_id)
    except Exception:
        return None


def run_efficiency(run_dir: str | Path) -> dict[str, Any]:
    """Score the run saved in ``run_dir``; unreadable or missing files count as absent."""
    root = Path(run_dir)
    entries: list[dict[str, Any]] = []
    try:
        lines = (root / "run.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    step_ms: list[float | None] = []
    for path in sorted(root.glob("step_*/step_timing.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("total_ms")
        except (OSError, ValueError, AttributeError):
            continue
        step_ms.append(value if isinstance(value, (int, float)) else None)
    return compute_efficiency(entries, step_ms)
