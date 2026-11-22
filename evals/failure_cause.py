"""KAN-175: shared failure-cause helpers for the benchmarks (labelling and grading input only).

Dependency-free on purpose: the OSWorld runner's venv (osw-venv) has no Operon install, so this
module must not import ``operon``. ``state_cause_value`` mirrors
``operon.agent.failure_cause.cause_from_state``; tests/evals/test_failure_cause.py checks the two
agree for every run status and stop reason."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

# Values of operon.models.common.RunFailureCause / StopReason / RunStatus, as plain strings.
_CAUSE_BY_STOP_REASON = {
    "max_step_limit_reached": "max_steps",
    "completion_unverified": "model_gave_up",
    "model_blocked": "model_blocked",
    "pre_step_perception_failed": "perception_failed",
    "perception_low_quality": "perception_failed",
    "interrupted": "interrupted",
}


def _value(item: object) -> str | None:
    value = getattr(item, "value", item)
    return str(value) if value else None


def state_cause_value(status: object, stop_reason: object, failure_cause: object = None) -> str | None:
    """The cause for a run's final state (stored cause wins), or None when the run did not fail."""
    stored = _value(failure_cause)
    if stored:
        return stored
    status_value, reason = _value(status), _value(stop_reason)
    if status_value == "cancelled":
        return "cancelled"
    if reason == "interrupted":
        return "interrupted"
    if status_value == "waiting_for_user":
        return "needs_user"
    if status_value != "failed":
        return None
    return _CAUSE_BY_STOP_REASON.get(reason or "", "other")

_STEP_DIR = re.compile(r"^step_(\d+)$")


def bench_cause(
    *,
    passed: bool,
    bench_error: bool,
    run_status: object,
    stop_reason: object,
    state_cause: object = None,
    timed_out: bool = False,
) -> str | None:
    """The one cause for a bench result, or None when the task passed."""
    if passed:
        return None
    if bench_error:
        return "bench_error"
    if timed_out:
        return "timeout"
    if _value(stop_reason) == "interrupted":
        return "interrupted"
    cause = state_cause_value(run_status, stop_reason, state_cause)
    if cause:
        return cause
    if _value(run_status) == "succeeded":
        return "wrong_result"
    return "other"


def by_cause(causes: Iterable[str | None]) -> dict[str, int]:
    """Count causes, most common first (ties by name); None is skipped."""
    counts: dict[str, int] = {}
    for cause in causes:
        if cause:
            counts[cause] = counts.get(cause, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def final_screenshot(run_dir: Path) -> Path | None:
    """The run's last screenshot by step index: ``after.png``, else ``before.png``, of the highest step that has one."""
    steps: list[tuple[int, Path]] = []
    try:
        for entry in run_dir.iterdir():
            match = _STEP_DIR.match(entry.name)
            if match and entry.is_dir():
                steps.append((int(match.group(1)), entry))
    except OSError:
        return None
    for _, directory in sorted(steps, reverse=True):
        for name in ("after.png", "before.png"):
            if (directory / name).is_file():
                return directory / name
    return None
