"""KAN-175: shared failure-cause helpers for the benchmarks (labelling and grading input only)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from operon.agent.failure_cause import cause_from_state
from operon.models.common import RunFailureCause

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
        return RunFailureCause.BENCH_ERROR.value
    if timed_out:
        return RunFailureCause.TIMEOUT.value
    if str(getattr(stop_reason, "value", stop_reason)) == "interrupted":
        return RunFailureCause.INTERRUPTED.value
    cause = cause_from_state(run_status, stop_reason, state_cause)
    if cause:
        return cause
    if str(getattr(run_status, "value", run_status)) == "succeeded":
        return RunFailureCause.WRONG_RESULT.value
    return RunFailureCause.OTHER.value


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
