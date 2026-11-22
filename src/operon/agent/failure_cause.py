"""KAN-175: one failure cause per failed run, from a fixed list a bench can group by.

Labelling only: nothing here changes, retries or blocks a model action. The loop and the
API call :func:`note_failure` where a run fails; :func:`cause_from_state` derives the cause
for runs saved before ``AgentState.failure_cause`` existed, so old state.json files and
reports group the same way. Benchmarks import this module (it imports no loop code).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx

from operon.models.common import (
    FailureCategory,
    LoopStage,
    RunFailureCause,
    RunStatus,
    StopReason,
)
from operon.models.logs import FailureRecord, RunEndLog
from operon.models.state import AgentState
from operon.store.run_logger import append_step_log_critical

# Longest failure text kept in state.json (a provider error body can be long).
DETAIL_LIMIT = 500

_CAUSE_BY_STOP_REASON: dict[str, RunFailureCause] = {
    StopReason.MAX_STEP_LIMIT_REACHED.value: RunFailureCause.MAX_STEPS,
    # Only the thin harness's "gave up" verdict ends a run with this reason.
    StopReason.COMPLETION_UNVERIFIED.value: RunFailureCause.MODEL_GAVE_UP,
    StopReason.MODEL_BLOCKED.value: RunFailureCause.MODEL_BLOCKED,
    StopReason.PRE_STEP_PERCEPTION_FAILED.value: RunFailureCause.PERCEPTION_FAILED,
    StopReason.PERCEPTION_LOW_QUALITY.value: RunFailureCause.PERCEPTION_FAILED,
    StopReason.INTERRUPTED.value: RunFailureCause.INTERRUPTED,
}

# Names of the model-client exception types (matched by class name along the MRO, so this
# module needs no import of the clients or backends).
_API_ERROR_TYPES = frozenset({"ModelTransportError", "AnthropicClientError", "GeminiClientError", "GeminiComputerUseError"})
_BLOCKED_TYPES = frozenset({"ModelBlockedError", "GeminiPromptBlockedError"})
_PERCEPTION_TYPES = frozenset({"PerceptionError", "BackendCompatibilityError"})


def _value(item: object) -> str | None:
    if item is None:
        return None
    return str(getattr(item, "value", item))


def cause_from_state(
    status: object,
    stop_reason: object,
    failure_cause: object = None,
) -> str | None:
    """The cause value for a run's final state, or None when the run did not fail.

    Takes enum members or their string values (as read from state.json). A stored cause
    wins; otherwise the cause is derived from the stop reason and status.
    """
    stored = _value(failure_cause)
    if stored:
        return stored
    status_value = _value(status)
    reason = _value(stop_reason)
    if status_value == RunStatus.CANCELLED.value:
        return RunFailureCause.CANCELLED.value
    if reason == StopReason.INTERRUPTED.value:
        return RunFailureCause.INTERRUPTED.value
    if status_value == RunStatus.WAITING_FOR_USER.value:
        return RunFailureCause.NEEDS_USER.value
    if status_value != RunStatus.FAILED.value:
        return None
    if reason in _CAUSE_BY_STOP_REASON:
        return _CAUSE_BY_STOP_REASON[reason].value
    return RunFailureCause.OTHER.value


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is(exc: BaseException, names: frozenset[str]) -> bool:
    return any(klass.__name__ in names for klass in type(exc).__mro__)


def cause_of_error(exc: BaseException) -> RunFailureCause:
    """Classify an exception that ended a run (it and its ``__cause__``/``__context__`` chain)."""
    chain = list(_chain(exc))
    if any(_is(item, _BLOCKED_TYPES) for item in chain):
        return RunFailureCause.MODEL_BLOCKED
    if any(isinstance(item, httpx.HTTPError) or _is(item, _API_ERROR_TYPES) for item in chain):
        return RunFailureCause.API_ERROR
    if any(_is(item, _PERCEPTION_TYPES) for item in chain):
        return RunFailureCause.PERCEPTION_FAILED
    return RunFailureCause.RUN_CRASHED


def error_text(exc: BaseException) -> str:
    """Short text for an exception: its message, or its type name when it has none."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def note_failure(
    state: AgentState,
    *,
    cause: RunFailureCause | None = None,
    detail: str | None = None,
) -> None:
    """Record the run's failure cause and text once; an existing cause is never overwritten."""
    if state.failure_cause is not None:
        return
    if cause is None:
        derived = cause_from_state(RunStatus.FAILED, state.stop_reason)
        cause = RunFailureCause(derived) if derived else RunFailureCause.OTHER
    state.failure_cause = cause
    if detail and detail.strip():
        state.failure_detail = detail.strip()[:DETAIL_LIMIT]


def clear_failure(state: AgentState) -> None:
    """Forget a recorded cause (a resumed run is no longer failed)."""
    state.failure_cause = None
    state.failure_detail = None


def append_run_end(
    log_path: str | Path,
    state: AgentState,
    *,
    status: RunStatus,
    category: FailureCategory | None,
    message: str,
) -> None:
    """Write the terminal ``run_end`` row to run.jsonl for a run that ended outside a step."""
    failure = None
    if category is not None:
        failure = FailureRecord(
            category=category,
            stage=LoopStage.RECOVER,
            retry_count=0,
            terminal=True,
            recoverable=False,
            reason=message,
            stop_reason=state.stop_reason,
        )
    append_step_log_critical(
        log_path,
        RunEndLog(
            run_id=state.run_id,
            step_index=state.step_count,
            status=status,
            stop_reason=state.stop_reason,
            failure_cause=state.failure_cause,
            failure=failure,
            message=message,
        ),
    )
