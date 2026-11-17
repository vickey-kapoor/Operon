"""Find and mark runs a dead process left in flight (OPERON_CHECKPOINTS, KAN-80).

Run state is saved to ``state.json`` on every status change, so after a restart the
store still holds runs that were RUNNING, PENDING or WAITING_FOR_USER with nothing
stepping them. :func:`mark_interrupted_runs` flags them; it never resumes one (the
screen may have changed, and a desktop run would take the mouse). A flagged run keeps
the ``waiting_for_user`` status, so every client already shows it as waiting for the
person, and carries ``stop_reason="interrupted"`` plus ``interrupted_from``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from operon.core.paths import runs_dir
from operon.models.common import RunStatus, StopReason
from operon.models.state import AgentState

logger = logging.getLogger(__name__)

IN_FLIGHT = frozenset({RunStatus.PENDING, RunStatus.RUNNING, RunStatus.WAITING_FOR_USER})


def _load(state_path: Path) -> AgentState | None:
    try:
        return AgentState.model_validate_json(state_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - one unreadable run must not hide the rest
        return None


def is_interrupted(state: AgentState) -> bool:
    return state.status is RunStatus.WAITING_FOR_USER and state.stop_reason is StopReason.INTERRUPTED


def _states(root: Path | None) -> list[tuple[Path, AgentState]]:
    base = Path(root) if root is not None else runs_dir()
    if not base.is_dir():
        return []
    found = []
    for state_path in sorted(base.glob("*/state.json")):
        state = _load(state_path)
        if state is not None:
            found.append((state_path, state))
    return found


def mark_interrupted_runs(root: Path | None = None, *, skip: frozenset[str] = frozenset(), min_age_s: float = 0.0) -> list[str]:
    """Flag every in-flight run on disk as interrupted; returns their run ids.

    ``skip`` holds runs this process is stepping (none at startup). A run already
    flagged is left alone, so scanning twice changes nothing. ``min_age_s`` spares runs
    saved more recently than that (a CLI scan while an API may still be stepping them)."""
    now = time.time()
    marked: list[str] = []
    for state_path, state in _states(root):
        if state.run_id in skip or state.status not in IN_FLIGHT or is_interrupted(state):
            continue
        if min_age_s and now - state_path.stat().st_mtime < min_age_s:
            continue
        state.interrupted_from = state.status
        state.status = RunStatus.WAITING_FOR_USER
        state.stop_reason = StopReason.INTERRUPTED
        state.in_takeover = False  # the person's takeover ended with the process
        state.pause_reason = "Interrupted: Operon restarted while this run was in progress. Resume to continue from the current screen."
        try:
            state_path.write_text(state.model_dump_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not mark run %s interrupted: %s", state.run_id, exc)
            continue
        marked.append(state.run_id)
    if marked:
        logger.info("checkpoints: marked %d interrupted run(s): %s", len(marked), ", ".join(marked))
    return marked


def list_interrupted(root: Path | None = None) -> list[AgentState]:
    """The runs waiting for an explicit resume, newest first."""
    runs = [(path.stat().st_mtime, state) for path, state in _states(root) if is_interrupted(state)]
    return [state for _, state in sorted(runs, key=lambda pair: pair[0], reverse=True)]
