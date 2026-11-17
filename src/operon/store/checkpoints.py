"""Find and mark runs a dead process left in flight (OPERON_CHECKPOINTS, KAN-80).

Run state is saved to ``state.json`` on every status change, so after a restart the
store still holds runs that were RUNNING, PENDING or WAITING_FOR_USER with nothing
stepping them. :func:`mark_interrupted_runs` flags them; it never resumes one (the
screen may have changed, and a desktop run would take the mouse). A flagged run keeps
the ``waiting_for_user`` status, so every client already shows it as waiting for the
person, and carries ``stop_reason="interrupted"`` plus ``interrupted_from``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from operon.core.paths import runs_dir
from operon.core.settings import get_settings
from operon.models.common import RunStatus, StopReason
from operon.models.state import AgentState
from operon.store import owner as run_owner

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


def _dead(state_path: Path, min_age_s: float, now: float) -> bool:
    """Whether nothing is stepping this in-flight run. A run with an owner file is judged by
    its owner; one without (saved before owner files) by how long ago it was last saved."""
    owner = run_owner.read(state_path.parent)
    if owner is not None:
        return not run_owner.owner_alive(owner)
    return not (min_age_s and now - state_path.stat().st_mtime < min_age_s)


def _flag(state: AgentState) -> None:
    state.interrupted_from = state.status
    state.status = RunStatus.WAITING_FOR_USER
    state.stop_reason = StopReason.INTERRUPTED
    state.in_takeover = False  # the person's takeover ended with the process
    state.pause_reason = (
        "Interrupted: Operon restarted while this run was in progress. "
        "Resume to continue from the current screen."
    )


def find_interrupted(
    root: Path | None = None, *, min_age_s: float = 0.0, only: str | None = None
) -> list[AgentState]:
    """Runs already flagged, plus those a scan would flag now, without writing anything."""
    now = time.time()
    found = []
    for state_path, state in _states(root):
        if only is not None and state.run_id != only:
            continue
        if is_interrupted(state):
            found.append(state)
        elif state.status in IN_FLIGHT and _dead(state_path, min_age_s, now):
            _flag(state)  # in memory only
            found.append(state)
    return found


def mark_interrupted_runs(
    root: Path | None = None,
    *,
    skip: frozenset[str] = frozenset(),
    min_age_s: float = 0.0,
    only: str | None = None,
) -> list[str]:
    """Flag every in-flight run on disk whose owner process is gone; returns their run ids.

    A run whose owner (``owner.json``) is alive is never touched, however long it has been
    quiet (a paused or taken-over run writes nothing). A run with no owner file is judged by
    age: saved less than ``min_age_s`` ago counts as live. ``skip`` holds runs this process
    is stepping; ``only`` limits the scan to one run. A second scan changes nothing."""
    marked: list[str] = []
    now = time.time()
    for state_path, state in _states(root):
        if state.run_id in skip or (only is not None and state.run_id != only):
            continue
        if state.status not in IN_FLIGHT or is_interrupted(state) or not _dead(state_path, min_age_s, now):
            continue
        _flag(state)
        try:
            state_path.write_text(state.model_dump_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not mark run %s interrupted: %s", state.run_id, exc)
            continue
        run_owner.release(state_path.parent)  # a dead owner's claim; the resumer writes its own
        marked.append(state.run_id)
    if marked:
        logger.info("checkpoints: marked %d interrupted run(s): %s", len(marked), ", ".join(marked))
    return marked


def list_interrupted(root: Path | None = None) -> list[AgentState]:
    """The runs waiting for an explicit resume, newest first."""
    runs = [(path.stat().st_mtime, state) for path, state in _states(root) if is_interrupted(state)]
    return [state for _, state in sorted(runs, key=lambda pair: pair[0], reverse=True)]


def recorded_provider(run_id: str, root: Path | None = None) -> str | None:
    """The environment provider a run started with, from its ``settings.json`` (None when the
    file or the key is missing: a run saved before settings were recorded)."""
    base = Path(root) if root is not None else runs_dir()
    try:
        value = json.loads((base / run_id / "settings.json").read_text(encoding="utf-8")).get("OPERON_ENVIRONMENT")
    except (OSError, ValueError, AttributeError):
        return None
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def check_resume_environment(state: AgentState, root: Path | None = None, current: str | None = None) -> None:
    """Refuse to resume a desktop run in a different environment provider than it started in.

    ``--env windows-sandbox`` only sets the original process's environment; a resume from a
    shell without it would build the live host executor and move a sandboxed task (and its
    permission mode) outside its isolation. A run with no recorded provider is unknown, and an
    unknown provider is never trusted: refused too, so the person picks one explicitly by
    starting a new run. Browser runs have no provider and are unaffected. Raises ValueError."""
    if state.environment != "desktop":
        return
    current = current or get_settings().environment
    original = recorded_provider(state.run_id, root)
    if original is None:
        raise ValueError(
            f"Run {state.run_id!r} has no recorded environment provider (saved by an older version), so "
            f"Operon cannot tell whether it is safe to resume on {current!r}; start a new run instead"
        )
    if original != current:
        raise ValueError(
            f"Run {state.run_id!r} was started in {original}, but this process uses {current}: resume it "
            f"with OPERON_ENVIRONMENT={original} (operon run --env {original})"
        )
