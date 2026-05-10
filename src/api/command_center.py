"""SSE streaming endpoint for the Command Center live step view.

GET /command-center/api/run/{run_id}/stream
  - Replays all existing StepLog entries from run.jsonl on connect.
  - Then polls the file every 200 ms for new entries and forwards them.
  - Emits a heartbeat comment every 15 s so the connection stays alive.
  - Emits a final `run_end` data event when the run reaches a terminal state
    and closes the stream automatically so the client does not have to poll
    indefinitely against a finished run.
"""

from __future__ import annotations

import asyncio
import json
import re as _re
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src.models.logs import PreStepFailureLog, StepLog
from src.models.policy import ActionType
from src.models.verification import VerificationStatus

router = APIRouter()

_RUN_ID_RE = _re.compile(r"^[A-Za-z0-9_\-]+$")
_MAX_RUN_ID_LENGTH = 64
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_POLL_INTERVAL = 0.2       # seconds
_HEARTBEAT_INTERVAL = 15.0 # seconds
_EMPTY_POLLS_BEFORE_COMPLETION_CHECK = 10


def _runs_root() -> Path:
    import os
    return Path(os.getenv("OPERON_RUNS_ROOT", "runs")).resolve()


def _validate(run_id: str) -> bool:
    return (
        bool(run_id)
        and len(run_id) <= _MAX_RUN_ID_LENGTH
        and bool(_RUN_ID_RE.match(run_id))
    )


# ── Step event shape ──────────────────────────────────────────────────────────

def _step_log_to_event(entry: StepLog) -> dict:
    """Convert a StepLog to the command-center step event payload.

    Shape:
        {type, n, action, target, value, source, conf, expects, status}
    """
    action = entry.executed_action.action
    atype  = action.action_type.value

    # ── target: most human-readable description of what is being acted on
    target: str = ""
    if action.action_type is ActionType.NAVIGATE:
        url = action.url or ""
        target = url.replace("https://", "").replace("http://", "").split("?")[0][:70]
    elif action.action_type in (ActionType.TYPE, ActionType.SELECT):
        target = action.selector or action.target_element_id or ""
    elif action.action_type in (ActionType.PRESS_KEY, ActionType.HOTKEY):
        target = action.key or ""
    elif action.action_type is ActionType.LAUNCH_APP:
        target = action.text or ""
    elif action.action_type is ActionType.SCROLL:
        amt = action.scroll_amount or 0
        target = "down" if amt > 0 else "up"
    elif action.action_type is ActionType.WAIT:
        target = f"{action.wait_ms}ms"
    elif action.action_type is ActionType.UPLOAD_FILE:
        target = action.text or ""
    else:
        target = action.selector or action.target_element_id or ""

    # Fall back to a truncated rationale when no structural target is available
    if not target:
        rat = entry.policy_decision.rationale
        target = rat[:50] + "…" if len(rat) > 50 else rat

    # ── value: typed text for type/select actions only
    value: str | None = None
    if action.action_type in (ActionType.TYPE, ActionType.SELECT):
        if action.text and len(action.text) <= 80:
            value = action.text

    # ── source: rule name or "LLM"
    source = entry.policy_decision.rule_name or "LLM"

    # ── expects: expected visual change, omitted when "none"
    ec = entry.policy_decision.expected_change.value
    expects: str | None = ec if ec != "none" else None

    # ── status: derive from failure record and verification outcome
    if entry.failure is not None:
        status = "failed"
    elif entry.verification_result.status in (
        VerificationStatus.SUCCESS,
        VerificationStatus.PROGRESSING_STABLE,
        VerificationStatus.UNCERTAIN,
        VerificationStatus.STABLE_WAIT,
    ):
        status = "done"
    else:
        status = "done"  # any other completed state is treated as done

    return {
        "type":    "step",
        "n":       entry.step_index,
        "action":  atype,
        "target":  target,
        "value":   value,
        "source":  source,
        "conf":    round(entry.policy_decision.confidence, 4),
        "expects": expects,
        "status":  status,
    }


def _parse_jsonl_line(raw: str) -> dict | None:
    """Parse one JSONL line into a step event dict, or None if not a StepLog."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return _step_log_to_event(StepLog.model_validate_json(raw))
    except (ValidationError, Exception):
        pass
    # PreStepFailureLog lines are silently skipped — they have no execution data
    # to map to the step event shape.
    return None


# ── File-tail async generator ─────────────────────────────────────────────────

async def _stream_run_log(log_path: Path, state_path: Path) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted strings for a run.jsonl file.

    Phase 1 — Replay all existing complete lines.
    Phase 2 — Poll every 200 ms for new lines; emit heartbeat every 15 s.
    Closes automatically when the run reaches a terminal state and no new
    lines have arrived for at least EMPTY_POLLS_BEFORE_COMPLETION_CHECK polls.
    """
    last_heartbeat = time.monotonic()
    buf: str = ""
    pos: int = 0

    # ── Phase 1: replay existing lines ───────────────────────────────────────
    if log_path.exists():
        try:
            with log_path.open("rb") as fh:
                for raw_line in fh:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    event = _parse_jsonl_line(line)
                    if event is not None:
                        yield f"data: {json.dumps(event)}\n\n"
                pos = fh.tell()
        except OSError:
            pass

    # After replay, if the run is already terminal, close immediately.
    terminal_status = _read_terminal_status(state_path)
    if terminal_status:
        yield f'data: {json.dumps({"type": "run_end", "status": terminal_status})}\n\n'
        return

    # ── Phase 2: poll for new lines ───────────────────────────────────────────
    empty_polls = 0
    while True:
        await asyncio.sleep(_POLL_INTERVAL)

        # Heartbeat
        now = time.monotonic()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield ": heartbeat\n\n"
            last_heartbeat = now

        # Read new bytes
        new_events: list[dict] = []
        if log_path.exists():
            try:
                with log_path.open("rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    buf += chunk.decode("utf-8", errors="replace")
                    *complete_lines, buf = buf.split("\n")
                    for line in complete_lines:
                        event = _parse_jsonl_line(line)
                        if event is not None:
                            new_events.append(event)
            except OSError:
                pass

        if new_events:
            empty_polls = 0
            for event in new_events:
                yield f"data: {json.dumps(event)}\n\n"
        else:
            empty_polls += 1

        # Check for run completion after enough empty polls (≥ 2 s of quiet)
        if empty_polls >= _EMPTY_POLLS_BEFORE_COMPLETION_CHECK:
            terminal_status = _read_terminal_status(state_path)
            if terminal_status:
                yield f'data: {json.dumps({"type": "run_end", "status": terminal_status})}\n\n'
                return
            empty_polls = 0  # reset; run still live, keep watching


def _read_terminal_status(state_path: Path) -> str | None:
    """Return the run status string if it is terminal, else None."""
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = state.get("status", "")
        return status if status in _TERMINAL_STATUSES else None
    except Exception:
        return None


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/command-center/api/run/{run_id}/stream")
async def stream_run_steps(run_id: str) -> StreamingResponse:
    """SSE stream of StepLog entries for a run.

    Replays existing steps immediately on connect, then tails run.jsonl at
    200 ms intervals.  Closes automatically when the run completes.
    """
    if not _validate(run_id):
        async def _reject() -> AsyncGenerator[str, None]:
            yield f'data: {json.dumps({"type": "error", "detail": "invalid run_id"})}\n\n'
        return StreamingResponse(
            _reject(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            status_code=400,
        )

    run_dir    = _runs_root() / run_id
    log_path   = run_dir / "run.jsonl"
    state_path = run_dir / "state.json"

    return StreamingResponse(
        _stream_run_log(log_path, state_path),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
