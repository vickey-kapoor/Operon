"""Front door (KAN-89): start a task, follow it, approve, take over, and report.

One small service used by the local task page (``/app``), its HTTP routes
(``/app/api/...``) and ``operon run`` (in process, or against a running API with
``--request``), so the three always behave the same way.

A run lives in the browser loop or the desktop loop. The service finds the loop
from the run itself, so callers only pass a run id.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from operon.agent import control as run_control
from operon.agent.policy.trust_gate import resolve_permission_mode
from operon.api import ws_stream as _ws_stream
from operon.api.runtime import loops as _loops
from operon.api.runtime import tasks as _tasks
from operon.models.common import RunStatus, RunTaskRequest, StrictModel
from operon.models.logs import StepLog
from operon.store.replay import load_run_replay

logger = logging.getLogger(__name__)

TERMINAL = frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED})
ENVIRONMENTS = ("browser", "desktop")

# Wall-clock bounds of runs started here, for the report's duration.
_started_at: dict[str, float] = {}
_finished_at: dict[str, float] = {}


class FrontDoorTask(StrictModel):
    """A task as the page and ``operon run`` send it."""

    intent: str = Field(min_length=1, max_length=500)
    environment: str = Field(default="browser", pattern=r"^(browser|desktop)$")
    start_url: str | None = Field(default=None, min_length=1)
    permission_mode: str | None = Field(default=None, pattern=r"^(ask|risky|auto)$")
    max_steps: int = Field(default=25, ge=1, le=200)
    headless: bool | None = None


class FrontDoorError(Exception):
    """A request the run's state does not allow. ``status`` is the HTTP status to answer."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _loop_for_environment(environment: str):
    if environment == "desktop":
        return _loops.get_desktop_agent_loop()
    return _loops.get_agent_loop()


def _environment_of(run_id: str) -> str:
    return "desktop" if _loops.desktop_loop_holds_run(run_id) else "browser"


def _loop_of(run_id: str):
    return _loop_for_environment(_environment_of(run_id))


async def _require_run(run_id: str):
    loop = _loop_of(run_id)
    record = await loop.run_store.get_run(run_id)
    if record is None:
        raise FrontDoorError("Run not found", 404)
    return loop, record


async def start_task(task: FrontDoorTask, *, auto_run: bool = True, live_view: bool = True) -> dict[str, Any]:
    """Create the run and, unless ``auto_run`` is False (offline tests), start stepping it.

    A browser run with ``live_view`` runs in observable mode: one headless Chromium on
    CDP port 9222 whose screencast reaches the page over ``/ws/stream``. Without it
    (``operon run`` in process, where no page watches) it is an ordinary batch run."""
    loop = _loop_for_environment(task.environment)
    request = RunTaskRequest(
        intent=task.intent,
        start_url=task.start_url,
        headless=task.headless,
        max_steps=task.max_steps,
        permission_mode=task.permission_mode,
        mode="observable" if live_view and task.environment == "browser" else "batch",
    )
    response = await loop.start_run(request)
    _started_at[response.run_id] = time.time()
    _ws_stream.set_active_run(response.run_id)
    _ws_stream.publish_event({
        "type": "run_started",
        "run_id": response.run_id,
        "intent": task.intent,
        "status": response.status.value,
        "environment": task.environment,
    })
    if auto_run:
        _tasks.schedule_auto_run(loop, response.run_id, task.max_steps)
    return await run_status(response.run_id)


def _action_summary(action) -> str:
    parts = [action.action_type.value]
    ctx = action.target_context
    name = getattr(getattr(ctx, "original_target", None), "primary_name", None) if ctx is not None else None
    if name:
        parts.append(f"'{name}'")
    if action.url:
        parts.append(action.url)
    if action.text:
        parts.append(f"\"{action.text[:80]}\"")
    if action.key:
        parts.append(action.key)
    return " ".join(parts)


def _screenshot(entry: StepLog) -> str:
    """The step's after-image, or its before-image when no action ran (a gate pause
    logs an after path that is never written)."""
    after = entry.after_artifact_path
    return after if Path(after).exists() else entry.before_artifact_path


def _steps(loop, run_id: str) -> list[dict[str, Any]]:
    try:
        entries = load_run_replay(run_id, root_dir=loop.run_store.root_dir)
    except (FileNotFoundError, OSError):
        return []
    steps = []
    for entry in entries:
        if not isinstance(entry, StepLog):
            continue
        steps.append({
            "step": entry.step_index,
            "action": _action_summary(entry.executed_action.action),
            "action_type": entry.executed_action.action.action_type.value,
            "ok": entry.executed_action.success,
            "detail": entry.executed_action.detail,
            "verification": entry.verification_result.status.value,
            "why": entry.policy_decision.rationale,
            # The full typed text: a STOP carries the task's answer here.
            "text": entry.executed_action.action.text,
            "screenshot": _screenshot(entry),
        })
    return steps


def _waiting(loop, record) -> dict[str, Any] | None:
    """What the run waits on from the person, or None when it does not wait."""
    if record.status is not RunStatus.WAITING_FOR_USER:
        return None
    if getattr(record, "in_takeover", False):
        return {"kind": "takeover", "reason": "You are in control. Hand back when you are done."}
    reason = getattr(record, "pause_reason", None) or record.hitl_message or "Paused."
    gate = loop.pending_gate(record.run_id) if hasattr(loop, "pending_gate") else None
    return {"kind": "approval" if gate is not None else "needs_you", "gate": gate, "reason": reason}


async def run_status(run_id: str, *, recent: int = 8) -> dict[str, Any]:
    loop, record = await _require_run(run_id)
    steps = _steps(loop, run_id)
    if record.status in TERMINAL:
        _finished_at.setdefault(run_id, time.time())
    latest = steps[-1]["screenshot"] if steps else None
    environment = _environment_of(run_id)
    return {
        "run_id": record.run_id,
        "intent": record.intent,
        "environment": environment,
        # A desktop the page streams and drives during a takeover (the Linux cloud
        # computer); a local Windows desktop shows step screenshots instead.
        "live_view": environment == "browser" or hasattr(getattr(loop, "executor", None), "live_frame_png"),
        "status": record.status.value,
        "done": record.status in TERMINAL,
        "step_count": record.step_count,
        "max_steps": record.max_steps,
        "permission_mode": resolve_permission_mode(getattr(record, "permission_mode", None)).value,
        "in_takeover": bool(getattr(record, "in_takeover", False)),
        "waiting": _waiting(loop, record),
        "latest_screenshot": latest,
        "recent_steps": steps[-recent:],
    }


async def approve(run_id: str) -> dict[str, Any]:
    """Approve what the run paused on (a gate's action, or 'continue' after the model
    asked for help) and keep it going."""
    loop, record = await _require_run(run_id)
    if record.status is not RunStatus.WAITING_FOR_USER:
        raise FrontDoorError(f"Run is {record.status.value}; nothing to approve")
    if getattr(record, "in_takeover", False):
        raise FrontDoorError("You are in control; hand back instead")
    await _tasks.resume_and_schedule(loop, run_id, approve_gated_action=True)
    _ws_stream.publish_event({"type": "control_ack", "action": "approve", "run_id": run_id})
    return await run_status(run_id)


async def decline(run_id: str, hint: str | None = None) -> dict[str, Any]:
    """Refuse the paused action; the model gets the person's correction instead."""
    loop, record = await _require_run(run_id)
    if record.status is not RunStatus.WAITING_FOR_USER:
        raise FrontDoorError(f"Run is {record.status.value}; nothing to decline")
    if getattr(record, "in_takeover", False):
        raise FrontDoorError("You are in control; hand back instead")
    text = "The user declined that action; do not do it."
    if hint and hint.strip():
        text += f" Instead: {hint.strip()}"
    run_control.set_override_hint(run_id, text)
    await _tasks.resume_and_schedule(loop, run_id, approve_gated_action=False)
    _ws_stream.publish_event({"type": "control_ack", "action": "decline", "run_id": run_id})
    return await run_status(run_id)


async def take_over(run_id: str) -> dict[str, Any]:
    loop, record = await _require_run(run_id)
    if record.status in TERMINAL:
        raise FrontDoorError(f"Run is {record.status.value}; nothing to take over")
    try:
        await loop.take_over(run_id)
    except ValueError as exc:
        raise FrontDoorError(str(exc)) from exc
    _ws_stream.publish_event({"type": "control_ack", "action": "takeover", "run_id": run_id})
    return await run_status(run_id)


async def hand_back(run_id: str, note: str | None = None) -> dict[str, Any]:
    loop, record = await _require_run(run_id)
    if not getattr(record, "in_takeover", False):
        raise FrontDoorError("Run is not under your control")
    await loop.hand_back(run_id, note)
    _tasks.schedule_auto_run(loop, run_id, record.max_steps)
    _ws_stream.publish_event({"type": "control_ack", "action": "handback", "run_id": run_id})
    return await run_status(run_id)


async def stop(run_id: str) -> dict[str, Any]:
    loop, record = await _require_run(run_id)
    if record.status not in TERMINAL:
        await loop.run_store.set_status(run_id, RunStatus.CANCELLED)
        await _tasks.cancel_auto_run(run_id)
        await _tasks.cleanup_cancelled_run(loop, run_id)
        _ws_stream.publish_event({"type": "run_completed", "run_id": run_id, "status": "cancelled"})
    return await run_status(run_id)


async def user_input(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The person's own click or keys during a takeover, sent from the page's live view."""
    loop, record = await _require_run(run_id)
    if not getattr(record, "in_takeover", False):
        raise FrontDoorError("Take over first: the agent is in control")
    send = getattr(loop.executor, "user_input", None)
    if send is None:
        raise FrontDoorError("This environment takes input on the screen itself, not from the page")
    try:
        delivered = await send(
            run_id,
            input_type=str(payload.get("input_type", "click")),
            x_ratio=float(payload.get("x_ratio", 0.0)),
            y_ratio=float(payload.get("y_ratio", 0.0)),
            text=str(payload.get("text", ""))[:2000],
            key=str(payload.get("key", ""))[:40],
            delta_y=int(payload.get("delta_y", 300)),
        )
    except ValueError as exc:
        raise FrontDoorError(str(exc), 400) from exc
    if not delivered:
        raise FrontDoorError("The run has no open browser")
    return {"ok": True}


_OUTCOME = {
    RunStatus.SUCCEEDED: "Done",
    RunStatus.FAILED: "Failed",
    RunStatus.CANCELLED: "Stopped",
    RunStatus.WAITING_FOR_USER: "Waiting for you",
    RunStatus.RUNNING: "Still running",
    RunStatus.PENDING: "Not started",
}


def _final_message(steps: list[dict[str, Any]], record) -> str | None:
    """The model's own last word: its STOP text or rationale, else the last pause reason."""
    for step in reversed(steps):
        if step["action_type"] == "stop":
            # The answer of an information task is the STOP text, not the rationale.
            return step.get("text") or step["why"] or step["detail"]
    if steps:
        return steps[-1]["why"] or steps[-1]["detail"]
    return getattr(record, "pause_reason", None) or record.hitl_message


def _usage_cost(run_id: str) -> float | None:
    try:
        from operon.api.observer import summarize_run_usage

        return round(summarize_run_usage(run_id).estimated_cost_usd, 4)
    except Exception:  # noqa: BLE001 - a report without cost beats no report
        logger.debug("usage summary failed for %s", run_id, exc_info=True)
        return None


async def build_report(run_id: str) -> dict[str, Any]:
    """A short result report: outcome, what it did, what the person approved or did."""
    loop, record = await _require_run(run_id)
    steps = _steps(loop, run_id)
    status = record.status
    stop_reason = record.stop_reason.value if record.stop_reason is not None else None
    started = _started_at.get(run_id)
    finished = _finished_at.get(run_id) if status in TERMINAL else None
    if started is not None and status in TERMINAL and finished is None:
        finished = _finished_at.setdefault(run_id, time.time())
    duration = round(finished - started, 1) if started is not None and finished is not None else None
    failed_steps = sum(1 for step in steps if not step["ok"])
    outcome = _OUTCOME.get(status, status.value)
    approvals = getattr(record, "approval_count", 0)
    takeovers = getattr(record, "takeover_count", 0)
    summary = f"{outcome} after {record.step_count} step{'s' if record.step_count != 1 else ''}"
    if approvals:
        summary += f", {approvals} approval{'s' if approvals != 1 else ''}"
    if takeovers:
        summary += f", {takeovers} takeover{'s' if takeovers != 1 else ''}"
    summary += "."
    return {
        "run_id": run_id,
        "intent": record.intent,
        "environment": _environment_of(run_id),
        "status": status.value,
        "outcome": outcome,
        "summary": summary,
        "stop_reason": stop_reason,
        "final_message": _final_message(steps, record),
        "steps": len(steps),
        "failed_steps": failed_steps,
        "approvals": approvals,
        "takeovers": takeovers,
        "permission_mode": resolve_permission_mode(getattr(record, "permission_mode", None)).value,
        "duration_s": duration,
        "cost_usd": _usage_cost(run_id),
        "run_dir": str(Path(loop.run_store.root_dir) / run_id),
        "timeline": steps,
    }


def format_report(report: dict[str, Any]) -> str:
    """The report as plain text, for ``operon run``."""
    lines = [
        f"{report['outcome']}: {report['intent']}",
        report["summary"],
    ]
    if report.get("final_message"):
        lines.append(f"Last word: {report['final_message']}")
    facts = [f"mode {report['permission_mode']}"]
    if report.get("failed_steps"):
        facts.append(f"{report['failed_steps']} step(s) failed")
    if report.get("duration_s") is not None:
        facts.append(f"{report['duration_s']}s")
    if report.get("cost_usd") is not None:
        facts.append(f"about ${report['cost_usd']:.4f}")
    if report.get("stop_reason"):
        facts.append(f"stop reason {report['stop_reason']}")
    lines.append(" · ".join(facts))
    lines.append(f"Artifacts: {report['run_dir']}")
    return "\n".join(lines)
