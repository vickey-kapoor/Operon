"""WebSocket events for a completed agent step.

Shared by the ``/step`` route and the background auto-run loop, so a run
streams the same step, HITL and completion events however it is driven.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from operon.agent.loop import AgentLoop
from operon.api import ws_stream as _ws_stream
from operon.models.common import RunResponse, RunStatus

logger = logging.getLogger(__name__)

_TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


def _action_target(action: dict) -> tuple[str, str | None]:
    action_type = action.get("action_type", "")
    if action_type == "navigate":
        target = (action.get("url") or "").replace("https://", "").replace("http://", "").split("?")[0][:70]
        return target, None
    if action_type in ("type", "select"):
        return action.get("selector") or action.get("target_element_id") or "", action.get("text")
    if action_type in ("press_key", "hotkey"):
        return action.get("key") or "", None
    if action_type == "launch_app":
        return action.get("text") or "", None
    if action_type == "scroll":
        amount = action.get("scroll_amount", 0) or 0
        return ("down" if amount > 0 else "up"), None
    return action.get("selector") or action.get("target_element_id") or "", None


def _read_perception_summary(perception_path: Path) -> str | None:
    if not perception_path.exists():
        return None
    try:
        perception = json.loads(perception_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    page_hint = perception.get("page_hint", "")
    element_count = len(perception.get("elements", []))
    focused = perception.get("focused_element") or {}
    focused_text = str(focused.get("text", ""))[:40] if focused else ""
    parts: list[str] = []
    if page_hint:
        parts.append(page_hint)
    if focused_text:
        parts.append(f"focus: {focused_text}")
    elif element_count:
        parts.append(f"{element_count} elements")
    return " . ".join(parts) if parts else None


def _build_step_event(loop, response: RunResponse) -> dict:
    event: dict = {
        "type": "step",
        "run_id": response.run_id,
        "step": response.step_count,
        "status": response.status.value,
    }
    policy_path = (
        Path(loop.run_store.root_dir)
        / response.run_id
        / f"step_{response.step_count}"
        / "policy_decision.json"
    )
    if not policy_path.exists():
        return event
    try:
        policy_decision = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return event

    action = policy_decision.get("action", {})
    target, value = _action_target(action)
    event.update({
        "confidence": policy_decision.get("confidence"),
        "subgoal": policy_decision.get("active_subgoal"),
        "action_type": action.get("action_type"),
        "rule_name": policy_decision.get("rule_name"),
        "rationale": policy_decision.get("rationale"),
        "action_expected_change": policy_decision.get("expected_change"),
        "action_target": target,
    })
    if value is not None:
        event["action_value"] = value

    perception = _read_perception_summary(policy_path.parent / "perception_parsed.json")
    if perception:
        event["perception"] = perception

    if action.get("x") is not None and action.get("y") is not None:
        _ws_stream.publish_event({
            "type": "action_intent",
            "action_type": action.get("action_type", "unknown"),
            "x": action["x"],
            "y": action["y"],
            "viewport_w": 1920,
            "viewport_h": 1080,
            "confidence": policy_decision.get("confidence", 0.0),
            "rule_name": policy_decision.get("rule_name"),
            "subgoal": policy_decision.get("active_subgoal"),
            "rationale": policy_decision.get("rationale", ""),
        })
    return event


def publish_run_completed(run_id: str, run_status: RunStatus) -> None:
    _ws_stream.set_active_run(None)
    _ws_stream.publish_event({
        "type": "run_completed",
        "run_id": run_id,
        "status": run_status.value,
    })


async def publish_step_outcome(loop: AgentLoop, response: RunResponse) -> None:
    """Publish the step event, plus HITL or completion events when the run paused or ended."""
    step_event = _build_step_event(loop, response)
    _ws_stream.publish_event(step_event)

    if response.status is RunStatus.WAITING_FOR_USER:
        hitl_state = await loop.run_store.get_run(response.run_id)
        _ws_stream.publish_event({
            "type": "hitl_required",
            "run_id": response.run_id,
            "message": hitl_state.hitl_message if hitl_state else None,
            "confidence": step_event.get("confidence"),
            "threshold": _ws_stream.get_confidence_threshold(),
            "pending_action": step_event.get("action_type"),
            "subgoal": step_event.get("subgoal"),
            "rationale": step_event.get("rationale"),
        })

    if response.status in _TERMINAL:
        publish_run_completed(response.run_id, response.status)


async def mark_run_failed(loop: AgentLoop, run_id: str) -> RunResponse | None:
    """Mark a run FAILED after an unhandled step error; None if it is already terminal or gone."""
    try:
        state = await loop.run_store.get_run(run_id)
        if state is None or state.status in _TERMINAL:
            return None
        updated = await loop.run_store.set_status(run_id, RunStatus.FAILED)
    except Exception as exc:
        logger.warning("could not mark run %s failed: %s", run_id, exc)
        return None
    publish_run_completed(run_id, updated.status)
    return RunResponse(
        run_id=updated.run_id,
        status=updated.status,
        intent=updated.intent,
        step_count=updated.step_count,
    )
