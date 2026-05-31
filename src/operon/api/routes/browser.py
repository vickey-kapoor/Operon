"""Browser automation run-control routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from operon.api import ws_stream as _ws_stream
from operon.api.dependencies import (
    cancel_auto_run,
    cleanup_cancelled_run,
    ensure_cdp_ready,
    get_agent_loop,
    schedule_auto_run,
    test_safe_mode_enabled,
    validate_run_id,
)
from operon.models.common import (
    CleanupRequest,
    CleanupResponse,
    HealthResponse,
    ResumeRequest,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopRunRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new run and immediately kick off the agent loop in the background."""
    if not test_safe_mode_enabled():
        await ensure_cdp_ready(mode=request.mode)
    loop = get_agent_loop()
    response = await loop.start_run(request)
    _ws_stream.set_active_run(response.run_id)
    _ws_stream.publish_event({
        "type": "run_started",
        "run_id": response.run_id,
        "intent": request.intent,
        "status": response.status.value,
    })
    if test_safe_mode_enabled():
        logger.info("[run-task] test safe mode active; auto-run skipped for run %s", response.run_id)
    else:
        schedule_auto_run(loop, response.run_id, request.max_steps)
    return response


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


@router.post("/step", response_model=RunResponse)
async def step_run(request: StepRequest) -> RunResponse:
    """Advance an existing browser run by one loop step."""
    validate_run_id(request.run_id)
    try:
        loop = get_agent_loop()
        response = await loop.step_run(request)
        step_event = _build_step_event(loop, response)
        _ws_stream.publish_event(step_event)

        if response.status.value == "waiting_for_user":
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

        if response.status.value in ("succeeded", "failed", "cancelled"):
            _ws_stream.set_active_run(None)
            _ws_stream.publish_event({
                "type": "run_completed",
                "run_id": response.run_id,
                "status": response.status.value,
            })
        return response
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unhandled exception in step_run for %s: %s", request.run_id, exc)
        loop = get_agent_loop()
        try:
            state = await loop.run_store.get_run(request.run_id)
            if state is not None and state.status not in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                updated = await loop.run_store.set_status(request.run_id, RunStatus.FAILED)
                return RunResponse(
                    run_id=updated.run_id,
                    status=updated.status,
                    intent=updated.intent,
                    step_count=updated.step_count,
                )
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/resume", response_model=RunResponse)
async def resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a HITL-paused run and continue the auto-loop."""
    validate_run_id(request.run_id)
    loop = get_agent_loop()
    try:
        response = await loop.resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    schedule_auto_run(loop, request.run_id, max_steps=200)
    return response


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup(request: CleanupRequest) -> CleanupResponse:
    """Close resources created during a browser-native run."""
    loop = get_agent_loop()
    executor = loop.executor
    if not hasattr(executor, "cleanup_run"):
        return CleanupResponse(run_id=request.run_id, closed_count=0, detail="Executor does not support cleanup")
    closed = executor.cleanup_run(request.run_id)
    detail = f"Closed {closed} browser session(s)" if closed else "No browser sessions to close"
    return CleanupResponse(run_id=request.run_id, closed_count=closed, detail=detail)


@router.get("/run/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Return the current state for a stored browser run."""
    validate_run_id(run_id)
    run = await get_agent_loop().run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
        hitl_message=getattr(run, "hitl_message", None),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a minimal in-process health response."""
    return HealthResponse(status="ok")


@router.post("/stop", response_model=RunResponse)
async def stop_run(request: StopRunRequest) -> RunResponse:
    """Cancel an active browser run. Safe to call on already-terminal runs."""
    validate_run_id(request.run_id)
    loop = get_agent_loop()
    run = await loop.run_store.get_run(request.run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    cancelled_now = False
    if run.status not in terminal:
        run = await loop.run_store.set_status(request.run_id, RunStatus.CANCELLED)
        cancelled_now = True
    if cancelled_now:
        await cancel_auto_run(request.run_id)
        await cleanup_cancelled_run(loop, request.run_id)
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )


@router.post("/run/{run_id}/stop")
async def stop_run_by_id(run_id: str) -> dict:
    """Stop a browser run by path parameter. Idempotent on already-terminal runs."""
    response = await stop_run(StopRunRequest(run_id=run_id))
    return {"run_id": response.run_id, "status": response.status.value}


@router.post("/run/{run_id}/pause")
async def pause_run(run_id: str) -> dict:
    """Pause an active browser run by transitioning it to waiting_for_user."""
    validate_run_id(run_id)
    run_store = get_agent_loop().run_store
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.status == RunStatus.RUNNING:
        run = await run_store.set_status(run_id, RunStatus.WAITING_FOR_USER)
    return {"run_id": run_id, "status": run.status.value}
