"""Browser automation run-control routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from operon.api import ws_stream as _ws_stream
from operon.api.dependencies import (
    cancel_auto_run,
    cleanup_cancelled_run,
    ensure_cdp_ready,
    get_agent_loop,
    mark_run_failed,
    publish_step_outcome,
    resume_and_schedule,
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


@router.post("/step", response_model=RunResponse)
async def step_run(request: StepRequest) -> RunResponse:
    """Advance an existing browser run by one loop step."""
    validate_run_id(request.run_id)
    try:
        loop = get_agent_loop()
        response = await loop.step_run(request)
        await publish_step_outcome(loop, response)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unhandled exception in step_run for %s: %s", request.run_id, exc)
        failed = await mark_run_failed(get_agent_loop(), request.run_id)
        if failed is not None:
            return failed
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/resume", response_model=RunResponse)
async def resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a HITL-paused run and continue the auto-loop."""
    validate_run_id(request.run_id)
    try:
        return await resume_and_schedule(get_agent_loop(), request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup(request: CleanupRequest) -> CleanupResponse:
    """Close resources created during a browser-native run."""
    validate_run_id(request.run_id)
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
