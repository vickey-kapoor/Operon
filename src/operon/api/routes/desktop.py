"""Desktop automation run-control routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from operon.api.dependencies import get_desktop_agent_loop, validate_run_id
from operon.models.common import (
    CleanupRequest,
    CleanupResponse,
    ResumeRequest,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopRunRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/desktop/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def desktop_run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new desktop automation run."""
    return await get_desktop_agent_loop().start_run(request)


@router.post("/desktop/step", response_model=RunResponse)
async def desktop_step_run(request: StepRequest) -> RunResponse:
    """Advance an existing desktop run by one step."""
    validate_run_id(request.run_id)
    try:
        return await get_desktop_agent_loop().step_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/desktop/resume", response_model=RunResponse)
async def desktop_resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a desktop run paused for user input."""
    validate_run_id(request.run_id)
    try:
        return await get_desktop_agent_loop().resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/desktop/stop", response_model=RunResponse)
async def desktop_stop_run(request: StopRunRequest) -> RunResponse:
    """Cancel an active desktop run. Safe to call on already-terminal runs."""
    validate_run_id(request.run_id)
    loop = get_desktop_agent_loop()
    run = await loop.run_store.get_run(request.run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    if run.status not in terminal:
        run = await loop.run_store.set_status(request.run_id, RunStatus.CANCELLED)
        try:
            await loop._cleanup_completed_run(request.run_id)
        except Exception as exc:
            logger.warning("cleanup after desktop cancellation failed for %s: %s", request.run_id, exc)
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )


@router.post("/desktop/run/{run_id}/stop")
async def desktop_stop_run_by_id(run_id: str) -> dict:
    """Cancel a desktop run by path parameter. Idempotent on terminal runs."""
    response = await desktop_stop_run(StopRunRequest(run_id=run_id))
    return {"run_id": response.run_id, "status": response.status.value}


@router.post("/desktop/cleanup", response_model=CleanupResponse)
async def desktop_cleanup(request: CleanupRequest) -> CleanupResponse:
    """Close applications launched during a desktop run."""
    loop = get_desktop_agent_loop()
    executor = loop.executor
    if not hasattr(executor, "cleanup_run"):
        return CleanupResponse(run_id=request.run_id, closed_count=0, detail="Executor does not support cleanup")
    closed = executor.cleanup_run(request.run_id)
    detail = f"Closed {closed} application(s)" if closed else "No applications to close"
    return CleanupResponse(run_id=request.run_id, closed_count=closed, detail=detail)


@router.get("/desktop/run/{run_id}", response_model=RunResponse)
async def desktop_get_run(run_id: str) -> RunResponse:
    """Return current state for a desktop run."""
    validate_run_id(run_id)
    run = await get_desktop_agent_loop().run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )
