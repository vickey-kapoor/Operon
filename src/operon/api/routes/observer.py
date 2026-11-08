"""Observer and artifact routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response

from operon.api.dependencies import get_agent_loop, validate_run_id
from operon.api.observer import (
    artifact_path_for_request,
    build_run_bundle,
    list_runs,
    load_run_snapshot,
    reconcile_orphaned_browser_run,
    usage_dashboard,
)
from operon.models.common import RunStatus

router = APIRouter()


@router.get("/observer/api/runs")
async def observer_runs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Return recent local runs for the observer sidebar."""
    executor = get_agent_loop().executor
    recent_runs = list_runs(limit=limit)
    if hasattr(executor, "current_url_for_run"):
        for run in recent_runs:
            if run["status"] != RunStatus.RUNNING.value:
                continue
            current_url = await executor.current_url_for_run(run["run_id"])
            reconcile_orphaned_browser_run(run["run_id"], has_live_session=current_url is not None)
    return {"runs": list_runs(limit=limit)}


@router.get("/observer/api/run/{run_id}")
async def observer_run(run_id: str) -> dict:
    """Return the current observer snapshot for one run."""
    validate_run_id(run_id)
    current_url = None
    executor = get_agent_loop().executor
    if hasattr(executor, "current_url_for_run"):
        current_url = await executor.current_url_for_run(run_id)
    reconcile_orphaned_browser_run(run_id, has_live_session=current_url is not None)
    try:
        snapshot = load_run_snapshot(run_id)
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    snapshot["run"]["current_url"] = current_url
    return snapshot


@router.get("/observer/api/usage")
async def observer_usage(limit: int = Query(default=500, ge=1, le=2000)) -> dict:
    """Return aggregated model usage and cost estimates across recent runs."""
    return usage_dashboard(limit=limit)


@router.get("/observer/api/artifact")
async def observer_artifact(path: str = Query(..., min_length=1)) -> FileResponse:
    """Serve a local artifact from the runs directory."""
    try:
        artifact_path = artifact_path_for_request(path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found") from None
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from None
    return FileResponse(artifact_path)


@router.get("/observer/api/export/{run_id}")
async def observer_export_run(run_id: str) -> Response:
    """Download a zip bundle of run artifacts and related outputs."""
    validate_run_id(run_id)
    try:
        bundle = build_run_bundle(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    headers = {"Content-Disposition": f'attachment; filename="{run_id}.zip"'}
    return Response(content=bundle, media_type="application/zip", headers=headers)


@router.get("/observer/api/live-browser/{run_id}")
async def observer_live_browser(run_id: str) -> Response:
    """Return a fresh PNG frame from an active browser-native run."""
    validate_run_id(run_id)
    executor = get_agent_loop().executor
    if not hasattr(executor, "live_frame_png"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Live browser view not available") from None
    png_bytes = await executor.live_frame_png(run_id)
    if png_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active browser session not found") from None
    return Response(content=png_bytes, media_type="image/png")
