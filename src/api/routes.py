"""Thin HTTP routes for the MVP browser-only API surface."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse

from src.agent.capture import BrowserCaptureService
from src.agent.loop import AgentLoop
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.api.observer import artifact_path_for_request, list_runs, load_run_snapshot
from src.clients.gemini import GeminiHttpClient
from src.executor.browser import PlaywrightBrowserExecutor
from src.executor.desktop import DesktopExecutor
from src.models.common import (
    HealthResponse,
    ResumeRequest,
    RunResponse,
    RunTaskRequest,
    StepRequest,
)
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore

router = APIRouter()
_agent_loop: AgentLoop | None = None
_desktop_agent_loop: AgentLoop | None = None
_DESKTOP_HTML_PATH = Path(__file__).resolve().parent / "static" / "desktop.html"
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def get_agent_loop() -> AgentLoop:
    """Build the MVP runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        gemini_client = GeminiHttpClient()
        executor = PlaywrightBrowserExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=BrowserCaptureService(executor=executor),
            perception_service=GeminiPerceptionService(gemini_client=gemini_client),
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=GeminiPolicyService(gemini_client=gemini_client),
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
        )
    return _agent_loop


def get_desktop_agent_loop() -> AgentLoop:
    """Build the desktop runtime lazily so env loading can happen first."""
    global _desktop_agent_loop
    if _desktop_agent_loop is None:
        gemini_client = GeminiHttpClient(timeout_seconds=120.0)
        executor = DesktopExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _desktop_agent_loop = AgentLoop(
            capture_service=BrowserCaptureService(executor=executor),
            perception_service=GeminiPerceptionService(
                gemini_client=gemini_client,
                prompt_path=_PROMPTS_DIR / "desktop_perception_prompt.txt",
            ),
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=GeminiPolicyService(
                    gemini_client=gemini_client,
                    prompt_path=_PROMPTS_DIR / "desktop_policy_prompt.txt",
                ),
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
        )
    return _desktop_agent_loop


@router.post("/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new run record for the Gmail draft workflow."""
    return await get_agent_loop().start_run(request)


@router.post("/step", response_model=RunResponse)
async def step_run(request: StepRequest) -> RunResponse:
    """Advance an existing run by one placeholder loop step."""
    try:
        return await get_agent_loop().step_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/resume", response_model=RunResponse)
async def resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a run that is paused waiting for user input."""
    try:
        return await get_agent_loop().resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/run/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Return the current state for a stored run."""
    run_store = get_agent_loop().run_store
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a minimal in-process health response."""
    return HealthResponse(status="ok")


@router.get("/observer/api/runs")
async def observer_runs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Return recent local runs for the observer sidebar."""
    return {"runs": list_runs(limit=limit)}


@router.get("/observer/api/run/{run_id}")
async def observer_run(run_id: str) -> dict:
    """Return the current observer snapshot for one run."""
    try:
        return load_run_snapshot(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/observer/api/artifact")
async def observer_artifact(path: str = Query(..., min_length=1)) -> FileResponse:
    """Serve a local artifact from the runs directory."""
    try:
        artifact_path = artifact_path_for_request(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return FileResponse(artifact_path)


# ── Desktop (computer-use) routes ───────────────────────────────


@router.get("/", response_class=HTMLResponse)
@router.get("/desktop-pilot", response_class=HTMLResponse)
async def pilot_ui() -> str:
    """Serve the Operon Pilot UI (unified desktop + browser)."""
    return _DESKTOP_HTML_PATH.read_text(encoding="utf-8")


@router.post("/desktop/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def desktop_run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new desktop automation run."""
    return await get_desktop_agent_loop().start_run(request)


@router.post("/desktop/step", response_model=RunResponse)
async def desktop_step_run(request: StepRequest) -> RunResponse:
    """Advance an existing desktop run by one step."""
    try:
        return await get_desktop_agent_loop().step_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/desktop/resume", response_model=RunResponse)
async def desktop_resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a desktop run paused for user input."""
    try:
        return await get_desktop_agent_loop().resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/desktop/run/{run_id}", response_model=RunResponse)
async def desktop_get_run(run_id: str) -> RunResponse:
    """Return current state for a desktop run."""
    run_store = get_desktop_agent_loop().run_store
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )
