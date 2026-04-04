"""HTTP routes for the Operon API surface."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse

from src.agent.backend import AgentBackend
from src.agent.browser_computer_use import BrowserComputerUseBackend
from src.agent.browser_json import BrowserJsonBackend
from src.agent.capture import ScreenCaptureService
from src.agent.combined import CombinedPerceptionPolicyService
from src.agent.fallback_backend import FallbackBackend
from src.agent.loop import AgentLoop
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.api.observer import artifact_path_for_request, list_runs, load_run_snapshot
from src.api.runtime_config import browser_mode_config, desktop_mode_config
from src.clients.gemini import GeminiHttpClient
from src.clients.gemini_computer_use import GeminiComputerUseHttpClient
from src.executor.browser_native import NativeBrowserExecutor
from src.executor.desktop import DesktopExecutor
from src.models.common import (
    CleanupRequest,
    CleanupResponse,
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
_MAX_RUN_ID_LENGTH = 64


def _validate_run_id(run_id: str) -> None:
    """Reject run_ids that would cause filesystem issues."""
    if len(run_id) > _MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
_DESKTOP_HTML_PATH = Path(__file__).resolve().parent / "static" / "desktop.html"
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _build_json_backend(*, prompt_name: str, model: str, timeout_seconds: float = 120.0) -> AgentBackend:
    return CombinedPerceptionPolicyService(
        gemini_client=GeminiHttpClient(model=model, timeout_seconds=timeout_seconds),
        prompt_path=_PROMPTS_DIR / prompt_name,
    )


def _build_browser_backend(executor) -> AgentBackend:
    config = browser_mode_config()
    if config.backend == "json":
        return BrowserJsonBackend(
            gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
        )
    if config.backend == "computer_use":
        primary = BrowserComputerUseBackend(
            client=GeminiComputerUseHttpClient(model=config.primary_model),
            prompt_path=_PROMPTS_DIR / "browser_computer_use_prompt.txt",
            browser_runtime=executor,
        )
        if config.fallback_backend == "json" and config.fallback_model:
            secondary = BrowserJsonBackend(
                gemini_client=GeminiHttpClient(model=config.fallback_model, timeout_seconds=120.0),
                prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
            )
            return FallbackBackend(primary=primary, secondary=secondary)
        return primary
    raise ValueError(f"Unsupported browser backend {config.backend!r}")


def _build_desktop_backend() -> AgentBackend:
    config = desktop_mode_config()
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Only 'json' is implemented in this slice."
        )
    return _build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )


def get_agent_loop() -> AgentLoop:
    """Build the MVP runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        browser_config = browser_mode_config()
        verifier_model = browser_config.verifier_model or browser_config.fallback_model or browser_config.primary_model
        policy_client = GeminiHttpClient(model=verifier_model, timeout_seconds=120.0)
        executor = NativeBrowserExecutor()
        backend = _build_browser_backend(executor)
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=backend,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=backend,
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=policy_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=policy_client,
        )
    return _agent_loop


def get_desktop_agent_loop() -> AgentLoop:
    """Build the desktop runtime lazily so env loading can happen first."""
    global _desktop_agent_loop
    if _desktop_agent_loop is None:
        desktop_config = desktop_mode_config()
        backend = _build_desktop_backend()
        verifier_model = desktop_config.verifier_model or desktop_config.primary_model
        policy_client = GeminiHttpClient(model=verifier_model, timeout_seconds=120.0)
        executor = DesktopExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _desktop_agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=backend,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=backend,
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=policy_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=policy_client,
        )
    return _desktop_agent_loop


@router.post("/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new run record for the Gmail draft workflow."""
    return await get_agent_loop().start_run(request)


@router.post("/step", response_model=RunResponse)
async def step_run(request: StepRequest) -> RunResponse:
    """Advance an existing run by one placeholder loop step."""
    _validate_run_id(request.run_id)
    try:
        return await get_agent_loop().step_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/resume", response_model=RunResponse)
async def resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a run that is paused waiting for user input."""
    _validate_run_id(request.run_id)
    try:
        return await get_agent_loop().resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/run/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Return the current state for a stored run."""
    _validate_run_id(run_id)
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
    _validate_run_id(run_id)
    try:
        return load_run_snapshot(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    except OSError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None


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
    _validate_run_id(request.run_id)
    try:
        return await get_desktop_agent_loop().step_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/desktop/resume", response_model=RunResponse)
async def desktop_resume_run(request: ResumeRequest) -> RunResponse:
    """Resume a desktop run paused for user input."""
    _validate_run_id(request.run_id)
    try:
        return await get_desktop_agent_loop().resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


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
    _validate_run_id(run_id)
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
