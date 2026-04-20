"""HTTP routes for the Operon API surface."""

from __future__ import annotations

import importlib
import re as _re
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse, Response

from core.contracts.perception import Environment as UnifiedEnvironment
from src.agent.anthropic_policy import AnthropicPolicyService
from src.agent.backend import AgentBackend
from src.agent.browser_computer_use import BrowserComputerUseBackend
from src.agent.browser_json import BrowserJsonBackend
from src.agent.capture import ScreenCaptureService
from src.agent.combined import CombinedPerceptionPolicyService
from src.agent.fallback_backend import FallbackBackend
from src.agent.loop import AgentLoop
from src.agent.perception import GeminiPerceptionService, PerceptionService
from src.agent.policy import GeminiPolicyService, PolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.api.observer import (
    artifact_path_for_request,
    build_run_bundle,
    list_runs,
    load_run_snapshot,
    reconcile_orphaned_browser_run,
)
from src.api.runtime_config import browser_mode_config, desktop_mode_config
from src.clients.anthropic import AnthropicHttpClient
from src.clients.gemini import GeminiHttpClient
from src.clients.gemini_computer_use import GeminiComputerUseHttpClient
from src.models.common import (
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
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore

router = APIRouter()
_agent_loop: AgentLoop | None = None
_desktop_agent_loop: AgentLoop | None = None
DesktopExecutor = None
NativeBrowserExecutor = None
_MAX_RUN_ID_LENGTH = 64


_RUN_ID_RE = _re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_run_id(run_id: str) -> None:
    """Reject run_ids that would cause filesystem issues or path traversal."""
    if len(run_id) > _MAX_RUN_ID_LENGTH or not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
_CONSOLE_HTML_PATH = Path(__file__).resolve().parent / "static" / "console.html"
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class _RuntimeServices:
    perception_service: PerceptionService
    policy_delegate: PolicyService


def _build_json_backend(*, prompt_name: str, model: str, timeout_seconds: float = 120.0) -> AgentBackend:
    return CombinedPerceptionPolicyService(
        gemini_client=GeminiHttpClient(model=model, timeout_seconds=timeout_seconds),
        prompt_path=_PROMPTS_DIR / prompt_name,
    )


def _build_policy_delegate(*, config, prompt_name: str) -> PolicyService:
    planner_provider = config.planner_provider.lower()
    if planner_provider == "anthropic":
        planner_model = config.planner_model or "claude-sonnet-4-20250514"
        return AnthropicPolicyService(
            anthropic_client=AnthropicHttpClient(model=planner_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / prompt_name,
        )
    return GeminiPolicyService(
        gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
        prompt_path=_PROMPTS_DIR / prompt_name,
    )


def _build_verifier_client(*, config):
    verifier_provider = config.verifier_provider.lower()
    verifier_model = config.verifier_model or config.fallback_model or config.primary_model
    if verifier_provider == "anthropic":
        verifier_model = verifier_model or "claude-sonnet-4-20250514"
        return AnthropicHttpClient(model=verifier_model, timeout_seconds=120.0)
    return GeminiHttpClient(model=verifier_model, timeout_seconds=120.0)


def _build_browser_services(executor) -> _RuntimeServices:
    config = browser_mode_config()
    if config.backend == "json":
        if config.planner_provider.lower() == "anthropic":
            return _RuntimeServices(
                perception_service=GeminiPerceptionService(
                    gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
                    prompt_path=_PROMPTS_DIR / "perception_prompt.txt",
                ),
                policy_delegate=_build_policy_delegate(config=config, prompt_name="policy_prompt.txt"),
            )
        backend = BrowserJsonBackend(
            gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
        )
        return _RuntimeServices(perception_service=backend, policy_delegate=backend)
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
            backend = FallbackBackend(primary=primary, secondary=secondary)
            return _RuntimeServices(perception_service=backend, policy_delegate=backend)
        return _RuntimeServices(perception_service=primary, policy_delegate=primary)
    raise ValueError(f"Unsupported browser backend {config.backend!r}")


def _build_desktop_services() -> _RuntimeServices:
    config = desktop_mode_config()
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Only 'json' is implemented in this slice."
        )
    if config.planner_provider.lower() == "anthropic":
        return _RuntimeServices(
            perception_service=GeminiPerceptionService(
                gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
                prompt_path=_PROMPTS_DIR / "desktop_perception_prompt.txt",
            ),
            policy_delegate=_build_policy_delegate(config=config, prompt_name="desktop_policy_prompt.txt"),
        )
    backend = _build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )
    return _RuntimeServices(perception_service=backend, policy_delegate=backend)


def get_agent_loop() -> AgentLoop:
    """Build the MVP runtime lazily so env loading can happen first."""
    global _agent_loop, NativeBrowserExecutor
    if _agent_loop is None:
        if NativeBrowserExecutor is None:
            NativeBrowserExecutor = importlib.import_module("src.executor.browser_native").NativeBrowserExecutor
        browser_config = browser_mode_config()
        verifier_client = _build_verifier_client(config=browser_config)
        executor = NativeBrowserExecutor()
        services = _build_browser_services(executor)
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=verifier_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=verifier_client,
            environment=UnifiedEnvironment.BROWSER,
        )
    return _agent_loop


def get_desktop_agent_loop() -> AgentLoop:
    """Build the desktop runtime lazily so env loading can happen first."""
    global _desktop_agent_loop, DesktopExecutor
    if _desktop_agent_loop is None:
        if DesktopExecutor is None:
            from src.executor.desktop import DesktopExecutor as _DesktopExecutor

            DesktopExecutor = _DesktopExecutor

        desktop_config = desktop_mode_config()
        services = _build_desktop_services()
        verifier_client = _build_verifier_client(config=desktop_config)
        executor = DesktopExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _desktop_agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=verifier_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=verifier_client,
            environment=UnifiedEnvironment.DESKTOP,
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


@router.post("/stop", response_model=RunResponse)
async def stop_run(request: StopRunRequest) -> RunResponse:
    """Cancel an active run. Safe to call on already-terminal runs."""
    _validate_run_id(request.run_id)
    run_store = get_agent_loop().run_store
    run = await run_store.get_run(request.run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    if run.status not in terminal:
        run = await run_store.set_status(request.run_id, RunStatus.CANCELLED)
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        intent=run.intent,
        step_count=run.step_count,
    )


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
    _validate_run_id(run_id)
    current_url = None
    executor = get_agent_loop().executor
    if hasattr(executor, "current_url_for_run"):
        current_url = await executor.current_url_for_run(run_id)
    reconcile_orphaned_browser_run(run_id, has_live_session=current_url is not None)
    try:
        snapshot = load_run_snapshot(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    except OSError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    snapshot["run"]["current_url"] = None
    if current_url is not None:
        snapshot["run"]["current_url"] = current_url
    return snapshot


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
    _validate_run_id(run_id)
    try:
        bundle = build_run_bundle(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found") from None
    headers = {"Content-Disposition": f'attachment; filename="{run_id}.zip"'}
    return Response(content=bundle, media_type="application/zip", headers=headers)


@router.get("/observer/api/live-browser/{run_id}")
async def observer_live_browser(run_id: str) -> Response:
    """Return a fresh PNG frame from an active browser-native run."""
    _validate_run_id(run_id)
    executor = get_agent_loop().executor
    if not hasattr(executor, "live_frame_png"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Live browser view not available") from None
    png_bytes = await executor.live_frame_png(run_id)
    if png_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active browser session not found") from None
    return Response(content=png_bytes, media_type="image/png")


# ── UI ──────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
@router.get("/console", response_class=HTMLResponse)
async def task_console_ui() -> str:
    """Serve the task console UI."""
    return _CONSOLE_HTML_PATH.read_text(encoding="utf-8")


# ── Desktop (computer-use) routes ───────────────────────────────


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
