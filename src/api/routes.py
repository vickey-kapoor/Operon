"""Thin HTTP routes for the MVP browser-only API surface."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.agent.capture import BrowserCaptureService
from src.agent.loop import AgentLoop
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import GeminiHttpClient
from src.executor.browser import PlaywrightBrowserExecutor
from src.models.common import HealthResponse, RunResponse, RunTaskRequest, StepRequest
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore

router = APIRouter()
_agent_loop: AgentLoop | None = None


def get_agent_loop() -> AgentLoop:
    """Build the MVP runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        gemini_client = GeminiHttpClient()
        browser_executor = PlaywrightBrowserExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=BrowserCaptureService(browser_executor=browser_executor),
            perception_service=GeminiPerceptionService(gemini_client=gemini_client),
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=GeminiPolicyService(gemini_client=gemini_client),
                memory_store=memory_store,
            ),
            browser_executor=browser_executor,
            verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
        )
    return _agent_loop


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
