"""HTTP routes for the Operon API surface."""

from __future__ import annotations

import asyncio
import importlib
import logging
import re as _re
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

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
from src.agent.video_verifier import VideoVerifier
from src.api import ws_stream as _ws_stream
from src.api.benchmark_suite import (
    create_single_task_suite,
    create_suite,
    get_all_tasks,
    get_suite,
    run_suite_background,
    stop_suite,
)
from src.api.observer import (
    artifact_path_for_request,
    build_run_bundle,
    list_runs,
    load_run_snapshot,
    reconcile_orphaned_browser_run,
    usage_dashboard,
)
from src.api.runtime_config import browser_mode_config, desktop_mode_config
from src.clients.anthropic import AnthropicHttpClient
from src.clients.gemini import GeminiHttpClient
from src.clients.gemini_computer_use import GeminiComputerUseHttpClient
from src.core.contracts.perception import Environment as UnifiedEnvironment
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
    StrictModel,
)
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore

logger = logging.getLogger(__name__)

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
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
_CONSOLE_HTML_PATH = _STATIC_DIR / "console.html"
_DASHBOARD_HTML_PATH = _STATIC_DIR / "dashboard" / "index.html"
_BENCHMARKS_HTML_PATH = _STATIC_DIR / "benchmarks" / "index.html"
_COMMAND_CENTER_HTML_PATH = _STATIC_DIR / "command-center" / "index.html"
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
    # Pass the static prompt template as cacheable system prompt. On Vertex AI
    # this primes a context cache on first call, cutting input tokens by ~90%
    # for the static prefix. On the public Gemini API this is a no-op.
    prompt_path = _PROMPTS_DIR / prompt_name
    cacheable = _read_static_prefix(prompt_path)
    return GeminiPolicyService(
        gemini_client=GeminiHttpClient(
            model=config.primary_model,
            timeout_seconds=120.0,
            cacheable_system_prompt=cacheable,
        ),
        prompt_path=prompt_path,
    )


def _read_static_prefix(prompt_path: Path) -> str | None:
    """Read the static prefix of a prompt template (everything up to the first
    `{` placeholder). This is the portion that doesn't change per-request and
    is the only piece worth caching."""
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except OSError:
        return None
    idx = text.find("{")
    if idx <= 0:
        return None
    prefix = text[:idx].rstrip()
    return prefix or None


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
    if config.backend == "browserbase":
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


def _build_browser_executor():
    browser_config = browser_mode_config()
    if browser_config.backend == "browserbase":
        return importlib.import_module(
            "src.executor.browserbase_native"
        ).BrowserbaseNativeBrowserExecutor()
    global NativeBrowserExecutor
    if NativeBrowserExecutor is None:
        NativeBrowserExecutor = importlib.import_module("src.executor.browser_native").NativeBrowserExecutor
    return NativeBrowserExecutor()


def get_agent_loop() -> AgentLoop:
    """Build the MVP runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        browser_config = browser_mode_config()
        verifier_client = _build_verifier_client(config=browser_config)
        executor = _build_browser_executor()
        services = _build_browser_services(executor)
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        # Video verification always uses Gemini regardless of the verifier provider,
        # because VideoVerifier calls generate_video_verification on a GeminiClient.
        video_gemini_client = GeminiHttpClient(
            model=browser_config.primary_model, timeout_seconds=120.0
        )
        video_verifier = VideoVerifier(video_gemini_client)
        _agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
                element_buffer=getattr(services.perception_service, "element_buffer", None),
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(
                gemini_client=verifier_client,
                video_verifier=video_verifier,
            ),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=video_gemini_client,
            environment=UnifiedEnvironment.BROWSER,
        )
        _ws_stream.set_executor(executor)
        _ws_stream.set_agent_loop(_agent_loop)
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
        video_gemini_client = GeminiHttpClient(
            model=desktop_config.primary_model, timeout_seconds=120.0
        )
        desktop_video_verifier = VideoVerifier(video_gemini_client)
        _desktop_agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
                element_buffer=getattr(services.perception_service, "element_buffer", None),
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(
                gemini_client=verifier_client,
                video_verifier=desktop_video_verifier,
            ),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            gemini_client=video_gemini_client,
            environment=UnifiedEnvironment.DESKTOP,
        )
    return _desktop_agent_loop



@router.post("/run-task", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_task(request: RunTaskRequest) -> RunResponse:
    """Create a new run and immediately kick off the agent loop in the background."""
    await _ensure_cdp_ready(mode=request.mode)
    loop = get_agent_loop()
    response = await loop.start_run(request)
    _ws_stream.set_active_run(response.run_id)
    _ws_stream.publish_event({
        "type": "run_started",
        "run_id": response.run_id,
        "intent": request.intent,
        "status": response.status.value,
    })
    task = asyncio.create_task(_auto_run_loop(loop, response.run_id, request.max_steps))
    task.add_done_callback(
        lambda t: logger.exception("[loop] run %s crashed", response.run_id, exc_info=t.exception())
        if not t.cancelled() and t.exception() is not None else None
    )
    logger.info("[run-task] loop scheduled for run %s (max_steps=%d)", response.run_id, request.max_steps)
    return response


async def _ensure_cdp_ready(mode: str = "batch") -> None:
    """Ensure a CDP browser is connected before the first step runs.

    For observable mode the executor owns Playwright Chromium launch and
    BrowserManager wiring — this function is a no-op in that case.

    Priority (batch/cdp modes):
      1. BrowserManager already connected → no-op.
      2. Port 9222 reachable (user's Chrome already open) → connect to it.
      3. Otherwise launch Chrome with --remote-debugging-port=9222, then connect.
    """
    if mode == "observable":
        return
    import subprocess
    import sys
    import tempfile

    import httpx

    from src.browser.manager import (
        BrowserManager,
        get_active_manager,
        set_active_manager,
    )

    # Already connected — nothing to do.
    bm = get_active_manager()
    if bm is not None and bm.is_connected:
        return

    # Check if Chrome is already listening on 9222.
    port_reachable = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            r = await client.get("http://localhost:9222/json/version")
        port_reachable = r.status_code == 200
    except Exception:
        pass

    if not port_reachable:
        # Launch Chrome with remote debugging enabled.
        chrome_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        if sys.platform == "darwin":
            chrome_candidates = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
        elif sys.platform.startswith("linux"):
            chrome_candidates = ["google-chrome", "chromium-browser", "chromium"]

        chrome_exe = next((p for p in chrome_candidates if _chrome_exists(p)), None)
        if chrome_exe is None:
            logger.warning("_ensure_cdp_ready: no Chrome found — cannot auto-launch")
            return

        profile_dir = tempfile.mkdtemp(prefix="operon-cdp-")
        args = [
            chrome_exe,
            "--remote-debugging-port=9222",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ]
        try:
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("_ensure_cdp_ready: Chrome launched, waiting for port 9222…")
        except Exception as exc:
            logger.warning("_ensure_cdp_ready: Chrome launch failed — %s", exc)
            return

        # Wait up to 12 s for the port to become reachable.
        deadline = asyncio.get_event_loop().time() + 12
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    r = await client.get("http://localhost:9222/json/version")
                if r.status_code == 200:
                    break
            except Exception:
                pass
        else:
            logger.warning("_ensure_cdp_ready: port 9222 not reachable after 12 s")
            return

    # Connect BrowserManager.
    existing = get_active_manager()
    if existing is not None:
        try:
            await existing.disconnect()
        except Exception:
            pass

    manager = BrowserManager()
    try:
        await manager.connect(9222)
        await manager.start_screencast(fps=15)
    except Exception as exc:
        logger.warning("_ensure_cdp_ready: connect failed — %s", exc)
        return

    _ws_stream.set_browser_manager(manager)
    set_active_manager(manager)
    logger.info("_ensure_cdp_ready: CDP browser connected and screencast live")


def _chrome_exists(path: str) -> bool:
    import shutil
    from pathlib import Path
    return Path(path).exists() or shutil.which(path) is not None


async def _auto_run_loop(loop: AgentLoop, run_id: str, max_steps: int) -> None:
    """Fire-and-forget step loop: runs until terminal state or max_steps."""
    from src.models.common import RunStatus, StepRequest, StopReason

    logger.info("[loop] starting run %s", run_id)
    while True:
        state = await loop.run_store.get_run(run_id)
        if state is None:
            logger.warning("[loop] run %s disappeared from store", run_id)
            break
        if state.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.WAITING_FOR_USER,
            RunStatus.CANCELLED,
        }:
            logger.info("[loop] run %s terminal: %s", run_id, state.status.value)
            break
        if state.step_count >= max_steps:
            logger.warning("[loop] run %s hit max_steps=%d — stopping", run_id, max_steps)
            state.stop_reason = StopReason.MAX_STEP_LIMIT_REACHED
            await loop.run_store.set_status(run_id, RunStatus.FAILED)
            try:
                await loop._cleanup_completed_run(run_id)
            except Exception as exc:
                logger.warning("[loop] cleanup after max_steps failed: %s", exc)
            break
        logger.info("[loop] run %s entering step %d", run_id, state.step_count + 1)
        await loop.step_run(StepRequest(run_id=run_id))



@router.post("/step", response_model=RunResponse)
async def step_run(request: StepRequest) -> RunResponse:
    """Advance an existing run by one placeholder loop step."""
    import logging as _logging
    _step_log = _logging.getLogger(__name__)
    _validate_run_id(request.run_id)
    try:
        loop = get_agent_loop()
        response = await loop.step_run(request)

        # Build the step event, enriched with policy decision details from the step artifact.
        _step_event: dict = {
            "type": "step",
            "run_id": response.run_id,
            "step": response.step_count,
            "status": response.status.value,
        }
        _pd_path = (
            Path(loop.run_store.root_dir)
            / response.run_id
            / f"step_{response.step_count}"
            / "policy_decision.json"
        )
        if _pd_path.exists():
            try:
                import json as _json
                _pd = _json.loads(_pd_path.read_text(encoding="utf-8"))
                _step_event["confidence"] = _pd.get("confidence")
                _step_event["subgoal"] = _pd.get("active_subgoal")
                _step_event["action_type"] = _pd.get("action", {}).get("action_type")
                _step_event["rule_name"] = _pd.get("rule_name")
                _step_event["rationale"] = _pd.get("rationale")
                _step_event["action_expected_change"] = _pd.get("expected_change")
                # Derive a display target for the command-center step stream
                _a = _pd.get("action", {})
                _atype = _a.get("action_type", "")
                if _atype == "navigate":
                    _step_event["action_target"] = (_a.get("url") or "").replace("https://", "").replace("http://", "").split("?")[0][:70]
                elif _atype in ("type", "select"):
                    _step_event["action_target"] = _a.get("selector") or _a.get("target_element_id") or ""
                    _step_event["action_value"] = _a.get("text")
                elif _atype in ("press_key", "hotkey"):
                    _step_event["action_target"] = _a.get("key") or ""
                elif _atype == "launch_app":
                    _step_event["action_target"] = _a.get("text") or ""
                elif _atype == "scroll":
                    _amt = _a.get("scroll_amount", 0) or 0
                    _step_event["action_target"] = "down" if _amt > 0 else "up"
                else:
                    _step_event["action_target"] = _a.get("selector") or _a.get("target_element_id") or ""
                # Perception summary from the sibling artifact
                _perc_path = _pd_path.parent / "perception_parsed.json"
                if _perc_path.exists():
                    try:
                        _perc = _json.loads(_perc_path.read_text(encoding="utf-8"))
                        _page_hint = _perc.get("page_hint", "")
                        _elem_count = len(_perc.get("elements", []))
                        _focused = _perc.get("focused_element") or {}
                        _focused_text = str(_focused.get("text", ""))[:40] if _focused else ""
                        _parts: list[str] = []
                        if _page_hint:
                            _parts.append(_page_hint)
                        if _focused_text:
                            _parts.append(f"focus: {_focused_text}")
                        elif _elem_count:
                            _parts.append(f"{_elem_count} elements")
                        _step_event["perception"] = " · ".join(_parts) if _parts else None
                    except Exception:
                        pass
                # Publish a dedicated action_intent event when coordinates are known.
                _action = _pd.get("action", {})
                if _action.get("x") is not None and _action.get("y") is not None:
                    _ws_stream.publish_event({
                        "type": "action_intent",
                        "action_type": _action.get("action_type", "unknown"),
                        "x": _action["x"],
                        "y": _action["y"],
                        "viewport_w": 1920,
                        "viewport_h": 1080,
                        "confidence": _pd.get("confidence", 0.0),
                        "rule_name": _pd.get("rule_name"),
                        "subgoal": _pd.get("active_subgoal"),
                        "rationale": _pd.get("rationale", ""),
                    })
            except Exception:
                pass
        _ws_stream.publish_event(_step_event)

        if response.status.value == "waiting_for_user":
            # Publish hitl_required so the Cockpit left pane shows the approval card.
            _hitl_state = await loop.run_store.get_run(response.run_id)
            _ws_stream.publish_event({
                "type": "hitl_required",
                "run_id": response.run_id,
                "message": _hitl_state.hitl_message if _hitl_state else None,
                "confidence": _step_event.get("confidence"),
                "threshold": _ws_stream.get_confidence_threshold(),
                "pending_action": _step_event.get("action_type"),
                "subgoal": _step_event.get("subgoal"),
                "rationale": _step_event.get("rationale"),
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
        _step_log.exception("unhandled exception in step_run for %s: %s", request.run_id, exc)
        # Mark the run as failed and return a clean response rather than a bare 500
        loop = get_agent_loop()
        try:
            from src.models.common import RunStatus
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
    _validate_run_id(request.run_id)
    loop = get_agent_loop()
    try:
        response = await loop.resume_run(request.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    task = asyncio.create_task(_auto_run_loop(loop, request.run_id, max_steps=200))
    task.add_done_callback(
        lambda t: logger.exception("[loop] resumed run %s crashed", request.run_id, exc_info=t.exception())
        if not t.cancelled() and t.exception() is not None else None
    )
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
        hitl_message=getattr(run, "hitl_message", None),
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


@router.get("/console", response_class=HTMLResponse)
async def task_console_ui() -> str:
    """Serve the command center UI (legacy /console path)."""
    return _COMMAND_CENTER_HTML_PATH.read_text(encoding="utf-8")


@router.get("/")
async def root_ui() -> RedirectResponse:
    """Redirect root to the command center."""
    return RedirectResponse(url="/command-center", status_code=307)


@router.get("/command-center", response_class=HTMLResponse)
async def command_center_root_ui() -> str:
    """Serve the command center UI."""
    return _COMMAND_CENTER_HTML_PATH.read_text(encoding="utf-8")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui() -> str:
    """Serve the standalone usage dashboard UI."""
    return _DASHBOARD_HTML_PATH.read_text(encoding="utf-8")


@router.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_ui() -> str:
    """Serve the WebArena benchmark suite runner UI."""
    return _BENCHMARKS_HTML_PATH.read_text(encoding="utf-8")


def _step_log_to_event(record: dict) -> dict | None:
    """Map a StepLog or PreStepFailureLog JSONL record to the SSE step event shape."""
    import json as _json  # noqa: F401 — local import avoids module-level shadowing

    if record.get("record_type") == "pre_step_failure":
        return {
            "n": record.get("step_index", 0),
            "action": "error",
            "target": (record.get("error_message") or "")[:70],
            "value": None,
            "source": "system",
            "conf": None,
            "expects": None,
            "status": "failed",
        }
    pd = record.get("policy_decision") or {}
    action = pd.get("action") or {}
    atype = action.get("action_type", "unknown")
    target = ""
    value = None
    if atype == "navigate":
        url = action.get("url") or ""
        target = url.replace("https://", "").replace("http://", "").split("?")[0][:70]
    elif atype in ("type", "select"):
        target = action.get("selector") or action.get("target_element_id") or ""
        text = action.get("text") or ""
        if text and len(text) <= 80:
            value = text
    elif atype in ("press_key", "hotkey"):
        target = action.get("key") or ""
    elif atype == "launch_app":
        target = action.get("text") or ""
    elif atype == "scroll":
        amt = action.get("scroll_amount") or 0
        target = "down" if amt > 0 else "up"
    elif atype == "wait":
        target = f"{action.get('wait_ms') or 0}ms"
    else:
        target = action.get("selector") or action.get("target_element_id") or ""
    failure = record.get("failure")
    expected = pd.get("expected_change") or "none"
    return {
        "n": record.get("step_index", 0),
        "action": atype,
        "target": target or (pd.get("active_subgoal") or "")[:70],
        "value": value,
        "source": pd.get("rule_name") or "LLM",
        "conf": pd.get("confidence"),
        "expects": expected if expected != "none" else None,
        "status": "failed" if failure else "done",
    }


@router.get("/command-center/api/run/{run_id}/stream")
async def command_center_run_stream(run_id: str) -> StreamingResponse:
    """SSE: replay existing run.jsonl steps, then tail new ones at 200 ms. Heartbeat every 15 s."""
    import json as _json
    _validate_run_id(run_id)
    jsonl_path = _RUNS_DIR / run_id / "run.jsonl"

    async def _generate():
        offset = 0
        if jsonl_path.exists():
            raw = jsonl_path.read_bytes()
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = _step_log_to_event(_json.loads(line))
                    if ev:
                        yield f"data: {_json.dumps(ev)}\n\n"
                except Exception:
                    pass
            offset = len(raw)
        ticks = 0
        while True:
            await asyncio.sleep(0.2)
            ticks += 1
            if ticks >= 75:  # 75 × 200 ms = 15 s
                yield ": heartbeat\n\n"
                ticks = 0
            if not jsonl_path.exists():
                continue
            size = jsonl_path.stat().st_size
            if size <= offset:
                continue
            with jsonl_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            offset += len(chunk)
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    ev = _step_log_to_event(_json.loads(line))
                    if ev:
                        yield f"data: {_json.dumps(ev)}\n\n"
                except Exception:
                    pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/command-center/api/run/{run_id}/screenshot")
async def command_center_screenshot(run_id: str) -> Response:
    """Return the current browser viewport as PNG for 2 Hz screenshot polling."""
    _validate_run_id(run_id)
    executor = get_agent_loop().executor
    if not hasattr(executor, "live_frame_png"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screenshot not available")
    png_bytes = await executor.live_frame_png(run_id)
    if png_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active browser session")
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/run/{run_id}/stop")
async def stop_run_by_id(run_id: str) -> dict:
    """Stop a run by path parameter. Idempotent on already-terminal runs."""
    _validate_run_id(run_id)
    run_store = get_agent_loop().run_store
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    if run.status not in terminal:
        run = await run_store.set_status(run_id, RunStatus.CANCELLED)
    return {"run_id": run_id, "status": run.status.value}


@router.post("/run/{run_id}/pause")
async def pause_run(run_id: str) -> dict:
    """Pause an active run by transitioning it to waiting_for_user."""
    _validate_run_id(run_id)
    run_store = get_agent_loop().run_store
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.status == RunStatus.RUNNING:
        run = await run_store.set_status(run_id, RunStatus.WAITING_FOR_USER)
    return {"run_id": run_id, "status": run.status.value}


@router.get("/command-center/{run_id}", response_class=HTMLResponse)
async def command_center_ui(run_id: str) -> str:
    """Serve the live command center for a specific run."""
    _validate_run_id(run_id)
    return _COMMAND_CENTER_HTML_PATH.read_text(encoding="utf-8")


# ── Benchmark suite ─────────────────────────────────────────────


_VALID_DIFFICULTIES = {"easy", "medium", "hard", "all"}


class _SuiteRequest(StrictModel):
    difficulty: str = "all"
    source: str = "all"
    max_steps: int = 15
    headless: bool = False
    mode: str = "batch"


class _SingleTaskRequest(StrictModel):
    task_id: str
    max_steps: int = 15
    headless: bool = False
    mode: str = "batch"


@router.post("/benchmark/run-suite")
async def benchmark_run_suite(body: _SuiteRequest) -> dict:
    """Start a benchmark suite run in the background and return its suite_id."""
    difficulty = body.difficulty.lower()
    if difficulty not in _VALID_DIFFICULTIES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"difficulty must be one of {sorted(_VALID_DIFFICULTIES)}")
    max_steps = max(5, min(body.max_steps, 40))
    suite = create_suite(difficulty, source=body.source)
    asyncio.create_task(run_suite_background(suite.suite_id, max_steps, get_agent_loop, headless=body.headless, mode=body.mode))
    return {"suite_id": suite.suite_id, "status": suite.status, "total": suite.total, "difficulty": suite.difficulty}


@router.post("/benchmark/stop-suite/{suite_id}")
async def benchmark_stop_suite(suite_id: str) -> dict:
    """Cancel a running benchmark suite."""
    stopped = stop_suite(suite_id)
    if not stopped:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Suite not found or not running")
    return {"suite_id": suite_id, "cancelled": True}


@router.post("/benchmark/run-task")
async def benchmark_run_task(body: _SingleTaskRequest) -> dict:
    """Start a single benchmark task in the background; returns suite_id for polling."""
    max_steps = max(5, min(body.max_steps, 40))
    try:
        suite = create_single_task_suite(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    asyncio.create_task(run_suite_background(suite.suite_id, max_steps, get_agent_loop, headless=body.headless, mode=body.mode))
    return {"suite_id": suite.suite_id, "task_id": body.task_id, "status": "running"}


@router.get("/benchmark/tasks")
async def benchmark_tasks(difficulty: str = "all", source: str = "all") -> list:
    """Return all benchmark tasks from all datasets for UI initialisation."""
    return get_all_tasks(difficulty=difficulty, source=source)


@router.get("/benchmark/suite/{suite_id}")
async def benchmark_suite_status(suite_id: str) -> dict:
    """Return the current status of a benchmark suite run."""
    suite = get_suite(suite_id)
    if suite is None:
        # Return a terminal EXPIRED state instead of a bare 404.
        # A bare 404 causes the browser agent to loop endlessly polling a
        # stale suite_id that was lost when the server restarted.
        return {
            "suite_id": suite_id,
            "status":   "expired",
            "detail":   "Suite state was lost — server restarted. Do not retry this suite_id.",
        }
    return suite.to_dict()


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
