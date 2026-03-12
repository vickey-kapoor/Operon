"""FastAPI REST + WebSocket server for UI Navigator."""

import asyncio
import base64
import collections
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.agent.core import AgentResult, StepEvent, UINavigatorAgent
from src.agent.clarifier import TaskClarifier
from src.api import session_routes
from src.api import webpilot_routes
from src.api.webpilot_routes import cleanup_sessions as _webpilot_cleanup_sessions, init_handler as _webpilot_init_handler
from src.api.models import (
    NavigateRequest,
    NavigateResponse,
    TaskListResponse,
    TaskRecord,
    TaskStatus,
    TaskStatusResponse,
)
from src.api.store import TaskStore, create_store
from src import metrics, tracing
from src.logging_config import configure_logging, request_id_var

# ---------------------------------------------------------------------------
# Logging — configure once at import time so all modules share the format
# ---------------------------------------------------------------------------

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version (single source of truth)
# ---------------------------------------------------------------------------

_VERSION = "1.4.0"

# ---------------------------------------------------------------------------
# Task store (persistent) + live records (in-memory, for WS event replay)
# ---------------------------------------------------------------------------

_store: TaskStore = create_store()

# Live records kept in-memory for WebSocket event replay during active runs.
_live_records: Dict[str, TaskRecord] = {}

# task_id -> list of connected WebSocket clients
_ws_clients: Dict[str, List[WebSocket]] = {}

# task_id -> asyncio.Task — kept so we can cancel running tasks.
_running_tasks: Dict[str, asyncio.Task] = {}

# Semaphore to limit concurrent browser sessions
_MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_TASKS", "5"))
_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_TASK_CHARS = 2_000

# Endpoints excluded from API-key authentication
_AUTH_SKIP_PATHS = {"/health", "/"}


def _get_api_keys() -> frozenset:
    """Return the set of valid API keys from the API_KEYS env var."""
    raw = os.environ.get("API_KEYS", "").strip()
    if not raw:
        return frozenset()
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


# ---------------------------------------------------------------------------
# Rate-limit state (in-memory sliding window per API key)
# ---------------------------------------------------------------------------

_RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))
# api_key -> deque of request timestamps (float, seconds)
_rate_windows: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
_rate_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

_CLEANUP_INTERVAL_SECONDS = 3600  # hourly
_TASK_MAX_AGE_SECONDS = 24 * 3600  # 24 hours
_LIVE_RECORD_TTL_SECONDS = 300.0   # 5 minutes after completion


async def _cleanup_loop() -> None:
    """Periodically delete tasks older than 24 hours from the persistent store."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            deleted = await _store.delete_expired(_TASK_MAX_AGE_SECONDS)
            if deleted:
                logger.info("Cleanup: deleted %d expired tasks", deleted)
        except Exception as exc:
            logger.warning("Cleanup loop error (non-fatal): %s", exc)


async def _cleanup_live_record(task_id: str) -> None:
    """
    Remove a finished task from the live in-memory maps after a short delay.

    The delay gives late-connecting WebSocket clients time to replay events.
    """
    await asyncio.sleep(_LIVE_RECORD_TTL_SECONDS)
    _live_records.pop(task_id, None)
    _ws_clients.pop(task_id, None)
    _running_tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing.setup_tracing()
    cleanup_task = asyncio.create_task(_cleanup_loop())

    # Initialise the WebPilot handler and start its session cleanup task.
    _stub_scenario = os.environ.get("WEBPILOT_STUB")
    if _stub_scenario:
        from src.agent.webpilot_stub import WebPilotStubHandler
        _wp_handler = WebPilotStubHandler(scenario=_stub_scenario)
        _webpilot_init_handler(_wp_handler)
        logger.info("WebPilot running in STUB mode (scenario=%r)", _stub_scenario)
    elif os.environ.get("GOOGLE_API_KEY"):
        from src.agent.vision import GeminiVisionClient
        from src.agent.planner import ActionPlanner
        from src.agent.webpilot_handler import LegacyWebPilotHandler
        _wp_vision = GeminiVisionClient()
        _wp_planner = ActionPlanner(vision_client=_wp_vision)
        # Use legacy handler as the shared TTS/narration provider.
        # Live API handlers are created per-session in webpilot_routes.
        _wp_handler = LegacyWebPilotHandler(vision_client=_wp_vision, planner=_wp_planner)
        # Pass the genai client so webpilot_routes can create per-session Live handlers.
        _webpilot_init_handler(_wp_handler, live_client=_wp_vision._client)
    webpilot_cleanup_task = asyncio.create_task(_webpilot_cleanup_sessions())

    logger.info(
        "UI Navigator server starting up",
        extra={"version": _VERSION, "max_concurrent": _MAX_CONCURRENT},
    )
    if not _get_api_keys():
        logger.warning(
            "API_KEYS is not set — authentication is DISABLED. Set API_KEYS in production."
        )
    if "*" in _cors_origins:
        logger.warning(
            "CORS_ORIGINS contains '*' — all origins are allowed. Restrict in production."
        )
    elif not _cors_raw:
        logger.warning(
            "CORS_ORIGINS is not set — defaulting to chrome-extension://* only. "
            "Set CORS_ORIGINS if other origins need access."
        )
    yield
    # Graceful shutdown: cancel in-flight agent tasks.
    if _running_tasks:
        logger.info(
            "Cancelling %d in-flight task(s) on shutdown", len(_running_tasks)
        )
        tasks = list(_running_tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    cleanup_task.cancel()
    webpilot_cleanup_task.cancel()
    logger.info("UI Navigator server shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UI Navigator",
    description=(
        "AI-powered browser automation using Gemini multimodal + Playwright. "
        "Observe browser screenshots, interpret visual elements, and perform actions."
    ),
    version=_VERSION,
    lifespan=lifespan,
)

# Include routers
app.include_router(session_routes.router)
app.include_router(webpilot_routes.router)

# ---------------------------------------------------------------------------
# Middleware — order matters: last app.add_middleware() runs first on request
# ---------------------------------------------------------------------------


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Attach a correlation ID to every request.

    Reads ``X-Request-ID`` from the incoming headers (so callers can provide
    their own ID for end-to-end tracing) or generates a new UUID.  The ID is
    echoed back in the response header and stored in a ``ContextVar`` so all
    log records emitted within the request context include it automatically.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, and latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            "HTTP %s %s → %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        metrics.emit("request_latency_ms", duration_ms, {"path": request.url.path})
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validate the ``X-API-Key`` header against the ``API_KEYS`` env var.

    - Skips validation if ``API_KEYS`` is not configured (dev / local mode).
    - Always skips ``/health`` and ``/`` so monitoring tools work unauthenticated.
    - Returns ``401`` for missing or invalid keys.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _AUTH_SKIP_PATHS:
            return await call_next(request)

        api_keys = _get_api_keys()
        if not api_keys:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if not provided or provided not in api_keys:
            logger.warning(
                "Rejected request — invalid API key",
                extra={"path": request.url.path},
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-API-key sliding-window rate limiter.

    Window size: 60 seconds.
    Limit: ``RATE_LIMIT_RPM`` requests per minute (default 60).
    Returns ``429`` with a ``Retry-After`` header when exceeded.
    Skips ``/health`` and unauthenticated requests.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _AUTH_SKIP_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return await call_next(request)

        now = time.time()
        window_start = now - 60.0

        async with _rate_lock:
            had_entry = api_key in _rate_windows
            window = _rate_windows[api_key]  # defaultdict creates empty deque if new
            while window and window[0] < window_start:
                window.popleft()
            if had_entry and not window:
                # Key existed but all timestamps expired → evict stale entry and
                # start fresh so the dict doesn't grow without bound.
                del _rate_windows[api_key]
                _rate_windows[api_key].append(now)
                return await call_next(request)

            if len(window) >= _RATE_LIMIT_RPM:
                retry_after = max(1, int(window[0] - window_start) + 1)
                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "path": request.url.path,
                        "rpm_limit": _RATE_LIMIT_RPM,
                        "retry_after": retry_after,
                    },
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )

            window.append(now)

        return await call_next(request)


# CORS origin allowlist — set CORS_ORIGINS as comma-separated list in production.
# Defaults to chrome-extension://* only (restrictive); wildcard is NOT the default.
_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins: List[str] = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw
    else ["chrome-extension://*"]
)
# Always include the Chrome extension origin pattern when using a custom list.
if "chrome-extension://*" not in _cors_origins:
    _cors_origins.append("chrome-extension://*")

# Register middleware — added last = runs first on incoming request
app.add_middleware(RateLimitMiddleware)        # innermost (runs 4th)
app.add_middleware(APIKeyMiddleware)           # runs 3rd
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)                                              # runs 2nd
app.add_middleware(RequestLoggingMiddleware)   # runs just after correlation ID
app.add_middleware(CorrelationIdMiddleware)    # outermost (runs 1st)


# ---------------------------------------------------------------------------
# GCS screenshot replacement helper
# ---------------------------------------------------------------------------


def _maybe_upload_screenshots(agent_result: AgentResult, task_id: str) -> None:
    """
    Replace base64 screenshot strings in ``agent_result`` with GCS signed URLs.

    No-op if ``GCS_BUCKET`` is not configured.  Failures are non-fatal.
    """
    from src import storage

    updated: List[str] = []
    for i, b64 in enumerate(agent_result.screenshots or []):
        try:
            png_bytes = base64.b64decode(b64)
            url = storage.upload_screenshot(png_bytes, task_id, i + 1)
            updated.append(url if url else b64)
        except Exception:
            updated.append(b64)
    agent_result.screenshots = updated


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------


async def _run_agent_task(task_id: str) -> None:
    """Execute an agent task in the background and stream events via WebSocket."""
    record = _live_records.get(task_id)
    if not record:
        logger.error("Task not found in live records", extra={"task_id": task_id})
        return

    record.status = TaskStatus.RUNNING
    await _store.upsert(record)
    metrics.emit("tasks_started")

    api_key = os.environ.get("GOOGLE_API_KEY")

    try:
        async with _semaphore:
            try:
                agent = UINavigatorAgent(
                    mode="browser",
                    api_key=api_key,
                    model=record.model if hasattr(record, "model") else None,
                    system_prompt=record.system_prompt if hasattr(record, "system_prompt") else None,
                )
                agent.task_id = task_id

                def on_step(event: StepEvent) -> None:
                    ws_event = {
                        "type": "step",
                        "step": event.step,
                        "observation": event.observation,
                        "action": "; ".join(event.actions_taken),
                        "screenshot": event.screenshot,
                    }
                    record.events.append(ws_event)
                    asyncio.create_task(_broadcast(task_id, ws_event))

                agent.on_step = on_step

                agent_result: AgentResult = await agent.run(
                    task=record.task,
                    start_url=record.start_url,
                    max_steps=record.max_steps,
                )

                _maybe_upload_screenshots(agent_result, task_id)

                record.result = agent_result
                record.status = TaskStatus.DONE
                await _store.upsert(record)
                metrics.emit(
                    "tasks_completed",
                    labels={"success": str(agent_result.success).lower()},
                )

                done_event = {
                    "type": "done",
                    "success": agent_result.success,
                    "result": agent_result.result or "",
                    "steps_taken": agent_result.steps_taken,
                    "error": agent_result.error,
                }
                record.events.append(done_event)
                await _broadcast(task_id, done_event)

                logger.info(
                    "Task finished",
                    extra={
                        "task_id": task_id,
                        "success": agent_result.success,
                        "steps_taken": agent_result.steps_taken,
                        "error": agent_result.error,
                    },
                )

            except asyncio.CancelledError:
                record.status = TaskStatus.CANCELLED
                await _store.upsert(record)
                metrics.emit("tasks_cancelled")
                cancel_event = {"type": "cancelled", "message": "Task was cancelled."}
                record.events.append(cancel_event)
                await _broadcast(task_id, cancel_event)
                logger.info("Task cancelled", extra={"task_id": task_id})
                raise  # must re-raise so asyncio cleans up the Task correctly

            except Exception as exc:
                logger.exception(
                    "Background task raised: %s", exc, extra={"task_id": task_id}
                )
                record.status = TaskStatus.ERROR
                await _store.upsert(record)
                metrics.emit("tasks_failed", labels={"reason": "exception"})
                error_event = {"type": "error", "message": str(exc)}
                record.events.append(error_event)
                await _broadcast(task_id, error_event)

    finally:
        # Always remove from running_tasks and schedule live-record cleanup.
        _running_tasks.pop(task_id, None)
        asyncio.create_task(_cleanup_live_record(task_id))


async def _broadcast(task_id: str, event: dict) -> None:
    """Send an event to all WebSocket clients subscribed to task_id."""
    # Cap stored events: strip screenshots from older entries to bound memory.
    record = _live_records.get(task_id)
    if record and len(record.events) > 5:
        for old_event in record.events[:-5]:
            old_event.pop("screenshot", None)

    clients = _ws_clients.get(task_id, [])
    dead: List[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            clients.remove(ws)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class ClarifyRequest(BaseModel):
    task: str = Field(..., max_length=_MAX_TASK_CHARS)


class ClarifyResponse(BaseModel):
    questions: List[str]


@app.post("/clarify", response_model=ClarifyResponse)
async def clarify_task(request: ClarifyRequest) -> ClarifyResponse:
    """
    Analyse a task description and return clarifying questions for any ambiguous inputs.
    Returns an empty list if the task is already clear enough to execute.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    clarifier = TaskClarifier(api_key=api_key)
    questions = await clarifier.get_questions(request.task)
    return ClarifyResponse(questions=questions)


@app.post("/navigate", response_model=NavigateResponse, status_code=202)
async def start_navigation(
    request: NavigateRequest,
) -> NavigateResponse:
    """
    Start a navigation task.  Returns immediately with a task_id.
    Use GET /tasks/{task_id} or WebSocket /ws/{task_id} to follow progress.
    Use DELETE /tasks/{task_id} to cancel a running task.
    """
    task_id = str(uuid.uuid4())
    record = TaskRecord(
        task_id=task_id,
        task=request.task,
        start_url=request.start_url,
        max_steps=request.max_steps,
    )
    _live_records[task_id] = record
    _ws_clients[task_id] = []
    await _store.upsert(record)

    bg_task = asyncio.create_task(_run_agent_task(task_id))
    _running_tasks[task_id] = bg_task

    logger.info(
        "Task created",
        extra={"task_id": task_id, "task_preview": request.task[:80]},
    )
    return NavigateResponse(task_id=task_id, status="started")


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> dict:
    """
    Cancel a running or pending navigation task.

    Returns the current task status.  If the task is already finished,
    the status is returned without error.
    """
    record = _live_records.get(task_id) or await _store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    if record.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"task_id": task_id, "status": str(record.status), "message": "Task is not running"}

    bg_task = _running_tasks.get(task_id)
    if bg_task and not bg_task.done():
        bg_task.cancel()
        logger.info("Task cancellation requested", extra={"task_id": task_id})
        return {"task_id": task_id, "status": "cancelling"}

    return {"task_id": task_id, "status": str(record.status)}


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> TaskListResponse:
    """List navigation tasks, sorted by creation time descending."""
    records, total = await _store.list_tasks(status=status, limit=limit, offset=offset)
    tasks = [
        TaskStatusResponse(
            task_id=r.task_id,
            status=str(r.status),
            task=r.task,
            start_url=r.start_url,
            max_steps=r.max_steps,
            result=r.result,
        )
        for r in records
    ]
    return TaskListResponse(tasks=tasks, total=total, limit=limit, offset=offset)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str) -> TaskStatusResponse:
    """Return the current status and result of a navigation task."""
    record = await _store.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return TaskStatusResponse(
        task_id=record.task_id,
        status=str(record.status),
        task=record.task,
        start_url=record.start_url,
        max_steps=record.max_steps,
        result=record.result,
    )


@app.websocket("/ws/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: str) -> None:
    """
    Stream real-time progress events for a navigation task.

    Events emitted:
      - {"type": "step", "step": int, "observation": str, "action": str, "screenshot": base64}
      - {"type": "done", "success": bool, "result": str, "steps_taken": int}
      - {"type": "cancelled", "message": str}
      - {"type": "error", "message": str}
    """
    record = _live_records.get(task_id) or await _store.get(task_id)
    if not record:
        await websocket.close(code=4404, reason=f"Task {task_id!r} not found")
        return

    await websocket.accept()
    _ws_clients.setdefault(task_id, []).append(websocket)
    logger.info("WebSocket client connected", extra={"task_id": task_id})

    try:
        # Replay any events that already happened (late subscriber).
        for event in record.events:
            await websocket.send_json(event)

        # If the task is already finished, close cleanly.
        if record.status in (TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED):
            await websocket.close()
            return

        # Keep the connection alive until the client disconnects.
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        clients = _ws_clients.get(task_id, [])
        try:
            clients.remove(websocket)
        except ValueError:
            pass
        logger.info("WebSocket client disconnected", extra={"task_id": task_id})


@app.post("/screenshot")
async def analyze_screenshot(
    file: Optional[UploadFile] = File(None),
    task: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
) -> dict:
    """
    Take a screenshot (or accept an uploaded one) and analyse it with Gemini.

    - If ``file`` is provided, it is used as the screenshot (max 5 MB).
    - If ``url`` is provided, the agent navigates there and captures a screenshot.
    - If neither, the agent takes a screenshot of about:blank.
    - ``task`` describes what to look for / do (defaults to a generic prompt).
    """
    from PIL import Image

    effective_task = task or "Describe all visible UI elements and their purposes."
    if task and len(task) > _MAX_TASK_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"task must be {_MAX_TASK_CHARS} characters or fewer",
        )

    # Validate URL for SSRF if provided.
    if url:
        from src.api.models import _validate_start_url
        try:
            url = _validate_start_url(url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    api_key = os.environ.get("GOOGLE_API_KEY")

    if file is not None:
        raw = await file.read()
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded image exceeds the {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
                ),
            )

        img = Image.open(io.BytesIO(raw)).convert("RGB")

        from src.agent.vision import GeminiVisionClient
        from src.agent.planner import ActionPlanner

        vision = GeminiVisionClient(api_key=api_key)
        planner = ActionPlanner(vision_client=vision)
        plan = await planner.plan(image=img, task=effective_task)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "screenshot": screenshot_b64,
            "analysis": plan.model_dump(),
        }

    # Navigate to URL (or blank) and capture.
    async with _semaphore:
        agent = UINavigatorAgent(mode="browser", api_key=api_key)
        result = await agent.take_and_analyze_screenshot(
            task=effective_task,
            start_url=url,
        )
        return result


@app.get("/health")
async def health_check() -> dict:
    """Simple health check endpoint — no authentication required."""
    counts = await _store.count_by_status()
    return {
        "status": "ok",
        "version": _VERSION,
        "active_tasks": counts.get("running", 0),
        "total_tasks": sum(counts.values()),
        "task_counts": counts,
    }


@app.get("/")
async def root() -> dict:
    """API root — returns basic service information."""
    return {
        "service": "UI Navigator",
        "version": _VERSION,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
    }


# Mount static files last so API routes take priority.
import pathlib as _pathlib
_static_dir = _pathlib.Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="ui")
else:
    logger.debug("Static directory %s not found — /ui mount skipped", _static_dir)
