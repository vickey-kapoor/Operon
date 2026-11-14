"""The guest agent (KAN-80): Operon's hands inside a run environment.

A provider that runs tasks on another machine (Windows Sandbox, a Linux container,
a VM) starts ``operon guest`` there. It serves that machine's desktop executor over
HTTP and nothing else: screenshots, actions, the live view and takeover input. The
host keeps the Gemini key, the model calls, the loop and the run files, so the guest
never holds a secret beyond its own one-time token.

Every request needs ``Authorization: Bearer <token>``. Files the executor writes
(screenshots, traces) travel back in the response as base64, so paths never cross.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import secrets
import socket
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from operon.models.policy import AgentAction

logger = logging.getLogger(__name__)

DEFAULT_GUEST_PORT = 8765
TOKEN_HEADER = "authorization"

# Executor methods the host may call by name (all optional on the executor).
CALLABLE_METHODS = frozenset(
    {
        "reset_desktop",
        "context_reset",
        "session_reset",
        "set_current_run_id",
        "close_apps_when_done",
        "aclose_run",
        "aclose",
    }
)
# ExecutedAction fields that name a file on the guest; their bytes are sent back.
FILE_FIELDS = ("artifact_path", "execution_trace_artifact_path")


class ExecuteRequest(BaseModel):
    action: AgentAction


class CallRequest(BaseModel):
    method: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class LiveFrameRequest(BaseModel):
    run_id: str


class UserInputRequest(BaseModel):
    run_id: str
    input_type: str
    x_ratio: float = 0.0
    y_ratio: float = 0.0
    text: str = ""
    key: str = ""
    delta_y: int = 300


def _read_b64(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        logger.debug("guest: could not read %s", path, exc_info=True)
        return None


def build_guest_executor():
    """This machine's desktop executor, with the live view (the person is not here),
    plus Operon's tools when ``OPERON_TOOLS`` (passed on by the host) turns them on:
    tools run here, so they act on this machine, never on the host."""
    from operon.agent.app_catalog import LINUX, desktop_platform
    from operon.agent.tools import enabled_tools

    if desktop_platform() == LINUX:
        from operon.executor.linux_desktop import LinuxDesktopExecutor

        executor = LinuxDesktopExecutor()
    else:
        from operon.executor.desktop import DesktopExecutor
        from operon.executor.live_view import LiveViewMixin

        class GuestWindowsExecutor(LiveViewMixin, DesktopExecutor):
            pass

        executor = GuestWindowsExecutor()
    tools = enabled_tools()
    if not tools:
        return executor
    from operon.executor.tools import ToolExecutor

    return ToolExecutor(executor, tools)


def create_guest_app(executor: Any, token: str) -> FastAPI:
    """The guest's HTTP API around ``executor``. ``token`` guards every route."""
    if not token:
        raise ValueError("the guest agent needs a token")
    expected = f"Bearer {token}".encode()

    def require_token(authorization: str = Header(default="")) -> None:
        if not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(status_code=401, detail="bad or missing guest token")

    app = FastAPI(title="Operon guest", docs_url=None, redoc_url=None, openapi_url=None)
    guarded = [Depends(require_token)]

    @app.get("/health", dependencies=guarded)
    async def health() -> dict[str, Any]:
        from operon import __version__
        from operon.agent.app_catalog import desktop_platform

        return {"ok": True, "platform": desktop_platform(), "version": __version__}

    @app.post("/capture", dependencies=guarded)
    async def capture() -> dict[str, Any]:
        frame = await executor.capture()
        png = _read_b64(frame.artifact_path)
        _remove(frame.artifact_path)
        return {"frame": frame.model_dump(mode="json"), "png": png}

    @app.post("/execute", dependencies=guarded)
    async def execute(body: ExecuteRequest) -> dict[str, Any]:
        result = await executor.execute(body.action)
        files = {name: _read_b64(getattr(result, name)) for name in FILE_FIELDS}
        return {"result": result.model_dump(mode="json"), "files": files}

    @app.post("/call", dependencies=guarded)
    async def call(body: CallRequest) -> dict[str, Any]:
        if body.method not in CALLABLE_METHODS:
            raise HTTPException(status_code=400, detail=f"method {body.method!r} is not callable")
        method = getattr(executor, body.method, None)
        if method is None:
            return {"present": False, "result": None}
        result = method(**body.kwargs)
        if hasattr(result, "__await__"):
            result = await result
        try:
            json.dumps(result)
        except TypeError:
            result = None
        return {"present": True, "result": result}

    @app.post("/live_frame", dependencies=guarded)
    async def live_frame(body: LiveFrameRequest) -> Response:
        hook = getattr(executor, "live_frame_png", None)
        frame = await hook(body.run_id) if hook is not None else None
        if frame is None:
            return Response(status_code=204)
        return Response(content=frame, media_type="image/jpeg")

    @app.post("/user_input", dependencies=guarded)
    async def user_input(body: UserInputRequest) -> dict[str, Any]:
        hook = getattr(executor, "user_input", None)
        if hook is None:
            return {"ok": False}
        try:
            ok = await hook(body.run_id, **body.model_dump(exclude={"run_id"}))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": bool(ok)}

    return app


def _remove(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def local_ip() -> str:
    """This machine's address on its default route (what the host connects to)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))  # no packet is sent for UDP connect
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def write_ready_file(path: Path, *, host: str, port: int) -> None:
    """Tell the provider where to connect, atomically (it polls for this file)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    os.replace(tmp, path)


def serve(args) -> int:
    """``operon guest``: serve this machine's desktop to the host that started it."""
    import sys

    import uvicorn

    if args.log_file:
        # pythonw (no console, so no window in the screenshots) has no stdout/stderr.
        log = open(args.log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives as long as the process
        sys.stdout = sys.stderr = log
        logging.basicConfig(stream=log, level=logging.INFO, force=True)

    token = (Path(args.token_file).read_text(encoding="utf-8").strip() if args.token_file else "") or os.getenv(
        "OPERON_GUEST_TOKEN", ""
    )
    if not token:
        print("operon guest: give --token-file or OPERON_GUEST_TOKEN", flush=True)
        return 2
    app = create_guest_app(build_guest_executor(), token)
    if args.ready_file:
        write_ready_file(Path(args.ready_file), host=args.advertise or local_ip(), port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def new_token() -> str:
    return secrets.token_urlsafe(32)


def add_parser(sub) -> None:
    parser = sub.add_parser("guest", help="(inside a run environment) serve this desktop to the host")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default all)")
    parser.add_argument("--port", type=int, default=DEFAULT_GUEST_PORT)
    parser.add_argument("--token-file", default=None, help="file holding the token (else $OPERON_GUEST_TOKEN)")
    parser.add_argument("--ready-file", default=None, help="write {host, port} here once listening")
    parser.add_argument("--log-file", default=None, help="append output here (for windowless Python)")
    parser.add_argument("--advertise", default=None, help="address to put in the ready file (default: own IP)")
    parser.set_defaults(func=serve)
