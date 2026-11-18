"""``operon mcp`` (KAN-96): Operon as an MCP server over stdio.

Claude Code, Claude Desktop and other MCP clients hand a computer-use task to Operon
and follow it. Computer-use runs take minutes, so no tool blocks until the run ends:
``start_task`` returns at once and the client polls with ``get_task`` / ``wait_for_task``.

The server runs tasks in this process through the same front door as ``operon run``
(``cli.LocalClient``), so permission modes, approvals and the trust gate behave the same.
It always uses permission mode ``ask`` and never approves for the client: a paused action
waits until the client calls ``approve`` or ``deny``. One run at a time.

The MCP SDK is an optional extra (``pip install operon-agent[mcp]``); it is imported only
here, never by ``operon`` itself.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from typing import Any

from operon.api.frontdoor import FrontDoorError

MCP_INSTALL_HINT = (
    "operon mcp needs the MCP SDK, which is an optional extra. Install it with:\n"
    '  pip install "operon-agent[mcp]"'
)

MAX_WAIT_S = 300
MAX_SCREENSHOT_EDGE = 1280
_POLL_S = 1.0

INSTRUCTIONS = """\
Operon is a vision-first computer-use agent: it drives a browser or the desktop from screenshots.
Tasks take minutes, so work in steps:
1. start_task(task, url?) returns a run_id at once. Only one run can be active at a time.
2. wait_for_task(run_id) returns when the run finishes, needs you, or the wait times out
   (call it again to keep waiting). get_task(run_id) is the instant version.
3. When the status is waiting_for_user, read waiting.reason (a gated action or a question from
   Operon) and tell the user. Call approve(run_id) only if the user agrees, otherwise deny(run_id).
   Never approve on your own.
4. When done is true, final_answer holds the result. stop_task(run_id) cancels a run.
get_screenshot(run_id) shows the latest screen; screenshots capture the whole screen.
Desktop runs (desktop=true) drive the real mouse and keyboard of this computer.\
"""


class OperonMCPTools:
    """The tool bodies, on one in-process front-door client. Holds the active run id."""

    def __init__(self, client: Any = None, *, poll_s: float = _POLL_S) -> None:
        self._client = client
        self._poll_s = poll_s
        self.active_run_id: str | None = None
        self._start_lock = asyncio.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            from operon.cli import LocalClient

            self._client = LocalClient()
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def _call(self, coro) -> dict[str, Any]:
        try:
            return await coro
        except FrontDoorError as exc:
            raise ValueError(str(exc)) from exc

    async def start_task(
        self, task: str, url: str | None = None, desktop: bool = False, env: str | None = None
    ) -> dict[str, Any]:
        # The client dispatches each request as its own task: check and start are one step.
        async with self._start_lock:
            return await self._start_task(task, url, desktop, env)

    async def _start_task(self, task: str, url: str | None, desktop: bool, env: str | None) -> dict[str, Any]:
        if self.active_run_id is not None:
            try:
                current = await self.client.status(self.active_run_id)
            except Exception:  # noqa: BLE001 - an unknown run no longer blocks a new one
                current = None
            if current is not None and not current["done"]:
                raise ValueError(
                    f"Run {self.active_run_id} is still {current['status']}. Operon runs one task at a time: "
                    "wait for it, or call stop_task first."
                )
            self.active_run_id = None
        requested = _check_environment(env, desktop)
        payload: dict[str, Any] = {
            "intent": task,
            "environment": "desktop" if desktop else "browser",
            "permission_mode": "ask",  # the client approves; Operon never does it for them
        }
        if url:
            payload["start_url"] = url
        if requested is not None:
            # The desktop environment is built once per process, so the first choice stays: it is
            # also what the app catalog and list_environments read for the rest of the process.
            os.environ["OPERON_ENVIRONMENT"] = requested
        try:
            status = await self._call(self.client.start(payload))
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - a missing key and the like reach the client as text
            raise ValueError(f"Could not start the task: {exc}") from exc
        self.active_run_id = status["run_id"]
        return {"run_id": status["run_id"], "status": status["status"], "environment": status["environment"]}

    async def get_task(self, run_id: str) -> dict[str, Any]:
        status = await self._call(self.client.status(run_id))
        steps = status.get("recent_steps", [])
        last = steps[-1] if steps else None
        waiting = status.get("waiting")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": status["status"],
            "done": status["done"],
            "steps": status["step_count"],
            "max_steps": status["max_steps"],
            "last_message": (last["why"] or last["detail"]) if last else None,
            "last_action": last["action"] if last else None,
            "waiting": None,
        }
        if waiting is not None:
            payload["waiting"] = {"kind": waiting["kind"], "reason": waiting["reason"]}
            for key in ("gate", "verdict", "category"):  # verdict: "deny" is a hard policy block
                if waiting.get(key):
                    payload["waiting"][key] = waiting[key]
        if status["done"]:
            report = await self._call(self.client.report(run_id))
            payload["final_answer"] = report.get("final_message")
            payload["summary"] = report.get("summary")
            payload["cost_usd"] = report.get("cost_usd")
        return payload

    async def wait_for_task(self, run_id: str, timeout_s: int = 60) -> dict[str, Any]:
        deadline = time.monotonic() + max(0, min(int(timeout_s), MAX_WAIT_S))
        while True:
            payload = await self.get_task(run_id)
            if payload["done"] or payload["waiting"] is not None or time.monotonic() >= deadline:
                return payload
            await asyncio.sleep(self._poll_s)

    async def approve(self, run_id: str) -> dict[str, Any]:
        await self._call(self.client.approve(run_id))
        return await self.get_task(run_id)

    async def deny(self, run_id: str) -> dict[str, Any]:
        await self._call(self.client.decline(run_id, None))
        return await self.get_task(run_id)

    async def stop_task(self, run_id: str) -> dict[str, Any]:
        await self._call(self.client.stop(run_id))
        return await self.get_task(run_id)

    async def screenshot_png(self, run_id: str) -> bytes:
        """The run's latest screenshot as PNG bytes, no larger than 1280 px on its long edge."""
        status = await self._call(self.client.status(run_id))
        path = status.get("latest_screenshot")
        if not path or not os.path.exists(path):
            raise ValueError("The run has no screenshot yet. Wait for its first step.")
        return await asyncio.to_thread(_downscaled_png, path)

    async def list_environments(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(_environments)


def _check_environment(env: str | None, desktop: bool) -> str | None:
    """Validate ``env`` and return it (lower-cased) when this call should select an environment.

    The desktop environment is built once per process, so a later call cannot switch it:
    asking for a different one than the process already uses is refused, never quietly
    served by the old one (which may be the live desktop)."""
    from operon.core.runtime import current_runtime
    from operon.environments.registry import environment_name, provider_names

    name = env.strip().lower() if env and env.strip() else None
    if name is not None:
        if not desktop:
            raise ValueError("env picks a desktop environment; pass desktop=true with it.")
        if name not in provider_names():
            raise ValueError(f"Unknown environment {env!r}. Choose one of: {', '.join(provider_names())}.")
    if not desktop:
        return None
    provider = current_runtime().desktop_provider
    if provider is not None:
        wanted = name or environment_name()
        if wanted != provider.name:
            raise ValueError(
                f"This operon mcp process already uses environment {provider.name!r}; "
                f"restart it to switch to {wanted!r}."
            )
        return None
    return name


def _downscaled_png(path: str, max_edge: int = MAX_SCREENSHOT_EDGE) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge))
        out = io.BytesIO()
        image.convert("RGB").save(out, format="PNG")
        return out.getvalue()


def _environments() -> list[dict[str, Any]]:
    from operon.environments.registry import (
        environment_name,
        get_provider,
        provider_names,
    )

    current = environment_name()
    rows = []
    for name in provider_names():
        provider = get_provider(name)
        check = provider.check()
        rows.append({
            "name": name,
            "ready": check.ok,
            "problems": list(check.problems),
            "summary": provider.summary,
            "selected": name == current,
        })
    return rows


def build_server(tools: OperonMCPTools | None = None):
    """The MCP server with Operon's tools. Raises ImportError when the ``mcp`` extra is missing."""
    from contextlib import asynccontextmanager

    from mcp.server.mcpserver import Image, MCPServer

    tools = tools or OperonMCPTools()

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield
        finally:
            await tools.close()  # in the server's own event loop: release what a run left open

    server = MCPServer("operon", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool()
    async def start_task(task: str, url: str | None = None, desktop: bool = False, env: str | None = None) -> dict[str, Any]:
        """Start a computer-use task and return its run_id at once (it does not wait for the result).

        Runs in the browser by default; url is the page to open first. desktop=true runs on the
        desktop instead: on the default `live` environment that drives the real mouse and keyboard of
        this computer, so set it only when the user asked for a desktop task. env picks another
        desktop environment (see list_environments). Permission mode is always "ask": risky actions
        pause until you call approve or deny. Only one run can be active at a time."""
        return await tools.start_task(task, url, desktop, env)

    @server.tool()
    async def get_task(run_id: str) -> dict[str, Any]:
        """Current state of a run: status, step count, last message, what it waits on (waiting: reason,
        verdict "confirm" or "deny", category), and once done the final_answer and cost."""
        return await tools.get_task(run_id)

    @server.tool()
    async def wait_for_task(run_id: str, timeout_s: int = 60) -> dict[str, Any]:
        """Wait until the run finishes, needs your approval, or timeout_s passes (at most 300), then
        return get_task's result. Call it again if the run is still going."""
        return await tools.wait_for_task(run_id, timeout_s)

    @server.tool()
    async def approve(run_id: str) -> dict[str, Any]:
        """Approve the action a waiting run paused on. Call this only when the user agrees. If
        waiting.verdict is "deny" the pause is a hard policy block: approving it overrides the block,
        so treat it as the person's own decision, never your own."""
        return await tools.approve(run_id)

    @server.tool()
    async def deny(run_id: str) -> dict[str, Any]:
        """Refuse the action a waiting run paused on; Operon is told not to do it and picks another way."""
        return await tools.deny(run_id)

    @server.tool()
    async def stop_task(run_id: str) -> dict[str, Any]:
        """Cancel a run."""
        return await tools.stop_task(run_id)

    async def get_screenshot(run_id: str) -> Image:
        """The run's latest screenshot (at most 1280 px on the long edge). Screenshots capture the whole
        screen, so anything open there can appear in it."""
        return Image(data=await tools.screenshot_png(run_id), format="png")

    # `Image` is imported here, so the string annotation (from __future__) cannot resolve by name.
    get_screenshot.__annotations__["return"] = Image
    server.tool()(get_screenshot)

    @server.tool()
    async def list_environments() -> list[dict[str, Any]]:
        """The desktop run environments and whether each is ready on this machine (like `operon envs`)."""
        return await tools.list_environments()

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the server on stdio until the client closes it. Exit 2 when the ``mcp`` extra is missing."""
    tools = OperonMCPTools()
    try:
        server = build_server(tools)
    except ImportError:
        print(MCP_INSTALL_HINT, file=sys.stderr)
        return 2
    from operon.core.settings import startup_problems

    errors, _warnings = startup_problems()
    if errors:  # stdout is the protocol, so everything human goes to stderr
        print("Invalid Operon configuration:\n  - " + "\n  - ".join(errors), file=sys.stderr)
        return 1
    # stdout is the protocol: mcp>=2.3's stdio transport redirects fd 1, so a stray print
    # from a run cannot corrupt it. The browser or desktop a run left open is closed by the
    # server's lifespan, inside its own event loop.
    server.run("stdio")
    return 0
