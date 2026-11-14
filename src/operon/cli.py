"""The ``operon`` command (KAN-89).

    operon setup                    first time: save the Gemini key, install the browser, self-check
    operon run "task"               run a task here, in this process, and print a report
    operon run --request "task"     send the task to a running API and follow it
    operon serve                    start the API and print the task page's link
    operon cloud ...                Operon's own computer on an Azure Linux VM (see operon.cloud)

Both ``run`` forms use the same front-door service as the task page
(``operon.api.frontdoor``), so permission modes, approvals, takeover and the report
behave the same everywhere. While a run waits for you, the terminal asks:
approve or decline a paused action, or hand back after a takeover. Press Ctrl+C
once to take over (the run pauses: no screenshots, no model calls); press it again
to quit. With ``--request`` the run keeps going on the server after you quit, and
the task page can pick it up.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from typing import Any, Callable

_DEFAULT_API = "http://127.0.0.1:8080"
_EXIT = {"succeeded": 0, "failed": 1, "cancelled": 1}


# ── clients: one interface, in process or over HTTP ──────────────────────────


class LocalClient:
    """Runs the task in this process, through the same service the API uses."""

    def __init__(self) -> None:
        from operon.api import frontdoor

        self._fd = frontdoor

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        # No page watches an in-process run, so its browser is an ordinary batch one.
        return await self._fd.start_task(self._fd.FrontDoorTask(**payload), live_view=False)

    async def status(self, run_id: str) -> dict[str, Any]:
        return await self._fd.run_status(run_id)

    async def approve(self, run_id: str) -> dict[str, Any]:
        return await self._fd.approve(run_id)

    async def decline(self, run_id: str, note: str | None) -> dict[str, Any]:
        return await self._fd.decline(run_id, note)

    async def take_over(self, run_id: str) -> dict[str, Any]:
        return await self._fd.take_over(run_id)

    async def hand_back(self, run_id: str, note: str | None) -> dict[str, Any]:
        return await self._fd.hand_back(run_id, note)

    async def stop(self, run_id: str) -> dict[str, Any]:
        return await self._fd.stop(run_id)

    async def report(self, run_id: str) -> dict[str, Any]:
        return await self._fd.build_report(run_id)

    def format_report(self, report: dict[str, Any]) -> str:
        return self._fd.format_report(report)

    async def close(self) -> None:
        # Release the browser and desktop resources the run opened, as the server does at shutdown.
        from operon.api.runtime.loops import iter_built_executors

        for executor in iter_built_executors():
            aclose = getattr(executor, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - exiting anyway
                pass
        await settle_subprocess_pipes()


async def settle_subprocess_pipes(delay_s: float = 0.25) -> None:
    """Give Windows a moment to close Playwright's driver pipes before the event loop ends.

    On Windows the proactor loop closes a stopped subprocess's pipes a few ticks later.
    When ``asyncio.run`` closes the loop first, their ``__del__`` prints a scary
    "Exception ignored ... I/O operation on closed pipe" traceback after an otherwise
    clean run. Other platforms close them at once, so this is a no-op there.
    """
    if sys.platform == "win32":
        await asyncio.sleep(delay_s)


class RemoteClient:
    """Sends the task to a running Operon API (``--request``)."""

    def __init__(self, base_url: str, api_key: str | None, *, transport=None) -> None:
        import httpx

        headers = {"X-API-Key": api_key} if api_key else {}
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=30.0, transport=transport)

    async def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._http.request(method, path, json=body)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {detail}")
        return response.json()

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", "/app/api/runs", payload)

    async def status(self, run_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/app/api/runs/{run_id}")

    async def approve(self, run_id: str) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/approve")

    async def decline(self, run_id: str, note: str | None) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/decline", {"note": note})

    async def take_over(self, run_id: str) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/takeover")

    async def hand_back(self, run_id: str, note: str | None) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/handback", {"note": note})

    async def stop(self, run_id: str) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/stop")

    async def report(self, run_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/app/api/runs/{run_id}/report")

    def format_report(self, report: dict[str, Any]) -> str:
        from operon.api.frontdoor import format_report

        return format_report(report)

    async def close(self) -> None:
        await self._http.aclose()


# ── driving a run from the terminal ───────────────────────────────────────────


class _Interrupts:
    """Ctrl+C: the first asks for a takeover, the second quits."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, signum, frame) -> None:  # noqa: ARG002 - signal handler signature
        self.count += 1
        if self.count >= 2:
            raise KeyboardInterrupt


async def drive(
    client,
    payload: dict[str, Any],
    *,
    ask: Callable[[str], str],
    out: Callable[[str], None] = print,
    interactive: bool = True,
    poll_s: float = 1.0,
    interrupts: _Interrupts | None = None,
    leave_paused: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    """Start the task and follow it to the end, asking ``ask`` whenever it waits for you.

    Returns ``(exit_code, report)``: 0 succeeded, 1 failed or stopped, 2 left paused
    (a non-interactive terminal with ``leave_paused``, i.e. ``--request``)."""
    status = await client.start(payload)
    run_id = status["run_id"]
    out(f"Run {run_id} started: {status['environment']}, {status['permission_mode']} mode. Ctrl+C to take over.")
    printed = 0
    while True:
        for step in status.get("recent_steps", []):
            if step["step"] > printed:
                mark = "" if step["ok"] else "  (failed)"
                out(f"  {step['step']:>3}. {step['action']}{mark}")
                printed = step["step"]
        if status["done"]:
            break
        if interrupts is not None and interrupts.count == 1 and not status["in_takeover"]:
            interrupts.count = 0
            status = await client.take_over(run_id)
            continue
        waiting = status.get("waiting")
        if waiting is not None:
            if not interactive:
                if leave_paused:
                    out(f"Waiting for you: {waiting['reason']}")
                    out("Left paused: approve it on the task page, or run again in a terminal.")
                    page = getattr(client, "base_url", None)
                    if page:
                        out(f"Follow it on {page}/app#run={run_id}")
                    return 2, None
                out(f"Waiting for you, but there is no terminal to ask: {waiting['reason']}. Stopping.")
                status = await client.stop(run_id)
                continue
            status = await _answer(client, run_id, waiting, ask, out)
            continue
        await asyncio.sleep(poll_s)
        status = await client.status(run_id)
    report = await client.report(run_id)
    out("")
    out(client.format_report(report))
    return _EXIT.get(report["status"], 1), report


async def _answer(client, run_id: str, waiting: dict[str, Any], ask, out) -> dict[str, Any]:
    kind = waiting["kind"]
    if kind == "takeover":
        out("You are in control: Operon takes no screenshots and makes no model calls.")
        reply = (await asyncio.to_thread(ask, "Press Enter to hand back (or type a note for Operon, or 's' to stop): ")).strip()
        if reply.lower() in {"s", "stop"}:
            return await client.stop(run_id)
        return await client.hand_back(run_id, reply or None)
    # A gate's reason already says what it wants ("Approval needed ...", "Blocked by policy ...").
    out(waiting["reason"] if kind == "approval" else f"Operon needs you: {waiting['reason']}")
    reply = (await asyncio.to_thread(
        ask, "[y] approve / [n] decline / type what to do instead / [t] take over / [s] stop: "
    )).strip()
    low = reply.lower()
    if low in {"y", "yes", "", "ok", "continue"}:
        return await client.approve(run_id)
    if low in {"s", "stop"}:
        return await client.stop(run_id)
    if low in {"t", "take over", "takeover"}:
        return await client.take_over(run_id)
    if low in {"n", "no"}:
        return await client.decline(run_id, None)
    return await client.decline(run_id, reply)


# ── commands ─────────────────────────────────────────────────────────────────


def _load_env() -> None:
    from dotenv import find_dotenv, load_dotenv

    from operon.core.paths import load_user_env, remember_shell_env

    remember_shell_env()
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_user_env()  # the key `operon setup` saved, for a pip install with no .env here


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent": args.task,
        "environment": "desktop" if args.desktop else "browser",
        "max_steps": args.max_steps,
    }
    if args.url:
        payload["start_url"] = args.url
    if args.mode:
        payload["permission_mode"] = args.mode
    if args.headless:
        payload["headless"] = True
    return payload


async def _run(args: argparse.Namespace) -> int:
    if args.request:
        from operon.api.auth import client_api_key

        client = RemoteClient(args.api, args.api_key or client_api_key())
    else:
        from operon.api.runtime_config import validate_modes

        errors = validate_modes()
        if errors:
            print("Invalid Operon configuration:\n  - " + "\n  - ".join(errors), file=sys.stderr)
            return 1
        client = LocalClient()
    interrupts = _Interrupts()
    previous = signal.signal(signal.SIGINT, interrupts)
    try:
        code, _ = await drive(
            client,
            _payload(args),
            ask=input,
            interactive=sys.stdin.isatty(),
            interrupts=interrupts,
            leave_paused=bool(args.request),
        )
        return code
    except Exception as exc:  # noqa: BLE001 - a refused task or an unreachable API ends the command
        print(f"operon run: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
        await client.close()


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from operon.api.auth import auth_disabled, local_api_key

    keys = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
    key = keys[0] if keys else (None if auth_disabled() else local_api_key())
    url = f"http://{'127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host}:{args.port}/app"
    # The key goes in the fragment, which the browser never sends to the server.
    print(f"Task page: {url}#key={key}" if key else f"Task page: {url}  (authentication is off)")
    uvicorn.run("operon.api.server:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from operon import __version__

    parser = argparse.ArgumentParser(prog="operon", description="Operon, a vision-first computer-use agent.")
    parser.add_argument("--version", action="version", version=f"operon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task and print a report")
    run.add_argument("task", help="what Operon should do, in plain words")
    run.add_argument("--request", action="store_true", help="send the task to a running API instead of running it here")
    run.add_argument("--api", default=os.getenv("OPERON_API_URL", _DEFAULT_API), help=f"API address for --request (default $OPERON_API_URL or {_DEFAULT_API})")
    run.add_argument("--api-key", default=None, help="API key for --request (default: API_KEYS, else .var/api-key)")
    run.add_argument("--desktop", action="store_true", help="run on the desktop instead of in the browser")
    run.add_argument("--url", default=None, help="page to open first (browser)")
    run.add_argument("--mode", choices=["ask", "risky", "auto"], default=None, help="permission mode (default $OPERON_PERMISSION_MODE, else risky)")
    run.add_argument("--max-steps", type=int, default=25)
    run.add_argument("--headless", action="store_true", help="hide the browser window")
    run.set_defaults(func=lambda a: asyncio.run(_run(a)))

    serve = sub.add_parser("serve", help="start the API and print the task page's link")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=_serve)

    from operon import cloud, setup_command

    setup_command.add_parser(sub)
    cloud.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nQuit.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
