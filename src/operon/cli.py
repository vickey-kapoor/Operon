"""The ``operon`` command (KAN-89).

    operon run "task"               run a task here, in this process, and print a report
    operon run --request "task"     send the task to a running API and follow it
    operon serve                    start the API
    operon cloud ...                Operon's own computer on an Azure Linux VM (see operon.cloud)

Both ``run`` forms use the same front-door service as the API's /app/api routes
(``operon.api.frontdoor``), so permission modes, approvals, takeover and the report
behave the same everywhere. While a run waits for you, the terminal asks:
approve or decline a paused action, or hand back after a takeover. Press Ctrl+C
once to take over (the run pauses: no screenshots, no model calls); press it again
to quit. With ``--request`` the run keeps going on the server after you quit, and
the run can be followed through the API.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

from operon.core.settings import get_settings

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

    async def resume_interrupted(self, run_id: str) -> dict[str, Any]:
        return await self._fd.resume_interrupted(run_id)

    async def interrupted(self) -> list[dict[str, Any]]:
        return self._fd.interrupted_runs()

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
        from operon.api.runtime.loops import close_built_runtime

        await close_built_runtime()
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

    async def resume_interrupted(self, run_id: str) -> dict[str, Any]:
        return await self._call("POST", f"/app/api/runs/{run_id}/resume-interrupted")

    async def interrupted(self) -> list[dict[str, Any]]:
        return (await self._call("GET", "/app/api/interrupted"))["runs"]

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


_MAX_AUTO_CONFIRMS = 10  # per run, so a model that keeps asking cannot loop forever


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
    resume_run_id: str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Start the task and follow it to the end, asking ``ask`` whenever it waits for you.

    Returns ``(exit_code, report)``: 0 succeeded, 1 failed or stopped, 2 left paused
    (a non-interactive terminal with ``leave_paused``, i.e. ``--request``)."""
    if resume_run_id is not None:
        status = await client.resume_interrupted(resume_run_id)
        run_id = status["run_id"]
        out(f"Run {run_id} resumed from step {status['step_count']}: {status['environment']}, "
            f"{status['permission_mode']} mode. Ctrl+C to take over.")
    else:
        status = await client.start(payload)
        run_id = status["run_id"]
        out(f"Run {run_id} started: {status['environment']}, {status['permission_mode']} mode. Ctrl+C to take over.")
    printed = 0
    auto_confirmed = 0
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
        if (waiting is not None and waiting["kind"] == "needs_you" and auto_confirmed < _MAX_AUTO_CONFIRMS
                and get_settings().auto_confirm):
            # OPERON_AUTO_CONFIRM (benchmark runs, nobody can click Resume): the model's own
            # "require confirmation" pause is answered yes. Gates ("approval") are never answered here.
            auto_confirmed += 1
            out(f"Auto-confirming (OPERON_AUTO_CONFIRM): {waiting['reason']}")
            status = await client.approve(run_id)
            continue
        if waiting is not None:
            if not interactive:
                if leave_paused:
                    out(f"Waiting for you: {waiting['reason']}")
                    out("Left paused: approve it through the API, or run again in a terminal.")
                    page = getattr(client, "base_url", None)
                    if page:
                        out(f"Run {run_id} is on {page}")
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

    from operon.core.paths import load_stored_keys, load_user_env, real_environment

    real = real_environment()
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_user_env()  # the optional per-user key file (~/.operon/operon.env), for a run with no .env here
    load_stored_keys(real)  # KAN-186: OS credential store beats both files, the real environment beats it


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent": args.task,
        "environment": "desktop" if args.desktop else "browser",
        "max_steps": args.max_steps if args.max_steps is not None else 25,
    }
    if args.url:
        payload["start_url"] = args.url
    if args.mode:
        payload["permission_mode"] = args.mode
    if args.headless:
        payload["headless"] = True
    return payload


def _api_answers(base_url: str) -> bool:
    """Whether an Operon API already answers at ``base_url``: its /health, or a key check on it."""
    import httpx

    try:
        return httpx.get(f"{base_url.rstrip('/')}/health", timeout=1.5).status_code in {200, 401, 403}
    except httpx.HTTPError:
        return False


def _checkpoint_client(args: argparse.Namespace, *, resume_id: str | None = None):
    """A client for the resume command (and `runs --request`), or None after printing why not.

    Locally only the explicit `resume <id>` marks anything, and only that run, and only when
    its owner process is gone; `runs` never writes."""
    if args.request:
        from operon.api.auth import client_api_key

        try:
            key = _resolve_api_key(args)
        except ValueError as exc:
            print(f"operon: {exc}", file=sys.stderr)
            return None
        return RemoteClient(args.api, key or client_api_key())
    if not get_settings().checkpoints:
        print("operon: checkpoints are off; set OPERON_CHECKPOINTS=on (and restart `operon serve`).", file=sys.stderr)
        return None
    if resume_id is not None:
        api = get_settings().api_url or _DEFAULT_API
        if _api_answers(api):
            print(f"operon resume: an Operon API is running at {api}; resuming here would put two processes on "
                  f"one run. Use `operon resume {resume_id} --request` instead.", file=sys.stderr)
            return None
        from operon.core.paths import runs_dir
        from operon.store.checkpoints import mark_interrupted_runs

        # Legacy runs (no owner file) count as dead after two minutes without a save.
        mark_interrupted_runs(runs_dir(), min_age_s=120.0, only=resume_id)
    return LocalClient()


async def _runs(args: argparse.Namespace) -> int:
    if args.request:
        client = _checkpoint_client(args)
        if client is None:
            return 2
        try:
            runs = await client.interrupted()
        except Exception as exc:  # noqa: BLE001 - an unreachable API ends the command
            print(f"operon runs: {exc}", file=sys.stderr)
            return 1
        finally:
            await client.close()
    else:
        if not get_settings().checkpoints:
            print("operon: checkpoints are off; set OPERON_CHECKPOINTS=on (and restart `operon serve`).", file=sys.stderr)
            return 1
        from operon.core.paths import runs_dir
        from operon.store.checkpoints import find_interrupted

        # Read-only: what a scan would flag is worked out in memory, nothing is written.
        runs = [
            {"run_id": s.run_id, "intent": s.intent, "environment": s.environment or "browser", "step_count": s.step_count}
            for s in find_interrupted(runs_dir(), min_age_s=120.0)
        ]
    if not runs:
        print("No interrupted runs.")
        return 0
    for run in runs:
        print(f"{run['run_id']}  {run['environment']:<7} step {run['step_count']:<3} {run['intent'][:70]}")
    print("Resume one with: operon resume <run_id>")
    return 0


async def _resume(args: argparse.Namespace) -> int:
    client = _checkpoint_client(args, resume_id=args.run_id)
    if client is None:
        return 2 if args.request else 1  # a bad --api-key-file is a usage error, as in `run` and `runs`
    if not args.request:
        from operon.core.settings import startup_problems

        errors, _warnings = startup_problems()
        if errors:
            print("Invalid Operon configuration:\n  - " + "\n  - ".join(errors), file=sys.stderr)
            return 1
    interrupts = _Interrupts()
    previous = signal.signal(signal.SIGINT, interrupts)
    try:
        code, _ = await drive(
            client, {}, ask=input, interactive=sys.stdin.isatty(), interrupts=interrupts,
            leave_paused=bool(args.request), resume_run_id=args.run_id,
        )
        return code
    except Exception as exc:  # noqa: BLE001 - a run that cannot resume ends the command
        print(f"operon resume: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
        await client.close()


async def _run(args: argparse.Namespace) -> int:
    if getattr(args, "env", None):
        if args.request:
            print("operon run: --env picks this machine's environment; the API you --request uses its own "
                  "OPERON_ENVIRONMENT.", file=sys.stderr)
            return 2
        os.environ["OPERON_ENVIRONMENT"] = args.env
    if args.request:
        from operon.api.auth import client_api_key

        try:
            key = _resolve_api_key(args)
        except ValueError as exc:
            print(f"operon run: {exc}", file=sys.stderr)
            return 2
        client = RemoteClient(args.api or get_settings().api_url or _DEFAULT_API, key or client_api_key())
    else:
        from operon.core.settings import startup_problems

        errors, warnings = startup_problems()
        for warning in warnings:
            print(f"operon run: setting ignored: {warning}", file=sys.stderr)
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


def _run_entry(args: argparse.Namespace) -> int:
    if not args.task:
        print("operon run: give a task", file=sys.stderr)
        return 2
    return asyncio.run(_run(args))


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def _display_host(host: str) -> str:
    host = host.strip().strip("[]")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host or "127.0.0.1"
    if address.is_unspecified:
        return "127.0.0.1"
    return f"[{host}]" if address.version == 6 else host


def _serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn
    from dotenv import find_dotenv, load_dotenv

    from operon.api.auth import api_key_file, auth_disabled, local_api_key
    from operon.core.paths import load_stored_keys, load_user_env, real_environment

    # The server's create_app() reloads .env with override=True; do the same first, so the checks
    # below judge the configuration the server will actually run with.
    real = real_environment()
    load_dotenv(find_dotenv(usecwd=True), override=True)
    load_user_env()  # as create_app() does next; never overrides
    load_stored_keys(real)  # KAN-186, as create_app() does
    if not _is_loopback(args.host):
        if not args.allow_remote:
            print(
                f"operon serve: refusing to listen on {args.host}; this API can drive your computer. "
                "Pass --allow-remote to allow it.",
                file=sys.stderr,
            )
            return 2
        if auth_disabled() and not get_settings().api_keys:
            print(
                f"operon serve: refusing to listen on {args.host} with authentication off; "
                "set API_KEYS or unset OPERON_ALLOW_UNAUTHENTICATED.",
                file=sys.stderr,
            )
            return 2
        print(f"operon serve: warning: listening on {args.host}; anyone who can reach this port and has the API key can drive this computer.", file=sys.stderr)
    print(f"API: http://{_display_host(args.host)}:{args.port}")
    if get_settings().api_keys:
        print("API key: from API_KEYS")
    elif not auth_disabled():
        local_api_key()  # creates the file
        print(f"API key file: {api_key_file()}")
    else:
        print("Authentication is off")
    level = args.log_level.upper()
    logging.basicConfig(level=level)
    logging.getLogger("operon").setLevel(level)
    uvicorn.run("operon.api.server:app", host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    """--api-key wins; else the contents of --api-key-file; raises ValueError when that file is unusable."""
    if args.api_key or not getattr(args, "api_key_file", None):
        return args.api_key
    try:
        key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read --api-key-file {args.api_key_file}: {exc.strerror or exc}") from exc
    if not key:
        raise ValueError(f"--api-key-file {args.api_key_file} is empty")
    return key


def build_parser() -> argparse.ArgumentParser:
    from operon import __version__

    parser = argparse.ArgumentParser(prog="operon", description="Operon, a vision-first computer-use agent.")
    parser.add_argument("--version", action="version", version=f"operon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task and print a report")
    run.add_argument("task", nargs="?", default=None, help="what Operon should do, in plain words")
    run.add_argument("--request", action="store_true", help="send the task to a running API instead of running it here")
    run.add_argument("--api", default=None, help=f"API address for --request (default $OPERON_API_URL or {_DEFAULT_API})")
    run.add_argument("--api-key", default=None, help="API key for --request (default: API_KEYS, else .var/api-key)")
    run.add_argument("--api-key-file", default=None, help="file holding the API key for --request (used when --api-key is not given)")
    run.add_argument("--desktop", action="store_true", help="run on the desktop instead of in the browser")
    run.add_argument("--env", default=None, help="desktop run environment (default $OPERON_ENVIRONMENT, else live; see `operon envs`)")
    run.add_argument("--url", default=None, help="page to open first (browser)")
    run.add_argument("--mode", choices=["ask", "risky", "auto"], default=None, help="permission mode (default $OPERON_PERMISSION_MODE, else risky)")
    run.add_argument("--max-steps", type=int, default=None, help="step limit (default 25)")
    run.add_argument("--headless", action="store_true", help="hide the browser window")
    run.set_defaults(func=_run_entry)

    for name, text in (("runs", "list runs a restart interrupted (OPERON_CHECKPOINTS)"),
                       ("resume", "continue an interrupted run from the current screen (OPERON_CHECKPOINTS)")):
        cp = sub.add_parser(name, help=text)
        if name == "runs":
            cp.add_argument("--interrupted", action="store_true", required=True, help="list the interrupted runs")
        else:
            cp.add_argument("run_id", help="the run to continue (see `operon runs --interrupted`)")
        cp.add_argument("--request", action="store_true", help="ask a running API instead of this process")
        cp.add_argument("--api", default=get_settings().api_url or _DEFAULT_API, help=f"API address for --request (default $OPERON_API_URL or {_DEFAULT_API})")
        cp.add_argument("--api-key", default=None, help="API key for --request (default: API_KEYS, else .var/api-key)")
        cp.add_argument("--api-key-file", default=None, help="file holding the API key for --request (used when --api-key is not given)")
        cp.set_defaults(func=(lambda a: asyncio.run(_runs(a))) if name == "runs" else (lambda a: asyncio.run(_resume(a))))

    serve = sub.add_parser("serve", help="start the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--log-level", choices=("debug", "info", "warning", "error", "critical"), default="info")
    serve.add_argument("--allow-remote", action="store_true", help="allow a non-loopback --host; this exposes the API beyond this machine")
    serve.set_defaults(func=_serve)

    from operon import cloud, keys_command, secrets_command
    from operon.environments import cli as environments_cli
    from operon.environments import guest

    secrets_command.add_parser(sub)
    keys_command.add_parser(sub)
    cloud.add_parser(sub)
    environments_cli.add_parser(sub)
    guest.add_parser(sub)
    return parser


def quiet_closed_pipe_noise() -> None:
    """Drop the "I/O operation on closed pipe" noise Windows prints at exit after a clean run.

    Playwright's driver pipes can outlive the event loop on Windows; when Python later
    garbage-collects their transports, ``__del__`` tries to print the closed pipe and
    raises, and Python reports it as "Exception ignored ...". Nothing is lost, so only
    that one message is dropped; every other unraisable error still prints.
    """
    previous = sys.unraisablehook

    def hook(unraisable) -> None:
        exc = unraisable.exc_value
        if isinstance(exc, ValueError) and "closed pipe" in str(exc):
            return
        previous(unraisable)

    sys.unraisablehook = hook


def _console_logging(command: str) -> None:
    """Keep library log lines out of `operon run` output: errors only, unless logging is set up."""
    import logging

    if command != "serve" and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    _load_env()
    if sys.platform == "win32":
        quiet_closed_pipe_noise()
    args = build_parser().parse_args(argv)
    _console_logging(args.command)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nQuit.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
