"""Run one visible agent task slowly while the local observer UI tracks each step."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib import error, request

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.benchmark import _build_loop
from src.models.common import RunStatus, RunTaskRequest, StepRequest


DEFAULT_INTENT = "Go to the Wikipedia article for Markov chain."
DEFAULT_START_URL = "https://www.wikipedia.org/"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a visible, step-delayed task while Operon Observer shows each step."
    )
    parser.add_argument("--intent", default=os.getenv("OPERON_INTENT", DEFAULT_INTENT))
    parser.add_argument("--start-url", default=os.getenv("OPERON_START_URL", DEFAULT_START_URL))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--step-delay-seconds", type=float, default=2.0)
    parser.add_argument("--initial-pause-seconds", type=float, default=5.0)
    parser.add_argument("--hold-open-seconds", type=float, default=15.0)
    parser.add_argument("--slow-mo-ms", type=int, default=350)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--no-open-observer", action="store_true")
    parser.add_argument("--reuse-server", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return ROOT


def configure_local_environment(root: Path) -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    temp_dir = root / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))


def configure_debug_timing(slow_mo_ms: int) -> None:
    os.environ["BROWSER_SLOW_MO_MS"] = str(max(0, slow_mo_ms))


def gemini_key_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def server_healthy(base_url: str) -> bool:
    try:
        with request.urlopen(f"{base_url}/health", timeout=1.5) as response:
            return response.status == 200
    except (error.URLError, TimeoutError):
        return False


def wait_for_server(base_url: str, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server_healthy(base_url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Observer server at {base_url} did not become healthy within {timeout_seconds:.1f}s.")


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def start_observer_server(host: str, port: int, root: Path) -> subprocess.Popen[str] | None:
    base_url = f"http://{host}:{port}"
    if server_healthy(base_url):
        return None
    if port_open(host, port):
        raise RuntimeError(f"Port {port} is already in use, but /health did not respond successfully.")

    python_exe = root / ".venv311" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise RuntimeError(f"Python runtime not found: {python_exe}")

    return subprocess.Popen(
        [
            str(python_exe),
            "-m",
            "uvicorn",
            "src.api.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(root),
    )


async def run_task(args: argparse.Namespace, root: Path) -> int:
    configure_local_environment(root)
    configure_debug_timing(args.slow_mo_ms)

    if not gemini_key_available():
        print("GEMINI_API_KEY is not configured. Set it in the environment or .env before running this script.", file=sys.stderr)
        return 2

    server_process: subprocess.Popen[str] | None = None
    base_url = f"http://{args.host}:{args.port}"

    if not args.reuse_server:
        server_process = start_observer_server(args.host, args.port, root)
    wait_for_server(base_url)

    if not args.no_open_observer:
        webbrowser.open(f"{base_url}/observer")

    loop, executor = _build_loop(root_dir=args.runs_root)
    try:
        response = await loop.start_run(RunTaskRequest(intent=args.intent, start_url=args.start_url))

        print(f"Observer: {base_url}/observer")
        print(f"Run ID: {response.run_id}")
        print(f"Intent: {args.intent}")
        print(f"Start URL: {args.start_url}")
        if args.initial_pause_seconds > 0:
            print(f"Waiting {args.initial_pause_seconds:.1f}s before the first step so the browser is visible...")
            await asyncio.sleep(args.initial_pause_seconds)

        while True:
            state = await loop.run_store.get_run(response.run_id)
            if state is None:
                raise RuntimeError(f"Run {response.run_id} was not found after start.")

            if state.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
                print(f"Terminal status: {state.status.value} stop_reason={state.stop_reason}")
                if args.hold_open_seconds > 0:
                    print(f"Holding browser open for {args.hold_open_seconds:.1f}s...")
                    await asyncio.sleep(args.hold_open_seconds)
                return 0

            if state.step_count >= args.max_steps:
                print(f"Stopping after {args.max_steps} steps without terminal success.")
                if args.hold_open_seconds > 0:
                    print(f"Holding browser open for {args.hold_open_seconds:.1f}s...")
                    await asyncio.sleep(args.hold_open_seconds)
                return 1

            next_step = state.step_count + 1
            print(f"Advancing step {next_step}...")
            response = await loop.step_run(StepRequest(run_id=response.run_id))
            state = await loop.run_store.get_run(response.run_id)
            print(f"Status after step {state.step_count}: {state.status.value}")
            await asyncio.sleep(max(0.0, args.step_delay_seconds))
    finally:
        await executor.close()
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


def main() -> int:
    args = parse_args()
    root = repo_root()
    return asyncio.run(run_task(args, root))


if __name__ == "__main__":
    raise SystemExit(main())
