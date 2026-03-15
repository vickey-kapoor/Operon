"""
UI Navigator Test Dashboard Server
====================================
Serves the test dashboard and exposes the latest test report.

Usage:
    python tests/dashboard_server.py [--port 3333]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env from repo root into os.environ (stdlib only, no python-dotenv)
# ---------------------------------------------------------------------------
def _load_dotenv(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)  # don't overwrite existing env vars


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_load_dotenv(_REPO_ROOT)
_DASHBOARD_HTML = _REPO_ROOT / "tests" / "dashboard" / "index.html"
_REPORT_PATH = Path(tempfile.gettempdir()) / "wp_test_report.json"
_JUDGE_PATH = Path(tempfile.gettempdir()) / "wp_judge_output.json"
_RUNNER_LOG = Path(tempfile.gettempdir()) / "wp_runner.log"
_JUDGE_LOG = Path(tempfile.gettempdir()) / "wp_judge.log"
_RUNNER_TIMEOUT = 180  # seconds before watchdog kills a stuck runner

# ---------------------------------------------------------------------------
# Subprocess tracking (module-level so handler + signal share state)
# ---------------------------------------------------------------------------
_runner_proc: subprocess.Popen | None = None
_judge_proc: subprocess.Popen | None = None
_fix_loop_proc: subprocess.Popen | None = None
_runner_log_fh = None  # file handle for runner log
_fix_loop_log_fh = None  # file handle for fix-loop log
_last_run: str | None = None  # ISO timestamp of last /run call


def _is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _close_fh(fh) -> None:  # noqa: ANN001
    """Close a file handle if it's open, ignoring errors."""
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # CORS — allow any origin so browser pages can call freely
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode()
        self._send(status, "application/json", body)

    def do_OPTIONS(self) -> None:  # preflight
        self._send(204, "text/plain", b"")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/":
            self._serve_html()
        elif path == "/report":
            self._get_report()
        elif path == "/judge":
            self._get_judge()
        elif path == "/run":
            self._run_tests()
        elif path == "/run_judge":
            self._run_judge()
        elif path == "/run_fix_loop":
            self._run_fix_loop()
        elif path == "/status":
            self._get_status()
        else:
            self._json(404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Route implementations
    # ------------------------------------------------------------------

    def _serve_html(self) -> None:
        if not _DASHBOARD_HTML.exists():
            self._send(404, "text/plain", b"index.html not found")
            return
        body = _DASHBOARD_HTML.read_bytes()
        self._send(200, "text/html; charset=utf-8", body)

    def _get_report(self) -> None:
        if not _REPORT_PATH.exists():
            self._json(200, {"status": "no_report_yet"})
            return
        try:
            data = json.loads(_REPORT_PATH.read_text())
            self._json(200, data)
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def _get_judge(self) -> None:
        if not _JUDGE_PATH.exists():
            self._json(200, {"status": "no_judge_yet"})
            return
        try:
            data = json.loads(_JUDGE_PATH.read_text())
            self._json(200, data)
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def _run_tests(self) -> None:
        global _runner_proc, _runner_log_fh, _last_run
        if _is_running(_runner_proc):
            self._json(200, {"status": "already_running"})
            return
        _close_fh(_runner_log_fh)
        _runner_log_fh = open(_RUNNER_LOG, "w")
        _runner_proc = subprocess.Popen(
            [sys.executable, "tests/agent_runner.py", "--scenario", "all"],
            stdout=_runner_log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
        _last_run = datetime.now(timezone.utc).isoformat()
        self._json(200, {"status": "started"})

    def _run_judge(self) -> None:
        global _judge_proc
        if _is_running(_runner_proc):
            self._json(409, {"status": "runner_in_progress",
                             "error": "Wait for the test run to finish before judging"})
            return
        if not _REPORT_PATH.exists():
            self._json(200, {"status": "no_report_to_judge"})
            return
        if _is_running(_judge_proc):
            self._json(200, {"status": "already_running"})
            return
        _judge_proc = subprocess.Popen(
            [sys.executable, "tests/judge_runner.py",
             "--report", str(_REPORT_PATH)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(_REPO_ROOT),
        )
        self._json(200, {"status": "started"})

    def _run_fix_loop(self) -> None:
        """Run agent_runner then judge_runner as a chained subprocess."""
        global _fix_loop_proc, _fix_loop_log_fh, _last_run
        if _is_running(_fix_loop_proc):
            self._json(200, {"status": "already_running"})
            return
        # Chain: run tests, then run judge
        if sys.platform == "win32":
            cmd = f'"{sys.executable}" tests/agent_runner.py --scenario all && "{sys.executable}" tests/judge_runner.py'
            shell = True
        else:
            cmd = f'{sys.executable} tests/agent_runner.py --scenario all && {sys.executable} tests/judge_runner.py'
            shell = True
        _close_fh(_fix_loop_log_fh)
        _fix_loop_log_fh = open(_RUNNER_LOG, "w")
        _fix_loop_proc = subprocess.Popen(
            cmd,
            stdout=_fix_loop_log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
            shell=shell,
        )
        _last_run = datetime.now(timezone.utc).isoformat()
        self._json(200, {"status": "started"})

    def _get_status(self) -> None:
        global _runner_log_fh, _fix_loop_log_fh

        def _exit_code(proc: subprocess.Popen | None) -> int | None:
            if proc is None:
                return None  # never started
            rc = proc.poll()
            return rc  # None while running, int when finished

        # Close log file handles once the owning process has finished
        if not _is_running(_runner_proc) and _runner_log_fh is not None:
            _close_fh(_runner_log_fh)
            _runner_log_fh = None
        if not _is_running(_fix_loop_proc) and _fix_loop_log_fh is not None:
            _close_fh(_fix_loop_log_fh)
            _fix_loop_log_fh = None

        self._json(200, {
            "runner_running": _is_running(_runner_proc),
            "runner_exit_code": _exit_code(_runner_proc),
            "judge_running": _is_running(_judge_proc),
            "judge_exit_code": _exit_code(_judge_proc),
            "fix_loop_running": _is_running(_fix_loop_proc),
            "fix_loop_exit_code": _exit_code(_fix_loop_proc),
            "last_run": _last_run,
        })

    # ------------------------------------------------------------------
    # Silence default request logging to keep output clean
    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN001
        pass


# ---------------------------------------------------------------------------
# Signal handler — clean up child processes on Ctrl-C
# ---------------------------------------------------------------------------
def _shutdown(signum, frame) -> None:  # noqa: ANN001
    print("\nShutting down — terminating child processes…")
    for proc in (_runner_proc, _judge_proc, _fix_loop_proc):
        if _is_running(proc):
            proc.terminate()
    _close_fh(_runner_log_fh)
    _close_fh(_fix_loop_log_fh)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="UI Navigator test dashboard server")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _shutdown)
    # SIGTERM for clean container/process-manager shutdown (not available on Windows)
    if hasattr(signal, 'SIGTERM') and sys.platform != 'win32':
        signal.signal(signal.SIGTERM, _shutdown)

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    print("Open this URL in your browser")
    server.serve_forever()


if __name__ == "__main__":
    main()
