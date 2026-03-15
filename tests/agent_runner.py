"""
WebPilot E2E Test Orchestration Script
=======================================
Intended to be executed by a separate Claude agent (not by humans directly).

Usage:
    python tests/agent_runner.py --scenario <name>

Scenarios:
    session_lifecycle     test_session_lifecycle
    task_flow             test_task_navigate_and_done
    confirm_flow          test_confirm_flow + test_confirm_denied
    interrupt             test_interrupt_redirect
    stuck_detection       test_stuck_detection
    all                   all of the above

Output:
    JSON report to stdout with schema:
    {
      "scenario": str,
      "passed": bool,
      "tests": [{"name": str, "status": "passed"|"failed"|"error", "message": str}],
      "ws_message_log": [...],
      "suggestions": []
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Scenario → pytest -k expression mapping
# ---------------------------------------------------------------------------
_SCENARIO_FILTERS: dict[str, str] = {
    "session_lifecycle": "test_session_lifecycle",
    "task_flow": "test_task_navigate_and_done",
    "confirm_flow": "test_confirm_flow or test_confirm_denied",
    "interrupt": "test_interrupt_redirect",
    "stuck_detection": "test_stuck_detection",
    "all": "test_webpilot_e2e",
}

# Each scenario test requires a specific stub. For 'all', the e2e test file
# manages its own per-scenario servers internally, so WEBPILOT_STUB is set
# to "navigate_and_done" here only to satisfy the lifespan guard in server.py;
# the individual test fixtures override it for each scenario.
_SCENARIO_STUB: dict[str, str] = {
    "session_lifecycle": "navigate_and_done",
    "task_flow": "navigate_and_done",
    "confirm_flow": "confirm_flow",
    "interrupt": "interrupt_redirect",
    "stuck_detection": "stuck_loop",
    "all": "navigate_and_done",
}

_MSG_LOG_PATH = Path(tempfile.gettempdir()) / "wp_test_messages.json"


# Regex to extract per-test duration from pytest -v output
# Matches patterns like "PASSED (0.42s)" or "FAILED (1.23s)"
_DURATION_RE = re.compile(r"\b(?:PASSED|FAILED|ERROR)\s*\((\d+(?:\.\d+)?)s\)")


def _parse_pytest_output(stdout: str, stderr: str) -> list[dict]:
    """Parse pytest -v output into a list of {name, status, message, duration_s} dicts."""
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        # Only match actual test result lines (contain "::" path separator),
        # skip summary lines like "= 5 passed, 2 failed =" or "FAILED tests/..."
        if "::" not in line:
            continue
        for status_token, status_value in [("PASSED", "passed"), ("FAILED", "failed"), ("ERROR", "error")]:
            if status_token in line:
                # Extract test name from line like "tests/test_webpilot_e2e.py::test_foo PASSED"
                parts = line.split("::")
                test_name = parts[-1].split()[0] if len(parts) >= 2 else line
                message = ""
                if status_value in ("failed", "error"):
                    # pytest writes tracebacks to stdout (with capture_output=True),
                    # so search both stdout and stderr for failure details.
                    message = _extract_failure_message(stdout, test_name)
                    if not message:
                        message = _extract_failure_message(stderr, test_name)
                # Extract per-test duration if present (e.g. "PASSED (0.42s)")
                duration_s = None
                dur_match = _DURATION_RE.search(line)
                if dur_match:
                    duration_s = float(dur_match.group(1))
                results.append({
                    "name": test_name,
                    "status": status_value,
                    "message": message,
                    "duration_s": duration_s,
                })
                break
    return results


def _extract_failure_message(output: str, test_name: str) -> str:
    """Extract the failure message for a given test from pytest output."""
    lines = output.splitlines()
    capture = False
    captured: list[str] = []
    for line in lines:
        if test_name in line and ("FAILED" in line or "ERROR" in line):
            capture = True
        if capture:
            captured.append(line)
            if len(captured) > 20:
                break
    return "\n".join(captured)[:500]


def _read_ws_message_log() -> list[dict]:
    """Read the WS message log written by the test suite."""
    if not _MSG_LOG_PATH.exists():
        return []
    try:
        return json.loads(_MSG_LOG_PATH.read_text())
    except Exception:
        return []


def _clear_ws_message_log() -> None:
    if _MSG_LOG_PATH.exists():
        _MSG_LOG_PATH.unlink()


def run_scenario(scenario: str) -> dict:
    if scenario not in _SCENARIO_FILTERS:
        return {
            "scenario": scenario,
            "passed": False,
            "tests": [{"name": scenario, "status": "error",
                        "message": f"Unknown scenario. Available: {sorted(_SCENARIO_FILTERS)}"}],
            "ws_message_log": [],
            "suggestions": [],
        }

    _clear_ws_message_log()
    started_at = time.time()

    k_filter = _SCENARIO_FILTERS[scenario]
    stub = _SCENARIO_STUB[scenario]

    repo_root = Path(__file__).parent.parent
    env = os.environ.copy()
    # The e2e test file manages its own per-scenario server fixtures, so
    # WEBPILOT_STUB here is only a fallback (some tests may use it directly).
    env["WEBPILOT_STUB"] = stub
    # Ensure no real Gemini key is required
    env.setdefault("GOOGLE_API_KEY", "stub")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_webpilot_e2e.py",
        "-v", "--tb=short",
        "-k", k_filter,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    try:
        stdout_data, stderr_data = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # Kill the process tree — proc.kill() only kills the root on Windows
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=10,
                )
            else:
                proc.kill()
        except Exception:
            pass
        stdout_data, stderr_data = proc.communicate()
        finished_at = time.time()
        ws_log = _read_ws_message_log()
        tests = _parse_pytest_output(stdout_data, stderr_data)
        tests.append({
            "name": scenario,
            "status": "error",
            "message": "pytest timed out after 120s",
            "duration_s": None,
        })
        return {
            "scenario": scenario,
            "passed": False,
            "tests": tests,
            "ws_message_log": ws_log,
            "suggestions": [],
            "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
            "finished_at": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
            "duration_s": round(finished_at - started_at, 1),
        }

    finished_at = time.time()
    tests = _parse_pytest_output(stdout_data, stderr_data)
    ws_log = _read_ws_message_log()
    passed = proc.returncode == 0

    return {
        "scenario": scenario,
        "passed": passed,
        "tests": tests,
        "ws_message_log": ws_log,
        "suggestions": [],  # filled by the judge agent
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at": datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(),
        "duration_s": round(finished_at - started_at, 1),
    }


def write_report(report: dict) -> None:
    """Write the report JSON to /tmp/wp_test_report.json.

    If /tmp is not writable, logs a warning and continues — the caller is
    responsible for the primary stdout output.
    """
    report_path = Path(tempfile.gettempdir()) / "wp_test_report.json"
    try:
        report_path.write_text(json.dumps(report, indent=2))
    except OSError as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not write report to %s: %s", report_path, exc
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="WebPilot E2E test orchestration script")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(_SCENARIO_FILTERS),
        help="Scenario to run (or 'all' for everything)",
    )
    args = parser.parse_args()

    report = run_scenario(args.scenario)
    print(json.dumps(report, indent=2))
    write_report(report)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
