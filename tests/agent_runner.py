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
import subprocess
import sys
import tempfile
import time
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


def _parse_pytest_output(stdout: str, stderr: str) -> list[dict]:
    """Parse pytest -v output into a list of {name, status, message} dicts."""
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        for status_token, status_value in [("PASSED", "passed"), ("FAILED", "failed"), ("ERROR", "error")]:
            if status_token in line:
                # Extract test name from line like "tests/test_webpilot_e2e.py::test_foo PASSED"
                parts = line.split("::")
                test_name = parts[-1].split()[0] if len(parts) >= 2 else line
                message = ""
                if status_value in ("failed", "error"):
                    # Include context lines from stderr for error messages
                    message = _extract_failure_message(stderr, test_name)
                results.append({"name": test_name, "status": status_value, "message": message})
                break
    return results


def _extract_failure_message(stderr: str, test_name: str) -> str:
    """Extract the failure message from pytest stderr for a given test."""
    lines = stderr.splitlines()
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

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
        timeout=120,
    )

    tests = _parse_pytest_output(result.stdout, result.stderr)
    ws_log = _read_ws_message_log()
    passed = result.returncode == 0

    return {
        "scenario": scenario,
        "passed": passed,
        "tests": tests,
        "ws_message_log": ws_log,
        "suggestions": [],  # filled by the judge agent
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
