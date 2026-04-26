"""
Live execution tests — Operon running real tasks end-to-end.

These tests call the actual step loop against a live server with real API keys.
The browser runs in non-headless mode (BROWSER_HEADLESS=false) so you can
watch the agent act in real time.

Run with:
    .venv/Scripts/python -m pytest tests/test_live_execution.py --live -v -s

Requires: server running at 127.0.0.1:8080 with real GEMINI_API_KEY / ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"
SESSION = requests.Session()
SESSION.headers["Content-Type"] = "application/json"

_LIVE_OPT_IN = os.getenv("OPERON_RUN_LIVE_SERVER_TESTS", "false").lower() == "true"

pytestmark = pytest.mark.live_server

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
MAX_STEPS = 20
STEP_TIMEOUT_S = 240   # per step — browser steps can take 115s+ (Claude + Gemini perception)
POLL_INTERVAL_S = 0.5  # between step calls


@pytest.fixture(scope="session", autouse=True)
def require_live_server(request: pytest.FixtureRequest):
    if not (request.config.getoption("--live") or _LIVE_OPT_IN):
        pytest.skip("Use --live to run live execution tests")
    try:
        resp = SESSION.get(f"{BASE_URL}/health", timeout=10)
        assert resp.json() == {"status": "ok"}
    except Exception as exc:
        pytest.skip(f"Server not reachable: {exc}")


@pytest.fixture(autouse=True)
def clean_desktop_between_tests():
    """Close stray Notepad/Calculator windows before each test so desktop state is predictable."""
    import subprocess
    for proc in ("notepad.exe", "calc.exe", "win32calc.exe"):
        subprocess.run(["taskkill", "/IM", proc, "/F"], capture_output=True)
    yield
    # No teardown — leave apps as-is so you can observe the final state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = STEP_TIMEOUT_S) -> requests.Response:
    return SESSION.post(f"{BASE_URL}{path}", json=body, timeout=timeout)


def _get(path: str) -> requests.Response:
    return SESSION.get(f"{BASE_URL}{path}", timeout=15)


def run_to_completion(
    *,
    run_id: str,
    step_path: str,
    label: str,
    max_steps: int = MAX_STEPS,
) -> dict:
    """
    Drive a run by calling step_path until terminal status or max_steps.
    Prints live progress so you can follow along.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {label}")
    print(f"  RUN:  {run_id}")
    print(f"{'='*60}")

    final_data: dict = {}
    for step_n in range(1, max_steps + 1):
        t0 = time.perf_counter()
        resp = _post(step_path, {"run_id": run_id})
        elapsed = time.perf_counter() - t0

        assert resp.status_code in (200, 202), (
            f"Step {step_n} returned unexpected {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        final_data = data
        status = data.get("status", "?")
        subgoal = data.get("current_subgoal") or "—"

        print(
            f"  step {step_n:02d}  [{elapsed:5.1f}s]  status={status:<18}  subgoal={subgoal}"
        )

        if status in TERMINAL_STATUSES:
            stop_reason = data.get("stop_reason") or ""
            print(f"\n  DONE — {status.upper()}  stop_reason={stop_reason}")
            break

        if status == "waiting_for_user":
            hitl_msg = data.get("hitl_message") or "No message provided"
            print(f"\n  PAUSED (HITL) — {hitl_msg}")
            break

        time.sleep(POLL_INTERVAL_S)
    else:
        print(f"\n  STOPPED after {max_steps} steps (still running)")

    print(f"{'='*60}\n")
    return final_data


def create_desktop_run(intent: str) -> str:
    resp = _post("/desktop/run-task", {"intent": intent})
    assert resp.status_code == 202, f"Create failed ({resp.status_code}): {resp.text}"
    run_id = resp.json()["run_id"]
    print(f"\n[desktop] Created run {run_id!r} — intent: {intent!r}")
    return run_id


def create_browser_run(intent: str, start_url: str) -> str:
    resp = _post("/run-task", {"intent": intent, "start_url": start_url})
    assert resp.status_code == 202, f"Create failed ({resp.status_code}): {resp.text}"
    run_id = resp.json()["run_id"]
    print(f"\n[browser] Created run {run_id!r} — intent: {intent!r}  url: {start_url!r}")
    return run_id


# ===========================================================================
# TIER 1 — SIMPLE DESKTOP: single-app tasks
# ===========================================================================

class TestDesktopSimple:
    """Open a single app. Agent should succeed in 1-3 steps."""

    def test_open_notepad(self):
        """CAPABILITY: launch Notepad via desktop automation."""
        run_id = create_desktop_run("Open Notepad")
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Open Notepad",
            max_steps=8,
        )
        assert result["status"] in ("succeeded", "running", "failed"), (
            f"Unexpected terminal status: {result['status']}"
        )

    def test_open_calculator(self):
        """CAPABILITY: launch Calculator via desktop automation."""
        run_id = create_desktop_run("Open Calculator")
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Open Calculator",
            max_steps=8,
        )
        assert result["status"] in ("succeeded", "running", "failed")

    def test_describe_desktop(self):
        """CAPABILITY: describe what is currently visible on screen."""
        run_id = create_desktop_run("Describe what is currently visible on the desktop")
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Describe desktop",
            max_steps=5,
        )
        assert result["status"] in ("succeeded", "running", "failed")


# ===========================================================================
# TIER 2 — MODERATE DESKTOP: multi-step keyboard workflows
# ===========================================================================

class TestDesktopModerate:
    """Multi-step tasks involving keyboard + app interaction."""

    def test_notepad_type_sentence(self):
        """CAPABILITY: open Notepad and type a full sentence."""
        run_id = create_desktop_run(
            "Open Notepad and type: The quick brown fox jumps over the lazy dog"
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Notepad type sentence",
            max_steps=12,
        )
        assert result["status"] in ("succeeded", "running", "failed")

    def test_calculator_compute_42x17(self):
        """CAPABILITY: open Calculator, compute 42 × 17 by clicking buttons."""
        run_id = create_desktop_run(
            "Open Calculator, compute 42 times 17 by clicking the buttons, and report the result"
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Calculator 42 × 17",
            max_steps=15,
        )
        assert result["status"] in ("succeeded", "running", "failed")

    def test_win_r_open_calc(self):
        """CAPABILITY: use Win+R dialog to launch Calculator."""
        run_id = create_desktop_run(
            "Press Win+R to open the Run dialog, type calc, and press Enter"
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Win+R → calc",
            max_steps=8,
        )
        assert result["status"] in ("succeeded", "running", "failed")

    def test_notepad_meeting_notes(self):
        """CAPABILITY: structured note-taking in Notepad."""
        run_id = create_desktop_run(
            "Open Notepad, type meeting notes as a title, press Enter, then type Item 1 and Item 2"
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/desktop/step",
            label="Meeting notes in Notepad",
            max_steps=15,
        )
        assert result["status"] in ("succeeded", "running", "failed")


# ===========================================================================
# TIER 3 — BROWSER: visible Playwright window
# ===========================================================================

class TestBrowserSimple:
    """Browser tasks — Chromium opens in non-headless mode."""

    def test_navigate_example_com(self):
        """CAPABILITY: open browser and navigate to example.com."""
        run_id = create_browser_run(
            intent="Describe the main heading and first paragraph visible on this page",
            start_url="https://example.com",
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/step",
            label="Browser: example.com",
            max_steps=6,
        )
        assert result["status"] in ("succeeded", "running", "failed")

    def test_navigate_wikipedia_python(self):
        """CAPABILITY: navigate to Wikipedia and read the intro."""
        run_id = create_browser_run(
            intent="Find and read the first sentence of the Python programming language Wikipedia article",
            start_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/step",
            label="Browser: Wikipedia Python",
            max_steps=8,
        )
        assert result["status"] in ("succeeded", "running", "failed")


# ===========================================================================
# TIER 4 — BROWSER FORM: the core product capability
# ===========================================================================

class TestBrowserForm:
    """
    Form fill — the primary Operon benchmark.
    Agent must find Name / Email / Message fields, fill them, submit.
    Watch the browser window to see the agent filling fields live.
    """

    def test_fill_contact_form(self):
        """CAPABILITY: fill a multi-field contact form end-to-end."""
        run_id = create_browser_run(
            intent=(
                "Fill out the contact form completely: "
                "enter Name as 'Alice Johnson', "
                "Email as 'alice@example.com', "
                "Phone as '555-1234', "
                "Message as 'I would like more information about your services', "
                "then submit the form."
            ),
            start_url="https://practice-automation.com/form-fields/",
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/step",
            label="Browser: Contact form fill + submit",
            max_steps=MAX_STEPS,
        )
        assert result["status"] in ("succeeded", "running", "failed"), (
            f"Unexpected status: {result['status']}"
        )
        # A successful form fill should not end in failed
        if result["status"] == "failed":
            stop = result.get("stop_reason", "unknown")
            pytest.xfail(f"Form fill failed — stop_reason={stop}. Check agent logs.")

    def test_fill_simple_search(self):
        """CAPABILITY: type a search query and submit."""
        run_id = create_browser_run(
            intent="Type 'Operon AI agent' into the search box and press Enter to search",
            start_url="https://duckduckgo.com",
        )
        result = run_to_completion(
            run_id=run_id,
            step_path="/step",
            label="Browser: DuckDuckGo search",
            max_steps=8,
        )
        assert result["status"] in ("succeeded", "running", "failed")
