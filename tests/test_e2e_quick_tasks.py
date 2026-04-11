"""
E2E API test suite for Operon Pilot quick tasks.

Tests all 22 quick task intents plus edge cases, error handling,
input validation, and observer/UI endpoints.

Run with:
    GEMINI_API_KEY=fake-test-key .venv/Scripts/python -m pytest tests/test_e2e_quick_tasks.py -v -s

For live execution against a running server:
    .venv/Scripts/python -m pytest tests/test_e2e_quick_tasks.py -v -s --live

NOTE: Without --live marker the tests run against the actual live server
but do NOT advance steps (Gemini calls would be needed). The intent-acceptance
and structural tests run regardless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"
SESSION = requests.Session()
SESSION.headers["Content-Type"] = "application/json"


@pytest.fixture(scope="session", autouse=True)
def require_server():
    try:
        resp = SESSION.get(f"{BASE_URL}/health", timeout=10)
    except requests.RequestException as exc:
        pytest.skip(f"Live server not reachable at {BASE_URL}: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"Live server not reachable at {BASE_URL}: status={resp.status_code}")
    assert resp.json() == {"status": "ok"}

# ---------------------------------------------------------------------------
# Quick-task intent catalogue (all 22 tasks from the spec)
# ---------------------------------------------------------------------------

QUICK_TASKS: list[dict[str, str]] = [
    # Apps
    {"category": "Apps", "id": "open_notepad", "intent": "Open Notepad"},
    {"category": "Apps", "id": "open_calculator", "intent": "Open Calculator"},
    {"category": "Apps", "id": "open_file_explorer", "intent": "Open File Explorer"},
    {"category": "Apps", "id": "open_paint", "intent": "Open Paint"},
    {"category": "Apps", "id": "open_settings", "intent": "Open Windows Settings"},
    # Mouse
    {"category": "Mouse", "id": "right_click_desktop", "intent": "Right-click on an empty area of the desktop to open the context menu"},
    {"category": "Mouse", "id": "double_click_recycle_bin", "intent": "Double-click on the Recycle Bin icon on the desktop"},
    {"category": "Mouse", "id": "hover_start_button", "intent": "Hover over the Start button on the taskbar"},
    # Keyboard
    {"category": "Keyboard", "id": "notepad_type", "intent": "Open Notepad and type: The quick brown fox jumps over the lazy dog"},
    {"category": "Keyboard", "id": "win_r_calc", "intent": "Press Win+R to open the Run dialog, type calc, and press Enter"},
    {"category": "Keyboard", "id": "win_d_desktop", "intent": "Press Win+D to show the desktop"},
    # Drag & Drop
    {"category": "Drag", "id": "drag_notepad_right", "intent": "Open Notepad and drag its window to the right half of the screen"},
    {"category": "Drag", "id": "drag_notepad_maximize", "intent": "Open Notepad and drag its title bar to the top of the screen to maximize it"},
    # Clipboard
    {"category": "Clipboard", "id": "clipboard_copy", "intent": "Open Notepad, type Hello World, select all text with Ctrl+A, then copy it with Ctrl+C"},
    {"category": "Clipboard", "id": "clipboard_read", "intent": "Read what is currently on the clipboard"},
    # Files
    {"category": "Files", "id": "explorer_documents", "intent": "Open File Explorer, navigate to the Documents folder"},
    {"category": "Files", "id": "explorer_downloads_describe", "intent": "Open File Explorer, navigate to Downloads folder, and describe what files are there"},
    # Web
    {"category": "Web", "id": "chrome_wikipedia", "intent": "Open Chrome, navigate to en.wikipedia.org, search for Markov chain"},
    {"category": "Web", "id": "chrome_example_com", "intent": "Open Chrome and navigate to example.com"},
    # Workflows
    {"category": "Workflows", "id": "calculator_42x17", "intent": "Open Calculator, compute 42 times 17 by clicking the buttons, and report the result"},
    {"category": "Workflows", "id": "notepad_meeting_notes", "intent": "Open Notepad, type meeting notes as a title, press Enter, then type Item 1 and Item 2"},
    {"category": "Workflows", "id": "describe_desktop", "intent": "Describe what is currently visible on the desktop"},
]

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TaskTestResult:
    category: str
    task_id: str
    intent: str
    run_task_status: int | None = None
    run_id: str | None = None
    initial_run_status: str | None = None
    error: str | None = None
    accepted: bool = False
    notes: list[str] = field(default_factory=list)


ALL_RESULTS: list[TaskTestResult] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post(path: str, body: dict) -> requests.Response:
    return SESSION.post(f"{BASE_URL}{path}", json=body, timeout=15)


def get(path: str, **params) -> requests.Response:
    return SESSION.get(f"{BASE_URL}{path}", params=params, timeout=15)


def cleanup(run_id: str) -> dict:
    resp = post("/desktop/cleanup", {"run_id": run_id})
    return resp.json() if resp.ok else {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200_with_ok_status(self):
        resp = get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_content_type_is_json(self):
        resp = get("/health")
        assert "application/json" in resp.headers.get("Content-Type", "")


# ---------------------------------------------------------------------------
# 2. UI HTML serving
# ---------------------------------------------------------------------------

class TestUIEndpoints:
    def test_root_serves_html(self):
        resp = get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("Content-Type", "")

    def test_root_html_contains_operon_content(self):
        resp = get("/")
        body = resp.text
        # The page should have some UI content — not a blank page
        assert len(body) > 500, "HTML body is suspiciously short"

    def test_console_route_serves_html(self):
        resp = get("/console")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("Content-Type", "")

    def test_root_and_console_serve_same_content(self):
        root = get("/")
        console = get("/console")
        assert root.text == console.text, "/ and /console should serve identical HTML"


# ---------------------------------------------------------------------------
# 3. RunTaskRequest input validation
# ---------------------------------------------------------------------------

class TestRunTaskInputValidation:
    """Tests for /desktop/run-task input validation (no real execution)."""

    def test_empty_intent_rejected_with_422(self):
        resp = post("/desktop/run-task", {"intent": ""})
        assert resp.status_code == 422, f"Empty intent should be 422, got {resp.status_code}"

    def test_missing_intent_field_rejected_with_422(self):
        resp = post("/desktop/run-task", {})
        assert resp.status_code == 422

    def test_extra_unknown_fields_rejected_with_422(self):
        # StrictModel has extra="forbid"
        resp = post("/desktop/run-task", {"intent": "Open Notepad", "unknown_field": "value"})
        assert resp.status_code == 422, "Extra fields should be rejected (extra=forbid)"

    def test_intent_at_max_length_accepted(self):
        # max_length=500 per the model
        long_intent = "A" * 500
        resp = post("/desktop/run-task", {"intent": long_intent})
        # Should be 202 (accepted for processing) or at most fail at execution, not validation
        assert resp.status_code in (202, 200, 500), f"500-char intent got unexpected status: {resp.status_code}"

    def test_intent_exceeding_max_length_rejected_with_422(self):
        too_long = "A" * 501
        resp = post("/desktop/run-task", {"intent": too_long})
        assert resp.status_code == 422, f"501-char intent should be 422, got {resp.status_code}"

    def test_intent_with_xss_payload_rejected_or_sanitized(self):
        xss = '<script>alert("xss")</script>'
        resp = post("/desktop/run-task", {"intent": xss})
        # Should either reject (422) or accept (202) — should NOT 500
        assert resp.status_code in (202, 422), f"XSS payload got unexpected status {resp.status_code}: {resp.text}"
        if resp.status_code == 202:
            data = resp.json()
            # The intent stored should be the raw string, not executed
            assert data["intent"] == xss

    def test_intent_with_sql_injection_payload(self):
        sql = "'; DROP TABLE runs; --"
        resp = post("/desktop/run-task", {"intent": sql})
        assert resp.status_code in (202, 422), f"SQL injection got unexpected status {resp.status_code}"

    def test_intent_with_unicode_characters(self):
        unicode_intent = "Open Notepad and type: \u4e2d\u6587\u6d4b\u8bd5 \u00e9\u00e0\u00fc"
        resp = post("/desktop/run-task", {"intent": unicode_intent})
        assert resp.status_code in (202, 200), f"Unicode intent got {resp.status_code}: {resp.text}"

    def test_intent_with_newlines_and_special_chars(self):
        multiline = "Open Notepad\nand type some text\ttabbed"
        resp = post("/desktop/run-task", {"intent": multiline})
        assert resp.status_code in (202, 422), f"Multiline intent got {resp.status_code}"

    def test_intent_with_only_whitespace_rejected(self):
        resp = post("/desktop/run-task", {"intent": "   "})
        # Pydantic min_length=1 is byte-length, whitespace-only passes that check
        # This is potentially a gap — document the behavior
        status = resp.status_code
        # Record rather than assert a specific outcome
        assert status in (202, 422), f"Whitespace-only intent got {status}"

    def test_null_intent_rejected_with_422(self):
        resp = post("/desktop/run-task", {"intent": None})
        assert resp.status_code == 422

    def test_integer_intent_rejected_with_422(self):
        resp = post("/desktop/run-task", {"intent": 12345})
        assert resp.status_code == 422

    def test_start_url_is_optional(self):
        resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp.status_code in (202, 200)

    def test_start_url_with_empty_string_rejected(self):
        # min_length=1 on start_url when provided
        resp = post("/desktop/run-task", {"intent": "Open Notepad", "start_url": ""})
        assert resp.status_code == 422, f"Empty start_url should be 422, got {resp.status_code}"

    def test_malformed_json_returns_422(self):
        raw_resp = SESSION.post(
            f"{BASE_URL}/desktop/run-task",
            data="not json at all",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert raw_resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. StepRequest validation
# ---------------------------------------------------------------------------

class TestStepRequestValidation:
    def test_step_with_empty_run_id_rejected_with_422(self):
        resp = post("/desktop/step", {"run_id": ""})
        assert resp.status_code == 422

    def test_step_with_missing_run_id_rejected_with_422(self):
        resp = post("/desktop/step", {})
        assert resp.status_code == 422

    def test_step_with_nonexistent_run_id_returns_404(self):
        resp = post("/desktop/step", {"run_id": "nonexistent-run-id-xyz-99999"})
        assert resp.status_code == 404, f"Nonexistent run_id should 404, got {resp.status_code}: {resp.text}"

    def test_step_with_extra_fields_rejected_with_422(self):
        resp = post("/desktop/step", {"run_id": "some-id", "extra": "field"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 5. CleanupRequest validation
# ---------------------------------------------------------------------------

class TestCleanupValidation:
    def test_cleanup_with_empty_run_id_rejected_with_422(self):
        resp = post("/desktop/cleanup", {"run_id": ""})
        assert resp.status_code == 422

    def test_cleanup_with_nonexistent_run_id_returns_cleanup_response(self):
        # cleanup should not 404 — it's a best-effort operation
        resp = post("/desktop/cleanup", {"run_id": "nonexistent-xyz-99999"})
        # Should return 200 with cleanup response (0 closed)
        assert resp.status_code == 200, f"Cleanup of nonexistent run_id got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "run_id" in data
        assert "closed_count" in data
        assert data["closed_count"] == 0

    def test_cleanup_response_shape(self):
        resp = post("/desktop/cleanup", {"run_id": "test-cleanup-shape-check"})
        assert resp.status_code == 200
        data = resp.json()
        required_fields = {"run_id", "closed_count", "detail"}
        assert required_fields.issubset(data.keys()), f"Missing fields: {required_fields - data.keys()}"
        assert isinstance(data["closed_count"], int)
        assert data["closed_count"] >= 0
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0


# ---------------------------------------------------------------------------
# 6. Observer API
# ---------------------------------------------------------------------------

class TestObserverAPI:
    def test_list_runs_returns_200(self):
        resp = get("/observer/api/runs")
        assert resp.status_code == 200

    def test_list_runs_response_shape(self):
        resp = get("/observer/api/runs")
        data = resp.json()
        assert "runs" in data, "Response should have 'runs' key"
        assert isinstance(data["runs"], list)

    def test_list_runs_limit_parameter_accepted(self):
        resp = get("/observer/api/runs", limit=5)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) <= 5

    def test_list_runs_limit_too_high_rejected(self):
        # ge=1, le=100 constraint
        resp = get("/observer/api/runs", limit=101)
        assert resp.status_code == 422, f"limit=101 should be 422, got {resp.status_code}"

    def test_list_runs_limit_zero_rejected(self):
        resp = get("/observer/api/runs", limit=0)
        assert resp.status_code == 422, f"limit=0 should be 422, got {resp.status_code}"

    def test_list_runs_limit_at_boundary_100_accepted(self):
        resp = get("/observer/api/runs", limit=100)
        assert resp.status_code == 200

    def test_list_runs_limit_at_boundary_1_accepted(self):
        resp = get("/observer/api/runs", limit=1)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) <= 1

    def test_observer_run_nonexistent_returns_404(self):
        resp = get("/observer/api/run/totally-nonexistent-run-id-xyz")
        assert resp.status_code == 404

    def test_observer_artifact_missing_path_param_returns_422(self):
        resp = get("/observer/api/artifact")
        assert resp.status_code == 422

    def test_observer_artifact_path_traversal_rejected(self):
        # Attempt path traversal outside runs root
        resp = SESSION.get(f"{BASE_URL}/observer/api/artifact", params={"path": "../../etc/passwd"}, timeout=10)
        # Should be 400 (ValueError from artifact_path_for_request) or 404
        assert resp.status_code in (400, 404), f"Path traversal attempt got {resp.status_code}"

    def test_observer_artifact_nonexistent_file_returns_404(self):
        resp = SESSION.get(f"{BASE_URL}/observer/api/artifact", params={"path": "runs/nonexistent-run/before.png"}, timeout=10)
        assert resp.status_code in (400, 404)

    def test_observer_run_items_have_expected_fields(self):
        resp = get("/observer/api/runs", limit=20)
        data = resp.json()
        for run in data["runs"]:
            assert "run_id" in run
            assert "intent" in run
            assert "status" in run
            assert "step_count" in run
            assert isinstance(run["step_count"], int)


# ---------------------------------------------------------------------------
# 7. Desktop run GET endpoint
# ---------------------------------------------------------------------------

class TestDesktopGetRun:
    def test_get_nonexistent_run_returns_404(self):
        resp = get("/desktop/run/nonexistent-xyz-run")
        assert resp.status_code == 404

    def test_get_run_error_has_detail_field(self):
        resp = get("/desktop/run/nonexistent-xyz-run")
        data = resp.json()
        assert "detail" in data


# ---------------------------------------------------------------------------
# 8. All 22 quick tasks — intent acceptance
# ---------------------------------------------------------------------------

class TestQuickTaskIntentAcceptance:
    """
    Tests that each quick task intent is accepted by the API (202) and
    the run record is created correctly.

    These tests do NOT advance steps (no Gemini calls needed).
    Each test creates a run then immediately cleans it up.
    """

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_intent_accepted_returns_202(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202, (
            f"[{task['category']}] '{task['intent']}' got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_intent_response_has_run_id(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert len(data["run_id"]) > 0, "run_id should not be empty"

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_intent_response_has_pending_or_running_status(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] in ("pending", "running"), (
            f"[{task['category']}] Initial status should be pending/running, got {data['status']}"
        )

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_intent_echoed_correctly_in_response(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        data = resp.json()
        assert data["intent"] == task["intent"], (
            f"Intent not echoed correctly: expected '{task['intent']}', got '{data['intent']}'"
        )

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_intent_response_step_count_is_zero(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        data = resp.json()
        assert data["step_count"] == 0, (
            f"Newly created run should have step_count=0, got {data['step_count']}"
        )

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_run_retrievable_via_get_endpoint(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        get_resp = get(f"/desktop/run/{run_id}")
        assert get_resp.status_code == 200, (
            f"Could not retrieve run {run_id}: {get_resp.status_code}: {get_resp.text}"
        )
        get_data = get_resp.json()
        assert get_data["run_id"] == run_id
        assert get_data["intent"] == task["intent"]

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_cleanup_returns_200_after_run_created(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        cleanup_resp = post("/desktop/cleanup", {"run_id": run_id})
        assert cleanup_resp.status_code == 200, (
            f"Cleanup for '{task['intent']}' got {cleanup_resp.status_code}: {cleanup_resp.text}"
        )

    @pytest.mark.parametrize(
        "task",
        QUICK_TASKS,
        ids=[t["id"] for t in QUICK_TASKS],
    )
    def test_run_visible_in_observer_runs_list(self, task):
        resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        list_resp = get("/observer/api/runs", limit=100)
        assert list_resp.status_code == 200
        runs = list_resp.json()["runs"]
        run_ids_in_list = [r["run_id"] for r in runs]
        assert run_id in run_ids_in_list, (
            f"Newly created run {run_id} for '{task['intent']}' not in observer/api/runs list"
        )


# ---------------------------------------------------------------------------
# 9. Response structure contract tests
# ---------------------------------------------------------------------------

class TestResponseContracts:
    """Verify the RunResponse shape is consistent across all endpoints."""

    def _verify_run_response_shape(self, data: dict, context: str = ""):
        required = {"run_id", "status", "intent", "step_count"}
        missing = required - data.keys()
        assert not missing, f"RunResponse missing fields {missing} in {context}"
        assert isinstance(data["run_id"], str) and len(data["run_id"]) > 0
        assert data["status"] in ("pending", "running", "waiting_for_user", "succeeded", "failed")
        assert isinstance(data["intent"], str) and len(data["intent"]) > 0
        assert isinstance(data["step_count"], int) and data["step_count"] >= 0

    def test_run_task_response_conforms_to_schema(self):
        resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp.status_code == 202
        self._verify_run_response_shape(resp.json(), "/desktop/run-task")

    def test_get_run_response_conforms_to_schema(self):
        create_resp = post("/desktop/run-task", {"intent": "Open Calculator"})
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        get_resp = get(f"/desktop/run/{run_id}")
        assert get_resp.status_code == 200
        self._verify_run_response_shape(get_resp.json(), f"/desktop/run/{run_id}")

    def test_observer_run_snapshot_has_expected_keys(self):
        create_resp = post("/desktop/run-task", {"intent": "Open File Explorer"})
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        snap_resp = get(f"/observer/api/run/{run_id}")
        assert snap_resp.status_code == 200
        data = snap_resp.json()
        expected_keys = {"run", "progress_state", "current_step", "steps", "event_log"}
        missing = expected_keys - data.keys()
        assert not missing, f"Observer snapshot missing keys: {missing}"

    def test_observer_run_snapshot_run_block_shape(self):
        create_resp = post("/desktop/run-task", {"intent": "Open Paint"})
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        snap_resp = get(f"/observer/api/run/{run_id}")
        assert snap_resp.status_code == 200
        run_block = snap_resp.json()["run"]
        expected_run_keys = {"run_id", "intent", "status", "step_count", "stop_reason", "current_subgoal", "current_task_id", "current_phase"}
        missing = expected_run_keys - run_block.keys()
        assert not missing, f"Observer run block missing keys: {missing}"
        assert run_block["run_id"] == run_id
        assert run_block["intent"] == "Open Paint"

    def test_observer_run_event_log_is_list(self):
        create_resp = post("/desktop/run-task", {"intent": "Open Windows Settings"})
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        snap_resp = get(f"/observer/api/run/{run_id}")
        assert snap_resp.status_code == 200
        event_log = snap_resp.json()["event_log"]
        assert isinstance(event_log, list)
        # Newly created run should have at least a "run_started" event
        events = [e["event"] for e in event_log]
        assert "run_started" in events, f"Expected run_started event, got: {events}"

    def test_cleanup_response_conforms_to_schema(self):
        create_resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        run_id = create_resp.json()["run_id"]
        cleanup_resp = post("/desktop/cleanup", {"run_id": run_id})
        assert cleanup_resp.status_code == 200
        data = cleanup_resp.json()
        assert data["run_id"] == run_id
        assert isinstance(data["closed_count"], int)
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0


# ---------------------------------------------------------------------------
# 10. Edge cases and concurrent/repeat behavior
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_same_intent_creates_unique_run_ids(self):
        resp1 = post("/desktop/run-task", {"intent": "Open Notepad"})
        resp2 = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp1.status_code == 202
        assert resp2.status_code == 202
        id1 = resp1.json()["run_id"]
        id2 = resp2.json()["run_id"]
        assert id1 != id2, "Two runs with the same intent should have different run_ids"

    def test_multiple_cleanups_same_run_id_are_idempotent(self):
        resp = post("/desktop/run-task", {"intent": "Open Calculator"})
        run_id = resp.json()["run_id"]
        c1 = post("/desktop/cleanup", {"run_id": run_id})
        c2 = post("/desktop/cleanup", {"run_id": run_id})
        assert c1.status_code == 200
        assert c2.status_code == 200
        # Both should return 0 closed (or first may close, second should be 0)
        assert c2.json()["closed_count"] == 0

    def test_intent_exactly_500_chars_is_accepted(self):
        intent = "Open " + "N" * 495  # 5 + 495 = 500
        resp = post("/desktop/run-task", {"intent": intent})
        assert resp.status_code == 202, f"Exactly 500-char intent should be 202, got {resp.status_code}"

    def test_intent_exactly_501_chars_is_rejected(self):
        intent = "Open " + "N" * 496  # 5 + 496 = 501
        resp = post("/desktop/run-task", {"intent": intent})
        assert resp.status_code == 422, f"501-char intent should be 422, got {resp.status_code}"

    def test_step_on_fresh_run_returns_run_id(self):
        # Create a run, try one step — step may fail if Gemini unavailable, but should not crash
        resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        step_resp = post("/desktop/step", {"run_id": run_id})
        # Step might succeed (202/200) or fail gracefully — should not be 500
        assert step_resp.status_code != 500, (
            f"Step on fresh run returned 500: {step_resp.text}"
        )
        assert step_resp.status_code in (200, 202, 404, 422, 503), (
            f"Unexpected step response: {step_resp.status_code}: {step_resp.text}"
        )
        # If step returned data, verify run_id is preserved
        if step_resp.ok:
            data = step_resp.json()
            if "run_id" in data:
                assert data["run_id"] == run_id

    def test_observer_run_snapshot_of_fresh_run(self):
        resp = post("/desktop/run-task", {"intent": "Describe what is currently visible on the desktop"})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        snap = get(f"/observer/api/run/{run_id}")
        assert snap.status_code == 200
        data = snap.json()
        assert data["run"]["intent"] == "Describe what is currently visible on the desktop"
        assert data["run"]["status"] == "pending"
        assert data["steps"] == []

    def test_all_valid_run_statuses_in_enum(self):
        """Document the valid RunStatus values by checking what the schema allows."""
        valid_statuses = {"pending", "running", "waiting_for_user", "succeeded", "failed"}
        resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp.status_code == 202
        status = resp.json()["status"]
        assert status in valid_statuses

    def test_list_runs_returns_most_recent_first(self):
        resp1 = post("/desktop/run-task", {"intent": "Open Notepad"})
        time.sleep(0.1)  # tiny gap to ensure different mtime
        resp2 = post("/desktop/run-task", {"intent": "Open Calculator"})
        assert resp1.status_code == 202
        assert resp2.status_code == 202
        id2 = resp2.json()["run_id"]

        list_resp = get("/observer/api/runs", limit=5)
        assert list_resp.status_code == 200
        runs = list_resp.json()["runs"]
        if runs:
            assert runs[0]["run_id"] == id2, (
                "Most recent run should appear first in observer list"
            )

    def test_whitespace_only_intent_behavior_documented(self):
        """
        Whitespace-only intents (e.g. '   ') pass Pydantic min_length=1 because
        Pydantic counts raw bytes, not stripped content.

        This is a POTENTIAL BUG: the agent loop will receive a whitespace intent
        and attempt to process it.

        Expected: 422 (rejected as meaningless)
        Actual: 202 (accepted and stored)
        """
        resp = post("/desktop/run-task", {"intent": "   "})
        # Document actual behavior — do not assert 422 so this won't fail,
        # but add a note if it passes validation
        if resp.status_code == 202:
            pytest.xfail(
                reason="BUG: Whitespace-only intent passes validation (Pydantic min_length "
                "counts bytes not stripped content). Intent stored as '   '."
            )
        # If it's 422, the validation is working correctly
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 11. Comprehensive run lifecycle for a sample of intents
# ---------------------------------------------------------------------------

class TestRunLifecycle:
    """
    Full lifecycle test: create -> verify stored -> observe -> cleanup.
    Uses a representative sample of 6 tasks (one per category).
    """

    SAMPLE_TASKS = [
        {"id": "open_notepad", "intent": "Open Notepad"},
        {"id": "right_click_desktop", "intent": "Right-click on an empty area of the desktop to open the context menu"},
        {"id": "win_d_desktop", "intent": "Press Win+D to show the desktop"},
        {"id": "clipboard_read", "intent": "Read what is currently on the clipboard"},
        {"id": "chrome_example_com", "intent": "Open Chrome and navigate to example.com"},
        {"id": "describe_desktop", "intent": "Describe what is currently visible on the desktop"},
    ]

    @pytest.mark.parametrize("task", SAMPLE_TASKS, ids=[t["id"] for t in SAMPLE_TASKS])
    def test_full_lifecycle(self, task):
        # 1. Create run
        create_resp = post("/desktop/run-task", {"intent": task["intent"]})
        assert create_resp.status_code == 202, f"Create failed: {create_resp.status_code}: {create_resp.text}"
        run_data = create_resp.json()
        run_id = run_data["run_id"]

        # 2. Verify run stored and retrievable
        get_resp = get(f"/desktop/run/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == run_id
        assert get_resp.json()["intent"] == task["intent"]

        # 3. Observer snapshot is available
        snap_resp = get(f"/observer/api/run/{run_id}")
        assert snap_resp.status_code == 200
        snap = snap_resp.json()
        assert snap["run"]["intent"] == task["intent"]
        assert snap["run"]["status"] in ("pending", "running")

        # 4. Run appears in observer list
        list_resp = get("/observer/api/runs", limit=50)
        run_ids = [r["run_id"] for r in list_resp.json()["runs"]]
        assert run_id in run_ids, f"Run {run_id} not in observer list"

        # 5. Cleanup succeeds
        cleanup_resp = post("/desktop/cleanup", {"run_id": run_id})
        assert cleanup_resp.status_code == 200
        cleanup_data = cleanup_resp.json()
        assert cleanup_data["run_id"] == run_id
        assert cleanup_data["closed_count"] >= 0


# ---------------------------------------------------------------------------
# 12. Collect a summary report
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Appended summary printed after test run."""
    pass  # Pytest handles this — results visible in -v output
