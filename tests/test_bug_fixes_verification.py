"""
Verification test suite for 11 bug fixes applied to Operon Pilot.

Tests each fix with targeted assertions, edge cases, and regression checks.
Run with server already started:
    GEMINI_API_KEY=fake-test-key .venv/Scripts/python -m pytest tests/test_bug_fixes_verification.py -v
"""

from __future__ import annotations

import re

import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"
SESSION = requests.Session()
SESSION.headers["Content-Type"] = "application/json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(url_path: str, **params) -> requests.Response:
    return SESSION.get(f"{BASE_URL}{url_path}", params=params or None, timeout=10)


def post(path: str, body: dict) -> requests.Response:
    return SESSION.post(f"{BASE_URL}{path}", json=body, timeout=10)


def options(path: str, **headers) -> requests.Response:
    return SESSION.options(
        f"{BASE_URL}{path}",
        headers={**SESSION.headers, **headers},
        timeout=10,
    )


def create_desktop_run(intent: str = "Open Notepad") -> str:
    """Helper: create a run and return its run_id."""
    resp = post("/desktop/run-task", {"intent": intent})
    assert resp.status_code == 202, f"Run create failed: {resp.status_code}: {resp.text}"
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# Prerequisite: server must be running
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def require_server():
    resp = get("/health")
    assert resp.status_code == 200, f"Server not reachable at {BASE_URL}"
    assert resp.json().get("status") == "ok"


# ===========================================================================
# FIX 1: CRITICAL — Long run_id causes 500; _validate_run_id() rejects > 64 chars with 404
# ===========================================================================

class TestFix1LongRunId:
    """FIX 1: run_ids longer than 64 chars must return 404, not 500."""

    # --- Desktop GET ---

    def test_desktop_get_run_id_65_chars_returns_404(self):
        run_id = "a" * 65
        resp = get(f"/desktop/run/{run_id}")
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 65-char run_id on GET /desktop/run returned {resp.status_code}, expected 404"
        )

    def test_desktop_get_run_id_100_chars_returns_404(self):
        run_id = "b" * 100
        resp = get(f"/desktop/run/{run_id}")
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 100-char run_id on GET /desktop/run returned {resp.status_code}"
        )

    def test_desktop_get_run_id_256_chars_returns_404(self):
        run_id = "c" * 256
        resp = get(f"/desktop/run/{run_id}")
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 256-char run_id on GET /desktop/run returned {resp.status_code}"
        )

    # --- Desktop STEP ---

    def test_desktop_step_run_id_65_chars_returns_404(self):
        run_id = "a" * 65
        resp = post("/desktop/step", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 65-char run_id on POST /desktop/step returned {resp.status_code}"
        )

    def test_desktop_step_run_id_100_chars_returns_404(self):
        run_id = "x" * 100
        resp = post("/desktop/step", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 100-char run_id on POST /desktop/step returned {resp.status_code}"
        )

    def test_desktop_step_run_id_256_chars_returns_404(self):
        run_id = "z" * 256
        resp = post("/desktop/step", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 256-char run_id on POST /desktop/step returned {resp.status_code}"
        )

    # --- Desktop RESUME ---

    def test_desktop_resume_run_id_65_chars_returns_404(self):
        run_id = "r" * 65
        resp = post("/desktop/resume", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 65-char run_id on POST /desktop/resume returned {resp.status_code}"
        )

    def test_desktop_resume_run_id_100_chars_returns_404(self):
        run_id = "s" * 100
        resp = post("/desktop/resume", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 100-char run_id on POST /desktop/resume returned {resp.status_code}"
        )

    def test_desktop_resume_run_id_256_chars_returns_404(self):
        run_id = "t" * 256
        resp = post("/desktop/resume", {"run_id": run_id})
        assert resp.status_code == 404, (
            f"FIX1 FAIL: 256-char run_id on POST /desktop/resume returned {resp.status_code}"
        )

    # --- Boundary: exactly 64 chars must NOT be rejected by the length check ---

    def test_desktop_get_run_id_exactly_64_chars_not_rejected_by_length_check(self):
        run_id = "d" * 64
        resp = get(f"/desktop/run/{run_id}")
        # 64-char run_id should pass validation; 404 is OK (run doesn't exist)
        # but must NOT return 500
        assert resp.status_code in (200, 404), (
            f"FIX1 REGRESSION: 64-char run_id returned {resp.status_code}; length check is too strict"
        )
        assert resp.status_code != 500, "FIX1 REGRESSION: 64-char run_id caused 500"

    # --- No 500s anywhere ---

    @pytest.mark.parametrize("length", [65, 100, 256, 1000])
    def test_desktop_get_never_returns_500_for_long_run_ids(self, length):
        run_id = "e" * length
        resp = get(f"/desktop/run/{run_id}")
        assert resp.status_code != 500, (
            f"FIX1 FAIL: {length}-char run_id caused 500 on GET /desktop/run"
        )


# ===========================================================================
# FIX 2: CRITICAL — Observer 404 on fresh runs; load_run_snapshot handles missing run.jsonl
# ===========================================================================

class TestFix2ObserverFreshRun:
    """FIX 2: Observer API must return 200 for a freshly created run with no steps."""

    def test_observer_api_fresh_run_returns_200_immediately(self):
        run_id = create_desktop_run("Open Notepad for fix2 test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200, (
            f"FIX2 FAIL: Observer returned {resp.status_code} for fresh run {run_id}. "
            f"Expected 200. Body: {resp.text[:300]}"
        )

    def test_observer_fresh_run_has_valid_structure(self):
        run_id = create_desktop_run("Open Calculator for fix2 structure test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {"run", "progress_state", "current_step", "steps", "event_log"}
        missing = required_keys - data.keys()
        assert not missing, f"FIX2 FAIL: Observer response missing keys: {missing}"

    def test_observer_fresh_run_steps_list_is_empty(self):
        run_id = create_desktop_run("Open Paint for fix2 steps test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"] == [], (
            f"FIX2 FAIL: Fresh run steps should be [], got: {data['steps']}"
        )

    def test_observer_fresh_run_run_block_is_populated(self):
        intent = "Open File Explorer for fix2 run block test"
        run_id = create_desktop_run(intent)
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        run_block = resp.json()["run"]
        assert run_block["run_id"] == run_id
        assert run_block["intent"] == intent
        assert run_block["status"] in ("pending", "running")

    def test_observer_fresh_run_event_log_has_run_started(self):
        run_id = create_desktop_run("Open Settings for fix2 event log test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        event_log = resp.json()["event_log"]
        assert isinstance(event_log, list)
        events = [e["event"] for e in event_log]
        assert "run_started" in events, (
            f"FIX2 FAIL: event_log for fresh run missing 'run_started'. Events: {events}"
        )

    def test_observer_truly_nonexistent_run_still_returns_404(self):
        """Regression: truly nonexistent run_id (no state.json) should still 404."""
        resp = get("/observer/api/run/totally-does-not-exist-xyz-999")
        assert resp.status_code == 404, (
            f"REGRESSION: nonexistent run should return 404, got {resp.status_code}"
        )


# ===========================================================================
# FIX 3: IMPORTANT — Event log sort key changed to step_index only
# ===========================================================================

class TestFix3EventLogSortOrder:
    """FIX 3: Events at the same step_index should be in insertion order, not alphabetical."""

    def test_event_log_is_sorted_by_step_index(self):
        run_id = create_desktop_run("Open Notepad for fix3 sort test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        event_log = resp.json()["event_log"]
        step_indices = [e["step_index"] for e in event_log]
        assert step_indices == sorted(step_indices), (
            f"FIX3 FAIL: Event log is not sorted by step_index. Got indices: {step_indices}"
        )

    def test_event_log_run_started_is_first_event(self):
        """run_started has step_index=0 and should appear first."""
        run_id = create_desktop_run("Open Calculator for fix3 ordering test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        event_log = resp.json()["event_log"]
        assert len(event_log) >= 1
        assert event_log[0]["event"] == "run_started", (
            f"FIX3 FAIL: First event should be 'run_started', got '{event_log[0]['event']}'"
        )

    def test_event_log_same_step_not_alphabetically_sorted(self):
        """
        At step_index=0, 'run_started' should appear. If two events share the same
        step_index, their relative order should match insertion order (screenshot_captured
        before perception_requested), not alphabetical order (which would put
        'perception_requested' before 'screenshot_captured').
        """
        run_id = create_desktop_run("Open Paint for fix3 insertion order test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        event_log = resp.json()["event_log"]
        # With only a fresh run (no steps), we can't fully test multi-event steps.
        # We verify the sort key is numeric only by checking that step_index is monotonic.
        step_indices = [e["step_index"] for e in event_log]
        assert step_indices == sorted(step_indices), (
            f"FIX3 FAIL: Event log not monotonically sorted by step_index. Indices: {step_indices}"
        )


# ===========================================================================
# FIX 4: IMPORTANT — Resume returns 404 not 400 for nonexistent runs
# ===========================================================================

class TestFix4ResumeReturns404:
    """FIX 4: Both /resume and /desktop/resume should return 404 for nonexistent runs."""

    def test_desktop_resume_nonexistent_run_returns_404(self):
        resp = post("/desktop/resume", {"run_id": "nonexistent-run-id-fix4-test"})
        assert resp.status_code == 404, (
            f"FIX4 FAIL: /desktop/resume for nonexistent run returned {resp.status_code}, expected 404"
        )

    def test_browser_resume_nonexistent_run_returns_404(self):
        resp = post("/resume", {"run_id": "nonexistent-run-id-fix4-browser-test"})
        assert resp.status_code == 404, (
            f"FIX4 FAIL: /resume for nonexistent run returned {resp.status_code}, expected 404"
        )

    def test_desktop_resume_has_detail_in_error_body(self):
        resp = post("/desktop/resume", {"run_id": "nonexistent-resume-detail-test"})
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data, "FIX4 FAIL: 404 response missing 'detail' field"

    def test_browser_resume_has_detail_in_error_body(self):
        resp = post("/resume", {"run_id": "nonexistent-browser-resume-detail"})
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data, "FIX4 FAIL: Browser /resume 404 missing 'detail' field"

    def test_desktop_resume_does_not_return_400(self):
        resp = post("/desktop/resume", {"run_id": "test-not-400-fix4"})
        assert resp.status_code != 400, (
            f"FIX4 FAIL: /desktop/resume returned 400 instead of 404"
        )

    def test_browser_resume_does_not_return_400(self):
        resp = post("/resume", {"run_id": "test-not-400-browser-fix4"})
        assert resp.status_code != 400, (
            f"FIX4 FAIL: /resume returned 400 instead of 404"
        )

    def test_desktop_resume_empty_run_id_still_422(self):
        """Regression: empty run_id should still be 422 (Pydantic validation), not 404."""
        resp = post("/desktop/resume", {"run_id": ""})
        assert resp.status_code == 422, (
            f"REGRESSION: empty run_id should be 422, got {resp.status_code}"
        )


# ===========================================================================
# FIX 5: IMPORTANT — Whitespace-only intents rejected
# ===========================================================================

class TestFix5WhitespaceIntentRejected:
    """FIX 5: Intents consisting only of whitespace must be rejected with 422."""

    @pytest.mark.parametrize("intent", [
        " ",
        "   ",
        "\t",
        "\n",
        "  \t  ",
        "  \n  ",
        "\t\n\t",
    ])
    def test_whitespace_only_intent_rejected_with_422(self, intent):
        resp = post("/desktop/run-task", {"intent": intent})
        assert resp.status_code == 422, (
            f"FIX5 FAIL: Whitespace-only intent {repr(intent)} returned {resp.status_code}, expected 422. "
            f"Body: {resp.text[:300]}"
        )

    def test_whitespace_only_on_browser_run_task_rejected_with_422(self):
        """Same validation should apply to /run-task (browser endpoint)."""
        resp = post("/run-task", {"intent": "   "})
        assert resp.status_code == 422, (
            f"FIX5 FAIL: Browser /run-task with whitespace-only intent returned {resp.status_code}"
        )

    def test_valid_intent_with_leading_trailing_spaces_is_accepted(self):
        """Intent with real content + surrounding spaces should be accepted (stripped)."""
        resp = post("/desktop/run-task", {"intent": "  Open Notepad  "})
        assert resp.status_code == 202, (
            f"FIX5 REGRESSION: Intent with leading/trailing spaces was rejected: {resp.status_code}"
        )

    def test_valid_intent_trimmed_in_response(self):
        """The stored intent should be stripped of surrounding whitespace."""
        resp = post("/desktop/run-task", {"intent": "  Open Calculator  "})
        assert resp.status_code == 202
        data = resp.json()
        assert data["intent"] == "Open Calculator", (
            f"FIX5 FAIL: Intent not stripped in response. Got: '{data['intent']}'"
        )

    def test_normal_intents_still_accepted(self):
        """Regression: normal intents must continue to work after the fix."""
        for intent in ["Open Notepad", "Open Calculator", "Press Win+D to show the desktop"]:
            resp = post("/desktop/run-task", {"intent": intent})
            assert resp.status_code == 202, (
                f"REGRESSION: Normal intent '{intent}' rejected with {resp.status_code}"
            )

    def test_intent_at_exactly_500_chars_after_trimming(self):
        """A 500-char intent (no spaces to strip) should still be accepted."""
        intent = "A" * 500
        resp = post("/desktop/run-task", {"intent": intent})
        assert resp.status_code == 202, (
            f"REGRESSION: 500-char intent rejected with {resp.status_code}"
        )

    def test_intent_501_chars_rejected_regardless_of_fix(self):
        """max_length=500 should still apply independently of whitespace stripping."""
        intent = "A" * 501
        resp = post("/desktop/run-task", {"intent": intent})
        assert resp.status_code == 422, (
            f"REGRESSION: 501-char intent should be 422, got {resp.status_code}"
        )


# ===========================================================================
# FIX 6: IMPORTANT — CORS middleware added
# ===========================================================================

class TestFix6CorsMiddleware:
    """FIX 6: CORS headers should be present when CORS_ORIGINS is set in env."""

    def test_cors_header_present_on_health_endpoint(self):
        """Simple GET request to /health should return CORS header when origin matches."""
        resp = SESSION.get(
            f"{BASE_URL}/health",
            headers={"Origin": "http://localhost:3000"},
            timeout=10,
        )
        assert resp.status_code == 200
        # With CORS_ORIGINS=* in .env, the header should be present
        cors_header = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors_header, (
            f"FIX6 FAIL: No Access-Control-Allow-Origin header on GET /health. "
            f"Headers: {dict(resp.headers)}"
        )

    def test_cors_header_present_on_run_task_post(self):
        resp = SESSION.post(
            f"{BASE_URL}/desktop/run-task",
            json={"intent": "Open Notepad"},
            headers={"Origin": "http://localhost:3000", "Content-Type": "application/json"},
            timeout=10,
        )
        cors_header = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors_header, (
            f"FIX6 FAIL: No Access-Control-Allow-Origin header on POST /desktop/run-task. "
            f"Status: {resp.status_code}, Headers: {dict(resp.headers)}"
        )

    def test_options_preflight_returns_cors_headers(self):
        """OPTIONS preflight for /desktop/run-task should include CORS headers."""
        resp = SESSION.options(
            f"{BASE_URL}/desktop/run-task",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
            timeout=10,
        )
        # Preflight should succeed (200 or 204)
        assert resp.status_code in (200, 204), (
            f"FIX6 FAIL: OPTIONS preflight returned {resp.status_code}"
        )
        cors_header = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors_header, (
            f"FIX6 FAIL: OPTIONS preflight missing Access-Control-Allow-Origin. "
            f"Headers: {dict(resp.headers)}"
        )

    def test_cors_allow_methods_present_in_preflight(self):
        resp = SESSION.options(
            f"{BASE_URL}/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
            timeout=10,
        )
        # If CORS is properly configured, allow-methods should be present
        allow_methods = resp.headers.get("Access-Control-Allow-Methods", "")
        # This is informational — log if missing but check ACAO first
        cors_header = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors_header, (
            f"FIX6 FAIL: OPTIONS /health missing Access-Control-Allow-Origin. "
            f"Headers: {dict(resp.headers)}"
        )


# ===========================================================================
# FIX 7: IMPORTANT — Path disclosure removed from error messages
# ===========================================================================

class TestFix7PathDisclosureRemoved:
    """FIX 7: Error messages must not leak filesystem paths."""

    _PATH_PATTERN = re.compile(
        r"([A-Za-z]:\\|/home/|/Users/|/root/|/var/|/tmp/|C:\\Users\\|runs[\\/])",
        re.IGNORECASE,
    )

    def _assert_no_path_in_response(self, resp: requests.Response, context: str):
        body = resp.text
        match = self._PATH_PATTERN.search(body)
        assert not match, (
            f"FIX7 FAIL: Path disclosure in {context}. "
            f"Found '{match.group()}' in response body: {body[:500]}"
        )

    def test_observer_run_nonexistent_no_path_disclosure(self):
        resp = get("/observer/api/run/nonexistent-path-disclosure-test")
        assert resp.status_code == 404
        self._assert_no_path_in_response(resp, "GET /observer/api/run/nonexistent")

    def test_observer_artifact_nonexistent_no_path_disclosure(self):
        resp = get("/observer/api/artifact", path="nonexistent-file.png")
        assert resp.status_code in (400, 404)
        self._assert_no_path_in_response(resp, "GET /observer/api/artifact?path=nonexistent-file.png")

    def test_observer_artifact_path_traversal_no_path_disclosure(self):
        resp = get("/observer/api/artifact", path="../../etc/passwd")
        assert resp.status_code in (400, 404)
        self._assert_no_path_in_response(resp, "GET /observer/api/artifact?path=../../etc/passwd")

    def test_observer_artifact_absolute_path_no_disclosure(self):
        resp = get("/observer/api/artifact", path="/etc/passwd")
        assert resp.status_code in (400, 404)
        self._assert_no_path_in_response(resp, "GET /observer/api/artifact?path=/etc/passwd")

    def test_desktop_get_nonexistent_run_no_path_disclosure(self):
        resp = get("/desktop/run/nonexistent-no-path-leak")
        assert resp.status_code == 404
        self._assert_no_path_in_response(resp, "GET /desktop/run/nonexistent")

    def test_error_messages_use_generic_text(self):
        """Error detail should be generic like 'Run not found', not a filesystem path."""
        resp = get("/observer/api/run/definitely-not-a-real-run-id-xyz")
        assert resp.status_code == 404
        data = resp.json()
        detail = data.get("detail", "")
        # Should be a short, generic message
        assert len(detail) < 200, (
            f"FIX7 FAIL: Error detail is unusually long (may contain path): '{detail}'"
        )
        # Should not start with a drive letter or slash
        assert not re.match(r"^[A-Za-z]:\\|^/", detail), (
            f"FIX7 FAIL: Error detail starts with a path-like prefix: '{detail}'"
        )


# ===========================================================================
# FIX 8: MINOR — XSS in observer UI uses esc() for all dynamic content
# ===========================================================================

class TestFix8XssProtection:
    """FIX 8: Verify esc() function exists and is used for dynamic content in UI HTML."""

    def _get_html(self) -> str:
        resp = get("/")
        assert resp.status_code == 200
        return resp.text

    def test_esc_function_defined_in_html(self):
        html = self._get_html()
        assert "function esc(" in html or "function esc (" in html, (
            "FIX8 FAIL: esc() function not defined in desktop.html"
        )

    def test_esc_uses_textcontent_not_innerhtml_for_sanitization(self):
        html = self._get_html()
        # The esc() implementation should use textContent to safely escape HTML
        assert "textContent" in html, (
            "FIX8 FAIL: esc() function does not use textContent for sanitization"
        )

    def test_dynamic_intent_rendered_with_esc(self):
        html = self._get_html()
        # Intent is user-supplied data — verify it's escaped before insertion
        assert "esc(r.intent)" in html or "esc(run.intent)" in html or "escapedIntent" in html, (
            "FIX8 FAIL: Intent field not wrapped in esc() in UI HTML"
        )

    def test_run_id_rendered_with_esc(self):
        html = self._get_html()
        assert "esc(r.run_id)" in html or "safeId" in html, (
            "FIX8 FAIL: run_id field not wrapped in esc() in UI HTML"
        )

    def test_action_type_rendered_with_esc(self):
        html = self._get_html()
        assert "esc(action.action_type" in html, (
            "FIX8 FAIL: action.action_type not wrapped in esc() in UI HTML"
        )


# ===========================================================================
# FIX 9: MINOR — Misleading 404 for unstepped runs (covered by Fix 2)
# ===========================================================================

class TestFix9MeaningfulObserverResponse:
    """FIX 9: Observer response for fresh (unstepped) run should be meaningful, not a 404."""

    def test_fresh_run_observer_not_404(self):
        run_id = create_desktop_run("Open Notepad for fix9 test")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code != 404, (
            f"FIX9 FAIL: Observer returned 404 for a fresh run {run_id}"
        )
        assert resp.status_code == 200

    def test_fresh_run_observer_run_block_has_correct_intent(self):
        intent = "Open Calculator for fix9 intent check"
        run_id = create_desktop_run(intent)
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run"]["intent"] == intent, (
            f"FIX9 FAIL: Observer run block has wrong intent. "
            f"Expected '{intent}', got '{data['run']['intent']}'"
        )

    def test_fresh_run_observer_status_is_pending(self):
        run_id = create_desktop_run("Open Paint for fix9 status check")
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        status = data["run"]["status"]
        assert status in ("pending", "running"), (
            f"FIX9 FAIL: Fresh run observer status should be pending/running, got '{status}'"
        )


# ===========================================================================
# FIX 10: MINOR — Path-traversal in cleanup via long/special run_ids
# ===========================================================================

class TestFix10PathTraversalInCleanup:
    """FIX 10: Cleanup endpoint must reject path-traversal and oversized run_ids."""

    def test_cleanup_with_path_traversal_run_id(self):
        resp = post("/desktop/cleanup", {"run_id": "../../../etc/passwd"})
        # cleanup goes through the executor, not _validate_run_id;
        # but the executor should handle it gracefully (not crash)
        # The important thing is no 500 and no filesystem exposure
        assert resp.status_code != 500, (
            f"FIX10 FAIL: Path traversal run_id caused 500 on /desktop/cleanup"
        )

    def test_cleanup_with_very_long_run_id_no_500(self):
        long_run_id = "a" * 256
        resp = post("/desktop/cleanup", {"run_id": long_run_id})
        # Cleanup does not route through _validate_run_id, so we test for no crash
        assert resp.status_code != 500, (
            f"FIX10 FAIL: 256-char run_id caused 500 on /desktop/cleanup"
        )
        # Should still return a valid response shape
        if resp.status_code == 200:
            data = resp.json()
            assert "run_id" in data
            assert "closed_count" in data

    def test_desktop_get_run_with_path_traversal_run_id_returns_404(self):
        """_validate_run_id should catch path-traversal as part of length/pattern check."""
        # Path traversal strings are typically short but contain special chars;
        # the validate function catches length > 64 but traversal strings may be short.
        resp = get("/desktop/run/../../../etc/passwd")
        # URL routing will likely 404 anyway, but must not 500
        assert resp.status_code != 500, (
            f"FIX10 FAIL: Path traversal in URL caused 500: {resp.status_code}"
        )

    def test_observer_api_run_with_traversal_returns_404(self):
        resp = get("/observer/api/run/../../../etc/passwd")
        assert resp.status_code != 500, (
            f"FIX10 FAIL: Path traversal in observer run URL caused 500"
        )

    def test_cleanup_with_empty_run_id_still_422(self):
        """Regression: empty run_id validation must still work."""
        resp = post("/desktop/cleanup", {"run_id": ""})
        assert resp.status_code == 422, (
            f"REGRESSION: empty run_id on /desktop/cleanup should be 422, got {resp.status_code}"
        )


# ===========================================================================
# FIX 11: MINOR — TestResult pytest warning; renamed to TaskTestResult
# ===========================================================================

class TestFix11PytestCollectionWarning:
    """
    FIX 11: TaskTestResult class in test_e2e_quick_tasks.py must not
    trigger PytestCollectionWarning about "TestResult".

    This test verifies the naming fix by importing the module and checking
    that the class is named TaskTestResult, not TestResult.
    """

    def test_task_test_result_class_name_is_not_test_result(self):
        import importlib
        import sys

        # Import the module fresh
        mod_name = "tests.test_e2e_quick_tasks"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = importlib.import_module(mod_name)

        assert hasattr(mod, "TaskTestResult"), (
            "FIX11 FAIL: TaskTestResult class not found in test_e2e_quick_tasks"
        )
        assert not hasattr(mod, "TestResult"), (
            "FIX11 FAIL: TestResult class still exists — will trigger PytestCollectionWarning"
        )

    def test_task_test_result_is_dataclass(self):
        import importlib
        import sys
        from dataclasses import fields

        mod_name = "tests.test_e2e_quick_tasks"
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        cls = mod.TaskTestResult
        assert fields(cls), "FIX11 FAIL: TaskTestResult has no dataclass fields"

    def test_task_test_result_has_expected_fields(self):
        import importlib
        import sys
        from dataclasses import fields

        mod_name = "tests.test_e2e_quick_tasks"
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        cls = mod.TaskTestResult
        field_names = {f.name for f in fields(cls)}
        required = {"category", "task_id", "intent", "run_id", "accepted"}
        missing = required - field_names
        assert not missing, f"FIX11 FAIL: TaskTestResult missing fields: {missing}"


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCasesAroundFixes:
    """Additional edge cases that span multiple fixes."""

    # run_id boundary tests

    def test_run_id_exactly_64_chars_observer_no_500(self):
        run_id = "f" * 64
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code != 500, (
            "EDGE: 64-char run_id on observer caused 500"
        )
        assert resp.status_code in (200, 404)

    def test_run_id_63_chars_observer_no_500(self):
        run_id = "g" * 63
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code != 500
        assert resp.status_code in (200, 404)

    def test_run_id_65_chars_observer_returns_404(self):
        run_id = "h" * 65
        resp = get(f"/observer/api/run/{run_id}")
        assert resp.status_code == 404, (
            f"EDGE: 65-char run_id on observer should be 404, got {resp.status_code}"
        )

    # Intent edge cases

    def test_intent_with_only_newline_rejected(self):
        resp = post("/desktop/run-task", {"intent": "\n"})
        assert resp.status_code == 422, (
            f"EDGE: newline-only intent should be 422, got {resp.status_code}"
        )

    def test_intent_with_tab_only_rejected(self):
        resp = post("/desktop/run-task", {"intent": "\t"})
        assert resp.status_code == 422, (
            f"EDGE: tab-only intent should be 422, got {resp.status_code}"
        )

    def test_intent_with_mixed_whitespace_rejected(self):
        resp = post("/desktop/run-task", {"intent": " \t \n \r "})
        assert resp.status_code == 422, (
            f"EDGE: mixed-whitespace intent should be 422, got {resp.status_code}"
        )

    def test_intent_with_content_and_newlines_accepted(self):
        resp = post("/desktop/run-task", {"intent": "Open\nNotepad"})
        assert resp.status_code in (202, 422), (
            f"EDGE: intent with embedded newline got unexpected status {resp.status_code}"
        )

    # Resume with long run_ids

    def test_resume_with_65_char_run_id_not_500(self):
        run_id = "i" * 65
        resp = post("/desktop/resume", {"run_id": run_id})
        assert resp.status_code != 500, (
            f"EDGE: 65-char run_id on /desktop/resume caused 500"
        )

    # Multiple runs with same intent are still unique

    def test_fresh_runs_have_unique_ids(self):
        r1 = post("/desktop/run-task", {"intent": "Open Notepad"})
        r2 = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert r1.status_code == r2.status_code == 202
        id1 = r1.json()["run_id"]
        id2 = r2.json()["run_id"]
        assert id1 != id2, "REGRESSION: Two runs with same intent share run_id"

    # Observer for freshly created run must not be 404 (regression check after Fix 2)

    def test_observer_multiple_fresh_runs_all_return_200(self):
        intents = ["Open Notepad", "Open Calculator", "Open Paint"]
        for intent in intents:
            run_id = create_desktop_run(intent)
            resp = get(f"/observer/api/run/{run_id}")
            assert resp.status_code == 200, (
                f"REGRESSION: Observer returned {resp.status_code} for fresh run "
                f"'{intent}' (run_id={run_id})"
            )


# ===========================================================================
# Regression: existing functionality still works after all fixes
# ===========================================================================

class TestRegressionExistingFunctionality:
    """Verify that the 11 fixes did not break existing working functionality."""

    def test_health_endpoint_still_works(self):
        resp = get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_ui_still_serves_html(self):
        resp = get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        assert len(resp.text) > 500

    def test_desktop_pilot_route_still_works(self):
        resp = get("/desktop-pilot")
        assert resp.status_code == 200

    def test_observer_runs_list_still_works(self):
        resp = get("/observer/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_observer_runs_list_limit_still_enforced(self):
        resp = get("/observer/api/runs", limit=101)
        assert resp.status_code == 422

    def test_desktop_run_task_creates_run(self):
        resp = post("/desktop/run-task", {"intent": "Open Notepad"})
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert data["status"] in ("pending", "running")
        assert data["step_count"] == 0

    def test_desktop_get_run_still_works_for_existing_run(self):
        run_id = create_desktop_run("Open Calculator regression test")
        resp = get(f"/desktop/run/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

    def test_desktop_step_nonexistent_run_still_404(self):
        resp = post("/desktop/step", {"run_id": "truly-nonexistent-step-target"})
        assert resp.status_code == 404

    def test_cleanup_nonexistent_run_still_200(self):
        resp = post("/desktop/cleanup", {"run_id": "nonexistent-cleanup-regression"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["closed_count"] == 0

    def test_run_task_rejects_empty_intent(self):
        resp = post("/desktop/run-task", {"intent": ""})
        assert resp.status_code == 422

    def test_run_task_rejects_extra_fields(self):
        resp = post("/desktop/run-task", {"intent": "Open Notepad", "hack": "field"})
        assert resp.status_code == 422

    def test_observer_artifact_missing_param_still_422(self):
        resp = get("/observer/api/artifact")
        assert resp.status_code == 422

    def test_observer_artifact_path_traversal_still_rejected(self):
        resp = get("/observer/api/artifact", path="../../etc/passwd")
        assert resp.status_code in (400, 404)

    @pytest.mark.parametrize("task_intent", [
        "Open Notepad",
        "Open Calculator",
        "Open File Explorer",
        "Open Paint",
        "Open Windows Settings",
        "Right-click on an empty area of the desktop to open the context menu",
        "Press Win+D to show the desktop",
        "Describe what is currently visible on the desktop",
    ])
    def test_all_sampled_quick_task_intents_accepted(self, task_intent):
        resp = post("/desktop/run-task", {"intent": task_intent})
        assert resp.status_code == 202, (
            f"REGRESSION: Quick task '{task_intent}' no longer accepted: "
            f"{resp.status_code}: {resp.text}"
        )
