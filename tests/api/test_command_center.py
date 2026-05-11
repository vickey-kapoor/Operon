"""Tests for the command center SSE stream and empty-state route."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.server import app
    return TestClient(app)


# ── Minimal StepLog fixture ────────────────────────────────────────────────────

def _make_step_log(step_index: int, action_type: str = "click", rule_name: str | None = None) -> dict:
    return {
        "run_id": "test-run-1",
        "step_id": f"step-{step_index}",
        "step_index": step_index,
        "before_artifact_path": f"runs/test-run-1/step_{step_index}/before.png",
        "after_artifact_path": f"runs/test-run-1/step_{step_index}/after.png",
        "perception_debug": {
            "prompt_artifact_path": "p.txt",
            "raw_response_artifact_path": "r.txt",
            "parsed_artifact_path": "parsed.json",
        },
        "policy_debug": {
            "prompt_artifact_path": "p.txt",
            "raw_response_artifact_path": "r.txt",
            "parsed_artifact_path": "parsed.json",
        },
        "perception": {
            "elements": [],
            "page_hint": "test_page",
            "focused_element": None,
            "raw_response": "",
            "quality_score": 1.0,
        },
        "policy_decision": {
            "action": {"action_type": action_type, "x": 100, "y": 200},
            "rationale": "clicking the button",
            "confidence": 0.85,
            "active_subgoal": "click button",
            "expected_change": "content",
            "rule_name": rule_name,
        },
        "executed_action": {
            "action": {"action_type": action_type, "x": 100, "y": 200},
            "success": True,
            "error_message": None,
            "execution_trace": {"attempts": [], "final_status": "success"},
        },
        "verification_result": {
            "success": True,
            "critic_verdict": True,
            "terminal_state": None,
            "stop_reason": None,
            "recovery_hint": None,
        },
        "recovery_decision": {
            "action": "continue",
            "reason": "ok",
        },
        "failure": None,
    }


# ── SSE route smoke tests ──────────────────────────────────────────────────────

def test_sse_invalid_run_id_returns_404(client: TestClient):
    """SSE endpoint rejects run_ids with path traversal characters."""
    resp = client.get("/command-center/api/run/../secret/stream")
    assert resp.status_code == 404


def test_step_log_to_event_replays_two_steps():
    """_step_log_to_event correctly maps two sequential StepLog records."""
    from src.api.routes import _step_log_to_event
    step1 = _make_step_log(1, "click")
    step2 = _make_step_log(2, "navigate")
    ev1 = _step_log_to_event(step1)
    ev2 = _step_log_to_event(step2)
    assert ev1["n"] == 1 and ev1["action"] == "click"
    assert ev1["source"] == "LLM"
    assert ev1["conf"] == pytest.approx(0.85)
    assert ev1["status"] == "done"
    assert ev2["n"] == 2 and ev2["action"] == "navigate"


def test_step_log_to_event_rule_source():
    """_step_log_to_event uses rule_name as source when set."""
    from src.api.routes import _step_log_to_event
    step = _make_step_log(1, "click", rule_name="_search_query_rule")
    ev = _step_log_to_event(step)
    assert ev["source"] == "_search_query_rule"


def test_sse_step_log_to_event_failed_status():
    """_step_log_to_event marks a step with failure as 'failed'."""
    from src.api.routes import _step_log_to_event
    record = _make_step_log(3)
    record["failure"] = {"category": "execution_error", "stage": "execution",
                         "retry_count": 1, "terminal": False, "recoverable": True,
                         "reason": "target shifted"}
    ev = _step_log_to_event(record)
    assert ev is not None
    assert ev["status"] == "failed"
    assert ev["n"] == 3


def test_sse_pre_step_failure_log():
    """_step_log_to_event handles PreStepFailureLog shape."""
    from src.api.routes import _step_log_to_event
    record = {
        "record_type": "pre_step_failure",
        "run_id": "r1",
        "step_id": "s1",
        "step_index": 2,
        "before_artifact_path": "before.png",
        "perception_debug": {},
        "failure": {},
        "error_message": "perception timed out",
    }
    ev = _step_log_to_event(record)
    assert ev is not None
    assert ev["n"] == 2
    assert ev["action"] == "error"
    assert ev["status"] == "failed"
    assert "perception timed out" in ev["target"]


# ── Empty state route ──────────────────────────────────────────────────────────

def test_command_center_empty_state_renders(client: TestClient):
    """GET /command-center (no run_id) serves the HTML with empty state elements."""
    resp = client.get("/command-center")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Empty state CTA
    assert "Start a new task" in resp.text
    assert "new task" in resp.text
    # Browser-dominant layout landmarks
    assert "emptyContent" in resp.text
    assert "sidebar" in resp.text
    assert "browserSection" in resp.text


def test_command_center_with_run_id_serves_same_html(client: TestClient):
    """GET /command-center/{run_id} serves the same HTML (run_id is client-side state)."""
    resp = client.get("/command-center/abc-123")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "command-center" in resp.text.lower()
