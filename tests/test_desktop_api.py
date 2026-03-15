"""Integration tests for Desktop Mode API endpoints.

Uses DESKTOP_MOCK=true and a mock handler (no Gemini calls).

Run:
    python -m pytest tests/test_desktop_api.py -v
"""
from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Set env vars before importing anything from the app
os.environ["DESKTOP_MOCK"] = "true"
os.environ["DESKTOP_MODE_ENABLED"] = "true"

from src.api.server import app  # noqa: E402
from src.api.desktop_models import DesktopAction  # noqa: E402
import src.api.desktop_routes as desktop_routes  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_b64() -> str:
    """Return a base64-encoded 1x1 white PNG."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return base64.b64encode(sig + ihdr + idat + iend).decode()


BLANK = _blank_b64()

# Pre-built actions
CLICK_ACTION = DesktopAction(
    action="click", x=100, y=200,
    narration="Click button", action_label="Click", is_irreversible=False,
)
DONE_ACTION = DesktopAction(
    action="done",
    narration="Task complete", action_label="Done", is_irreversible=False,
)
CONFIRM_ACTION = DesktopAction(
    action="confirm_required", x=100, y=200,
    narration="About to delete", action_label="Delete file", is_irreversible=True,
)
RIGHT_CLICK_ACTION = DesktopAction(
    action="right_click", x=300, y=400,
    narration="Right-clicking", action_label="Right click", is_irreversible=False,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_sessions():
    desktop_routes._sessions.clear()
    yield
    desktop_routes._sessions.clear()


@pytest.fixture
def mock_handler():
    """Inject a mock handler that returns click → done by default."""
    handler = MagicMock()
    handler.get_next_action_desktop = AsyncMock(side_effect=[CLICK_ACTION, DONE_ACTION])
    handler.get_next_action = AsyncMock(side_effect=[CLICK_ACTION, DONE_ACTION])
    handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)
    handler.classify_interruption_type = MagicMock()
    original = desktop_routes._desktop_handler
    desktop_routes._desktop_handler = handler
    yield handler
    desktop_routes._desktop_handler = original


# ---------------------------------------------------------------------------
# REST: POST /desktop/sessions
# ---------------------------------------------------------------------------

def test_create_session_returns_200(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    assert r.status_code == 200
    assert "session_id" in r.json()


def test_create_session_503_without_env_var(mock_handler):
    """If DESKTOP_MODE_ENABLED is not set, should return 503."""
    old = os.environ.get("DESKTOP_MODE_ENABLED")
    os.environ["DESKTOP_MODE_ENABLED"] = "false"
    try:
        client = TestClient(app)
        r = client.post("/desktop/sessions")
        assert r.status_code == 503
        assert "not enabled" in r.json()["detail"].lower()
    finally:
        if old is not None:
            os.environ["DESKTOP_MODE_ENABLED"] = old
        else:
            os.environ.pop("DESKTOP_MODE_ENABLED", None)


# ---------------------------------------------------------------------------
# REST: DELETE /desktop/sessions/{id}
# ---------------------------------------------------------------------------

def test_delete_session_200(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]
    r2 = client.delete(f"/desktop/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"


def test_delete_session_404(mock_handler):
    client = TestClient(app)
    r = client.delete("/desktop/sessions/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# REST: GET /desktop/sessions/{id}
# ---------------------------------------------------------------------------

def test_get_session_status(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]
    r2 = client.get(f"/desktop/sessions/{sid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["session_id"] == sid
    assert body["status"] == "idle"


def test_get_session_404(mock_handler):
    client = TestClient(app)
    r = client.get("/desktop/sessions/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# WS: session not found
# ---------------------------------------------------------------------------

def test_ws_invalid_session_closes_4404(mock_handler):
    client = TestClient(app)
    try:
        with client.websocket_connect("/desktop/ws/nonexistent-id") as ws:
            pass
    except Exception:
        pass  # Expected: starlette raises on non-1000 close code


# ---------------------------------------------------------------------------
# WS: full task → action → screenshot → done flow
# ---------------------------------------------------------------------------

def test_ws_task_flow(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Open Notepad", "screenshot": BLANK})

        msg1 = ws.receive_json()
        assert msg1["type"] == "thinking"

        msg2 = ws.receive_json()
        assert msg2["type"] == "action"
        assert msg2["action"] == "click"

        ws.send_json({"type": "screenshot", "screenshot": BLANK})

        msg3 = ws.receive_json()
        assert msg3["type"] == "thinking"

        msg4 = ws.receive_json()
        assert msg4["type"] == "done"


# ---------------------------------------------------------------------------
# WS: stop mid-task
# ---------------------------------------------------------------------------

def test_ws_stop(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Do something", "screenshot": BLANK})

        ws.receive_json()  # thinking
        ws.receive_json()  # action

        ws.send_json({"type": "stop"})
        msg = ws.receive_json()
        assert msg["type"] == "stopped"


# ---------------------------------------------------------------------------
# WS: interrupt — ABORT
# ---------------------------------------------------------------------------

def test_ws_interrupt_abort(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(return_value=CLICK_ACTION)

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Do something", "screenshot": BLANK})
        ws.receive_json()  # thinking
        ws.receive_json()  # action

        # Send interrupt with abort keyword during screenshot phase
        ws.send_json({"type": "interrupt", "instruction": "stop", "screenshot": BLANK})
        msg = ws.receive_json()
        assert msg["type"] == "stopped"


# ---------------------------------------------------------------------------
# WS: interrupt — REDIRECT
# ---------------------------------------------------------------------------

def test_ws_interrupt_redirect(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(return_value=CLICK_ACTION)
    mock_handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Do A", "screenshot": BLANK})
        ws.receive_json()  # thinking
        ws.receive_json()  # action

        ws.send_json({"type": "interrupt", "instruction": "Actually do B instead", "screenshot": BLANK})
        msg = ws.receive_json()
        assert msg["type"] == "thinking"
        msg2 = ws.receive_json()
        assert msg2["type"] == "done"


# ---------------------------------------------------------------------------
# WS: interrupt — REFINEMENT
# ---------------------------------------------------------------------------

def test_ws_interrupt_refinement(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(return_value=CLICK_ACTION)
    mock_handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Open file", "screenshot": BLANK})
        ws.receive_json()  # thinking
        ws.receive_json()  # action

        ws.send_json({"type": "interrupt", "instruction": "make it larger font", "screenshot": BLANK})
        msg = ws.receive_json()
        assert msg["type"] == "thinking"
        msg2 = ws.receive_json()
        assert msg2["type"] == "done"


# ---------------------------------------------------------------------------
# WS: confirm_required — approve
# ---------------------------------------------------------------------------

def test_ws_confirm_approve(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(side_effect=[CONFIRM_ACTION, DONE_ACTION])

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Delete file", "screenshot": BLANK})

        ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": True})

        action_msg = ws.receive_json()
        assert action_msg["type"] == "action"

        ws.send_json({"type": "screenshot", "screenshot": BLANK})

        ws.receive_json()  # thinking
        done = ws.receive_json()
        assert done["type"] == "done"


# ---------------------------------------------------------------------------
# WS: confirm_required — deny
# ---------------------------------------------------------------------------

def test_ws_confirm_deny(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(return_value=CONFIRM_ACTION)

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Delete file", "screenshot": BLANK})

        ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": False})

        stopped = ws.receive_json()
        assert stopped["type"] == "stopped"


# ---------------------------------------------------------------------------
# WS: step budget exhaustion
# ---------------------------------------------------------------------------

def test_ws_step_budget_exhaustion(mock_handler):
    # Always return click, never done — should hit step limit
    mock_handler.get_next_action_desktop = AsyncMock(return_value=CLICK_ACTION)

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    # Temporarily lower step limit for speed
    old_max = desktop_routes._MAX_LOOP_STEPS
    desktop_routes._MAX_LOOP_STEPS = 3
    try:
        with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
            ws.send_json({"type": "task", "intent": "Loop forever", "screenshot": BLANK})

            for _ in range(3):
                ws.receive_json()  # thinking
                ws.receive_json()  # action
                ws.send_json({"type": "screenshot", "screenshot": BLANK})

            # After 3 steps, should get stopped
            msg = ws.receive_json()
            assert msg["type"] == "stopped"
            assert "maximum" in msg.get("narration", "").lower()
    finally:
        desktop_routes._MAX_LOOP_STEPS = old_max


# ---------------------------------------------------------------------------
# WS: handler error
# ---------------------------------------------------------------------------

def test_ws_handler_error(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(side_effect=RuntimeError("boom"))

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Fail", "screenshot": BLANK})

        ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "boom" in msg["message"]


# ---------------------------------------------------------------------------
# WS: invalid JSON resilience
# ---------------------------------------------------------------------------

def test_ws_invalid_json(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_text("not valid json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]


# ---------------------------------------------------------------------------
# WS: handler not initialized
# ---------------------------------------------------------------------------

def test_ws_handler_not_initialized_closes_4503():
    original = desktop_routes._desktop_handler
    desktop_routes._desktop_handler = None
    try:
        client = TestClient(app)
        r = client.post("/desktop/sessions")
        sid = r.json()["session_id"]
        try:
            with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
                pass
        except Exception:
            pass  # Expected: close with 4503
    finally:
        desktop_routes._desktop_handler = original


# ---------------------------------------------------------------------------
# WS: stop message on outer loop
# ---------------------------------------------------------------------------

def test_ws_stop_outer_loop(mock_handler):
    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        ws.send_json({"type": "stop"})
        msg = ws.receive_json()
        assert msg["type"] == "stopped"


# ---------------------------------------------------------------------------
# WS: new task after done
# ---------------------------------------------------------------------------

def test_ws_new_task_after_done(mock_handler):
    mock_handler.get_next_action_desktop = AsyncMock(side_effect=[
        DONE_ACTION,
        CLICK_ACTION, DONE_ACTION,
    ])

    client = TestClient(app)
    r = client.post("/desktop/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/desktop/ws/{sid}") as ws:
        # First task — immediate done
        ws.send_json({"type": "task", "intent": "Task 1", "screenshot": BLANK})
        ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "done"

        # Second task in same session
        ws.send_json({"type": "task", "intent": "Task 2", "screenshot": BLANK})
        ws.receive_json()  # thinking
        msg2 = ws.receive_json()
        assert msg2["type"] == "action"

        ws.send_json({"type": "screenshot", "screenshot": BLANK})
        ws.receive_json()  # thinking
        msg3 = ws.receive_json()
        assert msg3["type"] == "done"


# ---------------------------------------------------------------------------
# Health endpoint includes desktop_sessions_active
# ---------------------------------------------------------------------------

def test_health_includes_desktop_sessions():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "desktop_sessions_active" in body
    assert body["desktop_sessions_active"] == 0
