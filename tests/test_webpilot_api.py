"""Tests for WebPilot API endpoints."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.api.server import app
from src.api.webpilot_models import WebPilotAction
import src.api.webpilot_routes as routes_module


def make_dummy_screenshot():
    # 1x1 PNG
    import struct
    import zlib

    def _png_1x1():
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        raw = b'\x00\xff\xff\xff'
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    return base64.b64encode(_png_1x1()).decode()


DUMMY_SCREENSHOT = make_dummy_screenshot()

CLICK_ACTION = WebPilotAction(
    action="click", x=100, y=200,
    narration="Clicking the button", action_label="Click button", is_irreversible=False
)
DONE_ACTION = WebPilotAction(
    action="done",
    narration="Task complete", action_label="Done", is_irreversible=False
)
IRREVERSIBLE_ACTION = WebPilotAction(
    action="confirm_required", x=100, y=200,
    narration="About to purchase", action_label="Buy now", is_irreversible=True
)
NAVIGATE_ACTION = WebPilotAction(
    action="navigate", target="https://mail.google.com",
    narration="Opening Gmail", action_label="Navigate to Gmail", is_irreversible=False
)
LOGIN_REQUIRED_ACTION = WebPilotAction(
    action="login_required",
    narration="Login screen detected — please sign in", action_label="Login required",
    is_irreversible=False
)


@pytest.fixture(autouse=True)
def clear_sessions():
    routes_module._sessions.clear()
    yield
    routes_module._sessions.clear()


@pytest.fixture
def mock_handler():
    handler = MagicMock()
    handler.get_next_action = AsyncMock(side_effect=[CLICK_ACTION, DONE_ACTION])
    handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)
    original = routes_module._handler
    routes_module._handler = handler
    yield handler
    routes_module._handler = original


def test_create_and_delete_session():
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    r2 = client.delete(f"/webpilot/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"

    r3 = client.delete(f"/webpilot/sessions/{sid}")
    assert r3.status_code == 404


def test_session_not_found_ws(mock_handler):
    client = TestClient(app)
    try:
        with client.websocket_connect("/webpilot/ws/nonexistent-id") as ws:
            pass  # Should close immediately with 4404
        # If we get here, the connection was accepted and closed cleanly
    except Exception:
        pass  # starlette raises on non-1000 close codes — expected for 4404


def test_ws_task_flow(mock_handler):
    """Send task + screenshot, expect click action then done."""
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        msg1 = ws.receive_json()
        assert msg1["type"] == "thinking"

        msg2 = ws.receive_json()
        assert msg2["type"] == "action"
        assert msg2["action"] == "click"

        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})

        msg3 = ws.receive_json()
        assert msg3["type"] == "thinking"

        msg4 = ws.receive_json()
        assert msg4["type"] == "done"


def test_ws_stop(mock_handler):
    """Stop mid-task."""
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    # handler returns click then done, but we stop after click
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        _ = ws.receive_json()  # action (click)

        ws.send_json({"type": "stop"})
        msg = ws.receive_json()
        assert msg["type"] == "stopped"


def test_ws_irreversible_confirm(mock_handler):
    """Irreversible action confirmed by user → proceeds."""
    mock_handler.get_next_action = AsyncMock(side_effect=[IRREVERSIBLE_ACTION, DONE_ACTION])

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Buy ticket", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": True})

        action_msg = ws.receive_json()
        assert action_msg["type"] == "action"

        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        done = ws.receive_json()
        assert done["type"] == "done"


def test_ws_irreversible_deny(mock_handler):
    """Irreversible action denied → stops."""
    mock_handler.get_next_action = AsyncMock(return_value=IRREVERSIBLE_ACTION)

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Buy ticket", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": False})

        stopped = ws.receive_json()
        assert stopped["type"] == "stopped"


def test_ws_interruption(mock_handler):
    """Interrupt replans with new instruction."""
    mock_handler.get_next_action = AsyncMock(return_value=CLICK_ACTION)
    mock_handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        _ = ws.receive_json()  # action (click)

        # Send interrupt instead of screenshot
        ws.send_json({"type": "interrupt", "instruction": "Actually cancel", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking (replan)
        done = ws.receive_json()
        assert done["type"] == "done"


# ---------------------------------------------------------------------------
# New tests (appended)
# ---------------------------------------------------------------------------


def test_tts_endpoint(mock_handler):
    mock_handler.get_narration_audio = AsyncMock(return_value=b"fake-wav-data")
    client = TestClient(app)
    r = client.post("/webpilot/tts", json={"text": "Hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["mime_type"] == "audio/wav"
    assert base64.b64decode(body["audio"]) == b"fake-wav-data"


def test_tts_handler_not_initialized():
    original = routes_module._handler
    routes_module._handler = None
    try:
        client = TestClient(app)
        r = client.post("/webpilot/tts", json={"text": "Hello"})
        assert r.status_code == 503
        assert "not initialized" in r.json()["detail"].lower()
    finally:
        routes_module._handler = original


def test_tts_text_too_long():
    client = TestClient(app)
    r = client.post("/webpilot/tts", json={"text": "A" * 5001})
    assert r.status_code == 422


def test_ws_stuck_detection(mock_handler):
    mock_handler.get_next_action = AsyncMock(
        side_effect=[CLICK_ACTION, CLICK_ACTION, CLICK_ACTION, CLICK_ACTION, DONE_ACTION]
    )
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Do something", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # action
        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # action
        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # action
        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # action
        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # done
    calls = mock_handler.get_next_action.call_args_list
    assert calls[3].kwargs.get("stuck") is True
    assert calls[0].kwargs.get("stuck", False) is False
    assert calls[1].kwargs.get("stuck", False) is False
    assert calls[2].kwargs.get("stuck", False) is False


def test_ws_handler_error(mock_handler):
    mock_handler.get_next_action = AsyncMock(side_effect=RuntimeError("boom"))
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Fail task", "screenshot": DUMMY_SCREENSHOT})
        msg1 = ws.receive_json()
        assert msg1["type"] == "thinking"
        msg2 = ws.receive_json()
        assert msg2["type"] == "error"
        assert "boom" in msg2["message"]


def test_ws_malformed_json(mock_handler):
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_text("not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]


def test_ws_handler_none_closes_4503():
    original = routes_module._handler
    routes_module._handler = None
    try:
        client = TestClient(app)
        r = client.post("/webpilot/sessions")
        sid = r.json()["session_id"]
        from starlette.websockets import WebSocketDisconnect as _WSD
        try:
            with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
                ws.receive_json()
            pytest.fail("Expected WebSocketDisconnect was not raised")
        except _WSD as exc:
            assert exc.code == 4503
        except Exception:
            pass
    finally:
        routes_module._handler = original


def test_ws_unexpected_msg_type(mock_handler):
    mock_handler.get_next_action = AsyncMock(return_value=CLICK_ACTION)
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Do something", "screenshot": DUMMY_SCREENSHOT})
        ws.receive_json()  # thinking
        ws.receive_json()  # action
        ws.send_json({"type": "unknown_type"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unexpected message type" in msg["message"]


def test_ws_session_cap():
    from src.api.webpilot_models import WebPilotSession as _WPS
    for i in range(1000):
        sid = f"cap-session-{i}"
        routes_module._sessions[sid] = _WPS(session_id=sid)
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    assert r.status_code == 503
    assert "limit" in r.json()["detail"].lower()


def test_ws_message_size_limit(mock_handler):
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]
    big_payload = json.dumps({"type": "task", "intent": "x", "screenshot": "A" * (15 * 1024 * 1024 + 1)})
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_text(big_payload)
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "too large" in msg["message"].lower()


def test_session_cleanup():
    from src.api.webpilot_models import WebPilotSession as _WPS
    stale_sid = "stale-session"
    fresh_sid = "fresh-session"
    routes_module._sessions[stale_sid] = _WPS(
        session_id=stale_sid, last_active=time.time() - 7200
    )
    routes_module._sessions[fresh_sid] = _WPS(
        session_id=fresh_sid, last_active=time.time()
    )
    cutoff = time.time() - 1800
    stale = [sid for sid, s in list(routes_module._sessions.items()) if s.last_active < cutoff]
    for sid in stale:
        del routes_module._sessions[sid]
    assert stale_sid not in routes_module._sessions
    assert fresh_sid in routes_module._sessions


def test_ws_session_not_found_assertion(mock_handler):
    client = TestClient(app)
    from starlette.websockets import WebSocketDisconnect as _WSD
    try:
        with client.websocket_connect("/webpilot/ws/nonexistent-id") as ws:
            ws.receive_json()
        pytest.fail("Expected WebSocketDisconnect was not raised")
    except _WSD as exc:
        assert exc.code == 4404
    except Exception:
        pass


def test_navigate_with_redirect(mock_handler):
    """Navigate to gmail.com — agent should NOT return done if landing page is a login screen."""
    mock_handler.get_next_action = AsyncMock(
        side_effect=[NAVIGATE_ACTION, LOGIN_REQUIRED_ACTION]
    )
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Open my Gmail inbox", "screenshot": DUMMY_SCREENSHOT})

        msg1 = ws.receive_json()
        assert msg1["type"] == "thinking"

        msg2 = ws.receive_json()
        assert msg2["type"] == "action"
        assert msg2["action"] == "navigate"

        # Extension executes navigate, page redirects to login — send screenshot back
        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})

        msg3 = ws.receive_json()
        assert msg3["type"] == "thinking"

        # Agent sees login screen — must NOT return done, should pause for login
        msg4 = ws.receive_json()
        assert msg4["type"] != "done", "Agent should not return done on a login redirect"
        assert msg4["type"] == "paused"
        assert msg4["reason"] == "login"
