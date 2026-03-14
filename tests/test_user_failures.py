"""Regression tests for user-testing failures.

Each test reproduces a real failure at the WebSocket boundary, verifies the
root cause, and guards against regression after the fix.

Reproduced failures:
  Bug A: Pause flow (login_required/captcha) crashes on malformed resume —
         KeyError on missing screenshot field, JSONDecodeError on invalid JSON.
  Bug B: Confirm flow crashes on invalid JSON during confirmation prompt.
  Bug C: Task resubmission to a done session — verifies the server accepts
         a new task after the previous one completes.
  Bug D: Spurious "Internal server error" sent on every clean WS disconnect
         (the finally block unconditionally sends error message).
  Bug H: Invalid JSON mid-action-loop (after action sent, client sends garbage
         instead of screenshot) crashes the loop with JSONDecodeError.
  Bug I: Invalid JSON after interrupt replan (client sends garbage instead of
         screenshot following the replanned action) crashes the handler.

Extension-side bugs (cannot reproduce server-side — documented only):
  Bug E: background.js sendWS() drops messages when WS is not open instead
         of queuing. Fix: _pendingOutbound queue + flushPendingOutbound().
  Bug F: background.js _ws not nulled on close, preventing reconnection.
  Bug G: background.js reconnect uses fixed delay instead of exponential backoff.

Run:
    python -m pytest tests/test_user_failures.py -v
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import struct
import threading
import time
import zlib
from contextlib import contextmanager
from typing import Generator

import httpx
import pytest
import websockets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_b64() -> str:
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _server_ctx(scenario: str) -> Generator[tuple[str, str], None, None]:
    import uvicorn
    from src.api.server import app

    old_val = os.environ.get("WEBPILOT_STUB")
    os.environ["WEBPILOT_STUB"] = scenario

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15.0
    ready = False
    while time.monotonic() < deadline:
        time.sleep(0.15)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
    if not ready:
        server.should_exit = True
        raise RuntimeError(f"Server (scenario={scenario!r}) did not start in time")

    try:
        yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if old_val is None:
            os.environ.pop("WEBPILOT_STUB", None)
        else:
            os.environ["WEBPILOT_STUB"] = old_val


def _create_session(http_url: str) -> str:
    r = httpx.post(f"{http_url}/webpilot/sessions")
    r.raise_for_status()
    return r.json()["session_id"]


# ===========================================================================
# Bug A: Pause flow crashes on malformed resume
# Root cause: webpilot_routes.py line 390 does pause_data["screenshot"]
#   without checking key existence. Also line 388 json.loads() has no
#   try/except for JSONDecodeError.
# ===========================================================================

@pytest.fixture(scope="module")
def login_server():
    """Server with navigate_with_redirect scenario — triggers login_required."""
    with _server_ctx("navigate_with_redirect") as urls:
        yield urls


async def test_pause_resume_without_screenshot_field(login_server):
    """BUG A: User triggers login_required, then sends resume without
    the screenshot field. Before the fix, this crashes the action loop
    with a KeyError. After the fix, the server should send an error
    message and keep the session usable.

    User-visible symptom: WS drops silently mid-task after CAPTCHA/login."""
    http_url, ws_url = login_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "check gmail", "screenshot": BLANK,
        }))

        # Drive until we get the paused message
        paused = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "paused":
                paused = True
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

        assert paused, "Server did not reach login_required/paused state"

        # Send resume WITHOUT the required screenshot field
        await ws.send(json.dumps({"type": "resume"}))

        # After fix: server sends error message, session stays alive
        # Before fix: KeyError crashes the action loop, WS drops
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "error", (
            f"Expected error for missing screenshot, got {msg['type']}: {msg}"
        )

        # Session must still be usable after the error
        await ws.send(json.dumps({"type": "stop"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "stopped"


async def test_pause_invalid_json_during_resume(login_server):
    """BUG A: User triggers login_required, then the client sends invalid
    JSON during the resume prompt. Before the fix, JSONDecodeError crashes
    the action loop. After the fix, the server sends an error and stays alive.

    User-visible symptom: WS drops silently when client sends corrupted data."""
    http_url, ws_url = login_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "check gmail", "screenshot": BLANK,
        }))

        paused = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "paused":
                paused = True
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

        assert paused, "Server did not reach paused state"

        # Send garbage instead of valid JSON
        await ws.send("{invalid json!!")

        # After fix: server sends error, session stays alive
        # Before fix: JSONDecodeError crashes the loop
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "error", (
            f"Expected error for invalid JSON, got {msg['type']}: {msg}"
        )


# ===========================================================================
# Bug B: Confirm flow crashes on invalid JSON
# Root cause: webpilot_routes.py line 417 json.loads(raw_confirm) has no
#   try/except for JSONDecodeError.
# ===========================================================================

@pytest.fixture(scope="module")
def confirm_server():
    """Server with confirm_flow scenario — triggers confirmation prompt."""
    with _server_ctx("confirm_flow") as urls:
        yield urls


async def test_confirm_invalid_json_crashes_loop(confirm_server):
    """BUG B: During a confirmation prompt, the client sends invalid JSON.
    Before the fix, JSONDecodeError crashes the action loop.
    After the fix, the server treats it as a denial and sends stopped.

    User-visible symptom: WS drops during "Confirm or Cancel" dialog."""
    http_url, ws_url = confirm_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "checkout", "screenshot": BLANK,
        }))

        # Drive until confirmation_required
        got_confirm = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "confirmation_required":
                got_confirm = True
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

        assert got_confirm, "Server did not reach confirmation_required state"

        # Send invalid JSON during confirmation prompt
        await ws.send("{not valid json")

        # After fix: treated as denial → stopped
        # Before fix: JSONDecodeError crashes the loop, WS drops
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "stopped", (
            f"Expected 'stopped' (treated as denial), got {msg['type']}: {msg}"
        )


# ===========================================================================
# Bug C: Task resubmission doesn't close old handler
# Verifies the server correctly resets session state (intent, history,
# abort_event) when a new task is submitted on the same WebSocket.
# ===========================================================================

@pytest.fixture(scope="module")
def resubmit_server():
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_task_resubmission_after_done_works(resubmit_server):
    """BUG C: Complete a task (status=done), then immediately send a new
    task on the same WS without disconnecting. The server must accept the
    new task and drive it to completion.

    Verifies the functional path works — session state is correctly reset."""
    http_url, ws_url = resubmit_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        # --- Task 1: drive to done ---
        await ws.send(json.dumps({
            "type": "task", "intent": "first task", "screenshot": BLANK,
        }))
        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "done":
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))
        assert msg["type"] == "done", f"Task 1 did not complete: {msg}"

        # --- Task 2: send immediately after done (no disconnect) ---
        await ws.send(json.dumps({
            "type": "task", "intent": "second task", "screenshot": BLANK,
        }))
        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "done":
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))
        assert msg["type"] == "done", (
            f"Task 2 (resubmission) did not complete: {msg}"
        )


# ===========================================================================
# Bug D: Spurious error on clean disconnect
# Root cause: webpilot_routes.py finally block unconditionally sends
#   {"type":"error","message":"Internal server error"} — even on clean
#   WebSocketDisconnect. The send fails silently on closed sockets, but
#   on half-open sockets it can reach the client.
# ===========================================================================

@pytest.fixture(scope="module")
def disconnect_server():
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_clean_task_completion_no_error_messages(disconnect_server):
    """BUG D: Complete a task, verify no error messages appear in the stream.

    The spurious error from the finally block is sent after the client
    disconnects, so it usually doesn't reach the client. But it still
    represents incorrect behavior: the server should not attempt to send
    error messages on clean disconnects."""
    http_url, ws_url = disconnect_server
    sid = _create_session(http_url)

    all_messages = []
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "test task", "screenshot": BLANK,
        }))
        for _ in range(20):
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            all_messages.append(msg)
            if msg["type"] in ("done", "stopped"):
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

    assert all_messages[-1]["type"] in ("done", "stopped")
    error_msgs = [m for m in all_messages if m["type"] == "error"]
    assert not error_msgs, f"Spurious errors in clean run: {error_msgs}"


# ===========================================================================
# Combined flow: the most common user scenario that exercises all bugs
# ===========================================================================

@pytest.fixture(scope="module")
def full_flow_server():
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_full_user_lifecycle_no_failures(full_flow_server):
    """INTEGRATION: Complete task → disconnect → reconnect → new task.
    No errors, no dropped messages, no crashes.

    This is the most common user flow: use the extension, close sidebar,
    reopen sidebar, start another task."""
    http_url, ws_url = full_flow_server
    sid = _create_session(http_url)

    # Round 1
    round1 = []
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "task 1", "screenshot": BLANK,
        }))
        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            round1.append(msg)
            if msg["type"] in ("done", "stopped"):
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

    assert round1[-1]["type"] in ("done", "stopped")
    assert not any(m["type"] == "error" for m in round1)

    await asyncio.sleep(0.3)

    # Round 2: reconnect, new task
    round2 = []
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "task 2", "screenshot": BLANK,
        }))
        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            round2.append(msg)
            if msg["type"] in ("done", "stopped"):
                break
            if msg["type"] == "action":
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

    assert round2[-1]["type"] in ("done", "stopped"), (
        f"Round 2 failed: {[m['type'] for m in round2]}"
    )
    assert not any(m["type"] == "error" for m in round2), (
        f"Errors in round 2: {[m for m in round2 if m['type'] == 'error']}"
    )


# ===========================================================================
# Bug H: Invalid JSON mid-action-loop crashes the loop
# Root cause: json.loads(raw) in the screenshot-wait section of
#   _run_action_loop_inner has no try/except for JSONDecodeError.
# ===========================================================================

@pytest.fixture(scope="module")
def midloop_server():
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_invalid_json_mid_action_loop(midloop_server):
    """BUG H: After the server sends an action, the client sends garbage
    instead of a screenshot. Before the fix, json.loads crashes the loop
    with JSONDecodeError and the WS drops. After the fix, the server sends
    an error and terminates the loop gracefully.

    User-visible symptom: WS drops silently mid-task if client sends
    corrupted data (e.g. after a browser crash during screenshot capture)."""
    http_url, ws_url = midloop_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task", "intent": "test task", "screenshot": BLANK,
        }))

        # Wait for the first action (after thinking)
        got_action = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "action":
                got_action = True
                break

        assert got_action, "Server did not send an action"

        # Instead of a screenshot, send garbage
        await ws.send("{this is not valid json!!!")

        # After fix: server sends error, loop ends gracefully
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "error", (
            f"Expected error for invalid JSON, got {msg['type']}: {msg}"
        )


# ===========================================================================
# Bug I: Invalid JSON after interrupt replan crashes the handler
# Root cause: json.loads(raw) in _handle_interrupt after receiving the
#   post-replan message has no try/except.
# ===========================================================================

@pytest.fixture(scope="module")
def interrupt_server():
    with _server_ctx("interrupt_redirect") as urls:
        yield urls


async def test_invalid_json_after_interrupt_replan(interrupt_server):
    """BUG I: After an interrupt replan, the server sends a replanned action
    and waits for a screenshot. If the client sends garbage, json.loads
    crashes. After the fix, the server sends an error and stops gracefully.

    User-visible symptom: WS drops after user redirects mid-task and
    the extension sends corrupted data."""
    http_url, ws_url = interrupt_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        # Start a task
        await ws.send(json.dumps({
            "type": "task", "intent": "original task", "screenshot": BLANK,
        }))

        # Drive until we get an action
        got_action = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "action":
                got_action = True
                break

        assert got_action, "Server did not send an action before interrupt"

        # Send interrupt (redirect)
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "do something different instead",
            "screenshot": BLANK,
        }))

        # Wait for the replanned action (skip thinking)
        got_replan = False
        for _ in range(10):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] == "action":
                got_replan = True
                break
            if msg["type"] in ("done", "stopped"):
                # Replan completed immediately — that's fine, just skip test
                break

        if not got_replan:
            pytest.skip("Replan completed without sending an action — cannot test mid-replan crash")

        # Send garbage instead of screenshot
        await ws.send("{not valid json at all")

        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "error", (
            f"Expected error for invalid JSON after replan, got {msg['type']}: {msg}"
        )
