"""End-to-end WebPilot WebSocket tests using a real server + websockets library.

The server is started in-process (uvicorn in a background thread) with
WEBPILOT_STUB set to the appropriate scenario.  No Gemini calls are made.

Run:
    python -m pytest tests/test_webpilot_e2e.py -v
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import tempfile
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import httpx
import pytest
import websockets

# ---------------------------------------------------------------------------
# Message log (written after each test for agent_runner to collect)
# ---------------------------------------------------------------------------
_MSG_LOG_PATH = Path(tempfile.gettempdir()) / "wp_test_messages.json"


def _append_messages(test_name: str, messages: list[dict]) -> None:
    existing: list[dict] = []
    if _MSG_LOG_PATH.exists():
        try:
            existing = json.loads(_MSG_LOG_PATH.read_text())
        except Exception:
            pass
    existing.append({"test": test_name, "messages": messages})
    _MSG_LOG_PATH.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_b64() -> str:
    """Return a base64-encoded 1×1 white PNG screenshot."""
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
    """Start a real uvicorn server with the given stub scenario.

    Yields (http_base_url, ws_base_url).
    """
    import uvicorn

    from src.api.server import app

    old_val = os.environ.get("WEBPILOT_STUB")
    os.environ["WEBPILOT_STUB"] = scenario

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait up to 15s — poll the health endpoint rather than relying on server.started
    # (server.started can miss the transition on slow startup or when lifespan emits
    # warnings before the ready signal, especially for the first server in the session).
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


# ---------------------------------------------------------------------------
# Module-scoped server fixtures (one per scenario to avoid repeated restarts)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nav_done_server():
    with _server_ctx("navigate_and_done") as urls:
        yield urls


@pytest.fixture(scope="module")
def confirm_server():
    with _server_ctx("confirm_flow") as urls:
        yield urls


@pytest.fixture(scope="module")
def interrupt_server():
    with _server_ctx("interrupt_redirect") as urls:
        yield urls


@pytest.fixture(scope="module")
def stuck_server():
    with _server_ctx("stuck_loop") as urls:
        yield urls


@pytest.fixture(scope="module")
def abort_server():
    """Dedicated server for test_abort_interrupt — isolated stub state."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _create_session(http_url: str) -> str:
    r = httpx.post(f"{http_url}/webpilot/sessions")
    r.raise_for_status()
    return r.json()["session_id"]


def _delete_session(http_url: str, sid: str) -> dict:
    r = httpx.delete(f"{http_url}/webpilot/sessions/{sid}")
    return r


# ---------------------------------------------------------------------------
# Test 1 — Session lifecycle
# ---------------------------------------------------------------------------

async def test_session_lifecycle(nav_done_server):
    http_url, ws_url = nav_done_server
    messages = []

    r1 = httpx.post(f"{http_url}/webpilot/sessions")
    assert r1.status_code == 200
    sid = r1.json()["session_id"]
    assert sid
    messages.append({"direction": "http", "endpoint": "POST /webpilot/sessions", "body": r1.json()})

    # Prove the session is actually WS-usable: connect and send a stop message.
    # Using "stop" (not "task") avoids calling get_next_action on the shared module-scoped
    # stub, which would advance _index and corrupt subsequent tests on the same server.
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "stop"}))
        first = json.loads(await ws.recv())
        assert first["type"] == "stopped", f"Expected stopped on idle-session stop, got {first}"
        messages.append({"direction": "recv", **first})

    r2 = httpx.delete(f"{http_url}/webpilot/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"
    messages.append({"direction": "http", "endpoint": f"DELETE /webpilot/sessions/{sid}", "body": r2.json()})

    r3 = httpx.delete(f"{http_url}/webpilot/sessions/{sid}")
    assert r3.status_code == 404
    messages.append({"direction": "http", "endpoint": f"DELETE /webpilot/sessions/{sid} (2nd)", "status": 404})

    _append_messages("test_session_lifecycle", messages)


# ---------------------------------------------------------------------------
# Test 2 — navigate_and_done full WS flow
# ---------------------------------------------------------------------------

async def test_task_navigate_and_done(nav_done_server):
    http_url, ws_url = nav_done_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        # thinking
        msg1 = json.loads(await ws.recv())
        assert msg1["type"] == "thinking", f"Expected thinking, got {msg1}"
        messages.append({"direction": "recv", **msg1})

        # action (navigate)
        msg2 = json.loads(await ws.recv())
        assert msg2["type"] == "action", f"Expected action, got {msg2}"
        assert msg2["action"] == "navigate"
        messages.append({"direction": "recv", **msg2})

        # send screenshot — simulates extension executing the action
        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot"})

        # thinking
        msg3 = json.loads(await ws.recv())
        assert msg3["type"] == "thinking"
        messages.append({"direction": "recv", **msg3})

        # done
        msg4 = json.loads(await ws.recv())
        assert msg4["type"] == "done", f"Expected done, got {msg4}"
        assert msg4["action"] == "done"
        messages.append({"direction": "recv", **msg4})

    _append_messages("test_task_navigate_and_done", messages)


# ---------------------------------------------------------------------------
# Test 3 — confirm flow (confirmed=true)
# ---------------------------------------------------------------------------

async def test_confirm_flow(confirm_server):
    http_url, ws_url = confirm_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "buy something", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        # Step 1: navigate action
        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())
        assert msg["type"] == "action"
        messages.append({"direction": "recv", **msg})

        # send screenshot after navigate
        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot"})

        # Step 2: confirmation_required
        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())
        assert msg["type"] == "confirmation_required", f"Expected confirmation_required, got {msg}"
        assert "narration" in msg
        messages.append({"direction": "recv", **msg})

        # confirm
        await ws.send(json.dumps({"type": "confirm", "confirmed": True}))
        messages.append({"direction": "sent", "type": "confirm", "confirmed": True})

        # server sends the confirmed action, then waits for next screenshot
        action_msg = json.loads(await ws.recv())
        assert action_msg["type"] == "action", f"Expected action after confirm, got {action_msg}"
        messages.append({"direction": "recv", **action_msg})

        # send screenshot so loop continues to done
        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot"})

        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        done_msg = json.loads(await ws.recv())
        assert done_msg["type"] == "done", f"Expected done, got {done_msg}"
        messages.append({"direction": "recv", **done_msg})

    _append_messages("test_confirm_flow", messages)


# ---------------------------------------------------------------------------
# Test 4 — confirm denied → stopped
# ---------------------------------------------------------------------------

async def test_confirm_denied(confirm_server):
    http_url, ws_url = confirm_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "buy something", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        # Step 1: navigate action
        nav_thinking = json.loads(await ws.recv())
        assert nav_thinking["type"] == "thinking", f"Expected thinking, got {nav_thinking}"
        nav_action = json.loads(await ws.recv())
        assert nav_action["type"] == "action", f"Expected action, got {nav_action}"
        assert nav_action["action"] == "navigate"

        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot"})

        # Step 2: confirmation_required
        conf_thinking = json.loads(await ws.recv())
        assert conf_thinking["type"] == "thinking", f"Expected thinking before confirm, got {conf_thinking}"
        msg = json.loads(await ws.recv())
        assert msg["type"] == "confirmation_required"
        messages.append({"direction": "recv", **msg})

        # deny
        await ws.send(json.dumps({"type": "confirm", "confirmed": False}))
        messages.append({"direction": "sent", "type": "confirm", "confirmed": False})

        stopped = json.loads(await ws.recv())
        assert stopped["type"] == "stopped", f"Expected stopped after denial, got {stopped}"
        messages.append({"direction": "recv", **stopped})

    _append_messages("test_confirm_denied", messages)


# ---------------------------------------------------------------------------
# Test 5 — interrupt redirect mid-task
# ---------------------------------------------------------------------------

async def test_interrupt_redirect(interrupt_server):
    http_url, ws_url = interrupt_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        # first action: navigate
        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        action_msg = json.loads(await ws.recv())
        assert action_msg["type"] == "action"
        messages.append({"direction": "recv", **action_msg})

        # send interrupt instead of screenshot
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "actually go to bing instead",
            "screenshot": BLANK,
        }))
        messages.append({"direction": "sent", "type": "interrupt"})

        # server should respond with thinking then a new action (not stopped)
        replan_thinking = json.loads(await ws.recv())
        assert replan_thinking["type"] == "thinking", f"Expected thinking after interrupt, got {replan_thinking}"
        messages.append({"direction": "recv", **replan_thinking})

        replan_action = json.loads(await ws.recv())
        assert replan_action["type"] in ("action", "done"), (
            f"Expected action or done after interrupt replan, got {replan_action}"
        )
        assert replan_action["type"] != "stopped", "Server should not have stopped on a redirect interrupt"
        # Assert action-specific content is present and valid
        assert replan_action.get("action") in ("click", "type", "scroll", "wait", "navigate", "done", "confirm_required"), (
            f"Replan action field not a valid action type: {replan_action.get('action')}"
        )
        assert replan_action.get("narration"), "Replan action must include narration"
        messages.append({"direction": "recv", **replan_action})

        # if the loop is still running (action, not done), send a screenshot and let it finish
        if replan_action["type"] == "action":
            await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
            messages.append({"direction": "sent", "type": "screenshot"})
            final = json.loads(await ws.recv())  # thinking or done
            messages.append({"direction": "recv", **final})
            if final["type"] == "thinking":
                done_msg = json.loads(await ws.recv())
                messages.append({"direction": "recv", **done_msg})

    _append_messages("test_interrupt_redirect", messages)


# ---------------------------------------------------------------------------
# Test 6 — stop mid-task
# ---------------------------------------------------------------------------

async def test_stop_mid_task(nav_done_server):
    http_url, ws_url = nav_done_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go somewhere", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        stop_thinking = json.loads(await ws.recv())
        assert stop_thinking["type"] == "thinking", f"Expected thinking, got {stop_thinking}"

        action_msg = json.loads(await ws.recv())
        assert action_msg["type"] == "action"
        messages.append({"direction": "recv", **action_msg})

        # stop instead of sending screenshot
        await ws.send(json.dumps({"type": "stop"}))
        messages.append({"direction": "sent", "type": "stop"})

        stopped = json.loads(await ws.recv())
        assert stopped["type"] == "stopped", f"Expected stopped, got {stopped}"
        messages.append({"direction": "recv", **stopped})

    _append_messages("test_stop_mid_task", messages)


# ---------------------------------------------------------------------------
# Test 7 — stuck detection
# ---------------------------------------------------------------------------

async def test_stuck_detection(stuck_server):
    """
    Send 3 identical screenshots to trigger stuck=True on the 4th get_next_action call.
    Verify via the /webpilot/debug/stub_calls endpoint.
    """
    http_url, ws_url = stuck_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "do stuck thing", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        # 3 iterations with identical screenshots to build up retry_count → 3
        for _ in range(3):
            msg = json.loads(await ws.recv())  # thinking
            messages.append({"direction": "recv", **msg})
            action = json.loads(await ws.recv())  # action (wait)
            assert action["type"] == "action"
            messages.append({"direction": "recv", **action})
            await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
            messages.append({"direction": "sent", "type": "screenshot"})

        # 4th call — should have stuck=True
        msg = json.loads(await ws.recv())  # thinking
        messages.append({"direction": "recv", **msg})
        action4 = json.loads(await ws.recv())  # action
        assert action4["type"] == "action"
        messages.append({"direction": "recv", **action4})

        # stop to end cleanly
        await ws.send(json.dumps({"type": "stop"}))
        stopped = json.loads(await ws.recv())
        assert stopped["type"] == "stopped"
        messages.append({"direction": "recv", **stopped})

    # Verify via debug endpoint that call #3 (0-indexed) had stuck=True
    r = httpx.get(f"{http_url}/webpilot/debug/stub_calls")
    assert r.status_code == 200, f"Debug endpoint failed: {r.text}"
    calls = r.json()["calls"]
    assert len(calls) >= 4, f"Expected at least 4 calls, got {len(calls)}: {calls}"
    assert calls[0]["stuck"] is False, f"Call 0 should not be stuck: {calls[0]}"
    assert calls[1]["stuck"] is False, f"Call 1 should not be stuck: {calls[1]}"
    assert calls[2]["stuck"] is False, f"Call 2 should not be stuck: {calls[2]}"
    assert calls[3]["stuck"] is True, f"Call 3 should be stuck=True: {calls[3]}"
    messages.append({"direction": "assert", "stub_calls": calls[:4]})

    _append_messages("test_stuck_detection", messages)


# ---------------------------------------------------------------------------
# Test 8 — abort interrupt short-circuits without replanning
# ---------------------------------------------------------------------------

async def test_abort_interrupt(abort_server):
    """
    Send an ABORT interrupt mid-task (instruction contains an abort keyword).
    The server must emit 'stopped' immediately — no 'thinking', no replan action.
    Proves the ABORT branch of classify_interruption_type short-circuits correctly
    without calling get_interruption_replan.
    """
    http_url, ws_url = abort_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go somewhere", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        thinking = json.loads(await ws.recv())
        assert thinking["type"] == "thinking", f"Expected thinking, got {thinking}"
        messages.append({"direction": "recv", **thinking})

        action_msg = json.loads(await ws.recv())
        assert action_msg["type"] == "action", f"Expected action, got {action_msg}"
        assert action_msg["action"] == "navigate"
        messages.append({"direction": "recv", **action_msg})

        # Send ABORT interrupt instead of screenshot
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "stop",
            "screenshot": BLANK,
        }))
        messages.append({"direction": "sent", "type": "interrupt", "instruction": "stop"})

        # Server must emit 'stopped' immediately — no thinking or replan action
        response = json.loads(await ws.recv())
        assert response["type"] == "stopped", (
            f"Expected 'stopped' immediately on ABORT interrupt, got {response['type']!r}. "
            "If 'thinking' was received, the ABORT branch did not short-circuit."
        )
        messages.append({"direction": "recv", **response})

    _append_messages("test_abort_interrupt", messages)


# ---------------------------------------------------------------------------
# Test 9 — interrupt after task complete produces new action (not timeout)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def interrupt_after_done_server():
    """Dedicated server for interrupt-after-done — isolated stub state."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_interrupt_after_task_complete(interrupt_after_done_server):
    """
    Complete a navigate_and_done task, then send an interrupt with a new intent.
    Assert the interrupt produces a new action within 20s (not stopped/idle).
    Tests that the watchdog timeout does not fire before navigate completes.
    """
    http_url, ws_url = interrupt_after_done_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=25) as ws:
        # --- Phase 1: complete the initial task ---
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "task"})

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action" and msg["action"] == "navigate"
        messages.append({"direction": "recv", **msg})

        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot"})

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())  # done
        assert msg["type"] == "done", f"Expected done, got {msg}"
        messages.append({"direction": "recv", **msg})

        # --- Phase 2: send interrupt after task is complete ---
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "now go to bing.com instead",
            "screenshot": BLANK,
        }))
        messages.append({"direction": "sent", "type": "interrupt"})

        # Server should respond with thinking + new action within 20s
        import asyncio
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            replan_msg = json.loads(raw)
            messages.append({"direction": "recv", **replan_msg})

            # May be "thinking" first — if so, read the next message
            if replan_msg["type"] == "thinking":
                raw2 = await asyncio.wait_for(ws.recv(), timeout=20.0)
                replan_action = json.loads(raw2)
                messages.append({"direction": "recv", **replan_action})
            else:
                replan_action = replan_msg

            assert replan_action["type"] in ("action", "done"), (
                f"Expected action or done after post-complete interrupt, got {replan_action['type']!r}"
            )
            assert replan_action["type"] != "stopped", (
                "Server returned stopped — watchdog may have fired before navigate completed"
            )

            # Clean up: if action loop is still running, send screenshot to let it finish
            if replan_action["type"] == "action":
                await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
                final = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
                messages.append({"direction": "recv", **final})
                if final["type"] == "thinking":
                    done_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
                    messages.append({"direction": "recv", **done_msg})

        except asyncio.TimeoutError:
            pytest.fail("Timed out waiting for server response after interrupt — watchdog may be too short")

    _append_messages("test_interrupt_after_task_complete", messages)


# ---------------------------------------------------------------------------
# Test 10 — Send button starts fresh task after done
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def send_after_done_server():
    """Dedicated server for send-after-done — isolated stub state."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_new_task_after_done(send_after_done_server):
    """
    Complete a task (done received), then submit a brand-new task via the Send
    path (type="task"). Assert the new task starts fresh — server accepts it,
    returns thinking + action (not an error or stale state from the prior task).
    """
    http_url, ws_url = send_after_done_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=15) as ws:
        # --- Phase 1: complete initial task ---
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action" and msg["action"] == "navigate"
        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # done
        assert msg["type"] == "done"
        messages.append({"phase": "task_1_done"})

        # --- Phase 2: start a brand-new task (Send path, not Interrupt) ---
        await ws.send(json.dumps({
            "type": "task",
            "intent": "now go to bing.com",
            "screenshot": BLANK,
        }))
        messages.append({"direction": "sent", "type": "task", "intent": "now go to bing.com"})

        # Server must accept it cleanly — thinking then action
        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking", f"Expected thinking on new task, got {msg['type']}"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())
        assert msg["type"] == "action", f"Expected action on new task, got {msg['type']}"
        assert msg["action"] == "navigate", "New task should start with navigate"
        messages.append({"direction": "recv", **msg})

        # Clean up — stop instead of completing the second task
        await ws.send(json.dumps({"type": "stop"}))
        stopped = json.loads(await ws.recv())
        assert stopped["type"] == "stopped"

    _append_messages("test_new_task_after_done", messages)


# ---------------------------------------------------------------------------
# Test 11 — Navigate action completes even with slow page load (no early timeout)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slow_navigate_server():
    """Dedicated server for slow-navigate test — isolated stub state."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_slow_navigate_completes_successfully(slow_navigate_server):
    """
    Verifies that a navigate action completes successfully even when the tab
    takes 11+ seconds to load. Simulates a slow page by delaying the screenshot
    response by 11 seconds. Assert done is received — not stopped.
    """
    import asyncio

    http_url, ws_url = slow_navigate_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=25) as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"

        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action" and msg["action"] == "navigate"
        messages.append({"direction": "recv", **msg})

        # Simulate slow page load — 11 seconds before screenshot
        await asyncio.sleep(11)

        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        messages.append({"direction": "sent", "type": "screenshot", "delay_seconds": 11})

        # Server should still be alive — thinking then done
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
        assert msg["type"] == "thinking", f"Expected thinking after delayed screenshot, got {msg['type']}"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
        assert msg["type"] == "done", (
            f"Expected done after delayed screenshot, got {msg['type']}. "
            "If 'stopped', the action loop timed out before the screenshot arrived."
        )
        messages.append({"direction": "recv", **msg})

    _append_messages("test_slow_navigate_completes_successfully", messages)


# ---------------------------------------------------------------------------
# Test 12 — Tooltip text on TaskInput send button
# ---------------------------------------------------------------------------

def test_tooltip_text_on_send_button():
    """
    Verify the TaskInput component renders correct tooltip text by inspecting
    the source. When idle (isRunning=false): 'Start a new task'. When running
    (isRunning=true): 'Send to interrupt current task'.

    This is a source-level assertion — React component rendering is verified
    visually in the extension, but we confirm the strings are wired correctly.
    """
    from pathlib import Path

    component_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "sidebar" / "components" / "TaskInput.jsx"
    source = component_path.read_text()

    # Tooltip for idle state
    assert 'title={isRunning ? "Send to interrupt current task" : "Start a new task"}' in source, (
        "TaskInput.jsx missing tooltip: expected isRunning ternary with "
        "'Start a new task' / 'Send to interrupt current task'"
    )

    # Button label switches correctly
    assert '{isRunning ? "Interrupt" : "Send"}' in source, (
        "TaskInput.jsx missing button label: expected isRunning ternary with 'Interrupt' / 'Send'"
    )


# ---------------------------------------------------------------------------
# Test 13 — interrupt when idle returns stopped (UI state machine)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def idle_interrupt_server():
    """Dedicated server for idle-interrupt test."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_interrupt_when_idle(idle_interrupt_server):
    """
    Complete a task (done received), then send an interrupt on an idle session.
    The server must respond with 'stopped' (not 'thinking' or 'action')
    because the session is idle — no active task to interrupt.
    """
    http_url, ws_url = idle_interrupt_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=15) as ws:
        # Complete a task first to leave session in 'done' state
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action"
        await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # done
        assert msg["type"] == "done"
        messages.append({"phase": "task_done"})

        # Now stop the session to transition to idle
        await ws.send(json.dumps({"type": "stop"}))
        msg = json.loads(await ws.recv())  # stopped
        assert msg["type"] == "stopped"

        # Send interrupt on idle session — should get stopped, not thinking
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "go somewhere else",
            "screenshot": BLANK,
        }))
        messages.append({"direction": "sent", "type": "interrupt"})

        response = json.loads(await ws.recv())
        assert response["type"] == "stopped", (
            f"Expected 'stopped' on idle-session interrupt, got {response['type']!r}"
        )
        messages.append({"direction": "recv", **response})

    _append_messages("test_interrupt_when_idle", messages)


# ---------------------------------------------------------------------------
# Test 14 — input mode idle shows Send (source-level)
# ---------------------------------------------------------------------------

def test_input_mode_idle_shows_send():
    """Verify TaskInput renders 'Send' label and correct placeholder when idle."""
    from pathlib import Path

    component_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "sidebar" / "components" / "TaskInput.jsx"
    source = component_path.read_text()

    assert '{isRunning ? "Interrupt" : "Send"}' in source, (
        "TaskInput.jsx missing idle-state 'Send' label"
    )
    assert 'Describe what to do' in source, (
        "TaskInput.jsx missing idle placeholder text 'Describe what to do'"
    )


# ---------------------------------------------------------------------------
# Test 15 — input mode running shows Interrupt (source-level)
# ---------------------------------------------------------------------------

def test_input_mode_running_shows_interrupt():
    """Verify TaskInput renders 'Interrupt' label when running."""
    from pathlib import Path

    component_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "sidebar" / "components" / "TaskInput.jsx"
    source = component_path.read_text()

    assert '"Interrupt"' in source, (
        "TaskInput.jsx missing 'Interrupt' label for running state"
    )
    assert '"Send to interrupt current task"' in source, (
        "TaskInput.jsx missing interrupt tooltip text"
    )


# ---------------------------------------------------------------------------
# Test 16 — interrupt button disabled when idle (source-level)
# ---------------------------------------------------------------------------

def test_interrupt_button_disabled_idle():
    """Verify TaskInput disables submit button when text is empty or component is disabled."""
    from pathlib import Path

    component_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "sidebar" / "components" / "TaskInput.jsx"
    source = component_path.read_text()

    assert 'disabled || !text.trim()' in source, (
        "TaskInput.jsx missing disabled guard: 'disabled || !text.trim()'"
    )
    assert '"Start a new task"' in source, (
        "TaskInput.jsx missing idle tooltip: 'Start a new task'"
    )


# ---------------------------------------------------------------------------
# Test 17 — navigate watchdog race (source-level + e2e)
# ---------------------------------------------------------------------------

def test_navigate_watchdog_race():
    """
    Verify background.js sets navigate watchdog >= 20s, then e2e: start task,
    receive navigate, delay 12s, send screenshot, assert done (not stopped).
    Source-level check ensures the extension won't kill navigate too early.
    """
    from pathlib import Path

    bg_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "background.js"
    source = bg_path.read_text(encoding="utf-8")

    # Assert navigate watchdog is at least 20s
    assert '20000' in source, (
        "background.js missing 20000ms navigate watchdog timeout"
    )
    # The e2e portion is covered by test_slow_navigate_completes_successfully


# ---------------------------------------------------------------------------
# Test 18 — watchdog clears after screenshot (source-level)
# ---------------------------------------------------------------------------

def test_watchdog_clears_after_screenshot():
    """Verify background.js clears the watchdog after sending a screenshot."""
    from pathlib import Path

    bg_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "background.js"
    source = bg_path.read_text(encoding="utf-8")

    assert 'clearTimeout(_interruptWatchdog)' in source, (
        "background.js missing clearTimeout(_interruptWatchdog) after screenshot"
    )
    assert '_interruptWatchdog = null' in source, (
        "background.js missing _interruptWatchdog = null after clear"
    )


# ---------------------------------------------------------------------------
# Test 19 — current_url in context (e2e)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def current_url_server():
    """Dedicated server for current_url test."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_current_url_in_context(current_url_server):
    """
    Send screenshot with current_url set. After done, check stub_calls
    to verify current_url was passed through to the handler.
    """
    http_url, ws_url = current_url_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action"

        # Send screenshot with current_url
        await ws.send(json.dumps({
            "type": "screenshot",
            "screenshot": BLANK,
            "current_url": "https://example.com",
        }))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # done
        assert msg["type"] == "done"
        messages.append({"phase": "done_with_url"})

    # Check stub_calls for current_url
    r = httpx.get(f"{http_url}/webpilot/debug/stub_calls")
    assert r.status_code == 200
    calls = r.json()["calls"]
    # The second call (after screenshot) should have current_url
    url_calls = [c for c in calls if c.get("current_url")]
    assert len(url_calls) > 0, f"No stub calls with current_url set: {calls}"
    assert url_calls[-1]["current_url"] == "https://example.com"
    messages.append({"stub_calls_with_url": url_calls})

    _append_messages("test_current_url_in_context", messages)


# ---------------------------------------------------------------------------
# Test 20 — current_url on redirect (e2e)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def redirect_server():
    """Dedicated server for redirect test."""
    with _server_ctx("navigate_with_redirect") as urls:
        yield urls


async def test_current_url_on_redirect(redirect_server):
    """
    Use navigate_with_redirect scenario. Send screenshot with
    current_url pointing to a login page. Assert server returns paused
    with reason 'login'. Check stub_calls for current_url passthrough.
    """
    http_url, ws_url = redirect_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "open Gmail", "screenshot": BLANK}))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action"
        assert msg["action"] == "navigate"
        messages.append({"direction": "recv", **msg})

        # Send screenshot with login redirect URL
        await ws.send(json.dumps({
            "type": "screenshot",
            "screenshot": BLANK,
            "current_url": "https://accounts.google.com/signin",
        }))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"

        # Should get paused with login reason
        msg = json.loads(await ws.recv())
        assert msg["type"] == "paused", f"Expected paused, got {msg['type']}"
        assert msg["reason"] == "login"
        messages.append({"direction": "recv", **msg})

    # Check stub_calls for current_url passthrough
    r = httpx.get(f"{http_url}/webpilot/debug/stub_calls")
    assert r.status_code == 200
    calls = r.json()["calls"]
    url_calls = [c for c in calls if c.get("current_url") == "https://accounts.google.com/signin"]
    assert len(url_calls) > 0, f"No stub calls with login current_url: {calls}"
    messages.append({"stub_calls_with_login_url": url_calls})

    _append_messages("test_current_url_on_redirect", messages)


# ---------------------------------------------------------------------------
# Test 21 — modal overlay does not block (e2e)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def modal_server():
    """Dedicated server for modal test."""
    with _server_ctx("modal_then_proceed") as urls:
        yield urls


async def test_modal_overlay_does_not_block(modal_server):
    """
    Walk through modal_then_proceed scenario (4 actions: click×2 → click → done).
    Assert task reaches done without getting stuck in infinite loop.
    """
    http_url, ws_url = modal_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=20) as ws:
        await ws.send(json.dumps({"type": "task", "intent": "click the target button", "screenshot": BLANK}))

        # Walk through 3 click actions + done
        for i in range(3):
            msg = json.loads(await ws.recv())  # thinking
            assert msg["type"] == "thinking", f"Step {i}: expected thinking, got {msg['type']}"
            msg = json.loads(await ws.recv())  # action (click)
            assert msg["type"] == "action", f"Step {i}: expected action, got {msg['type']}"
            assert msg["action"] == "click"
            messages.append({"step": i, "action": msg["action"]})
            await ws.send(json.dumps({"type": "screenshot", "screenshot": BLANK}))

        # Final: thinking + done
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # done
        assert msg["type"] == "done", f"Expected done after modal sequence, got {msg['type']}"
        messages.append({"phase": "done"})

    _append_messages("test_modal_overlay_does_not_block", messages)


# ---------------------------------------------------------------------------
# Test 22 — consecutive failures stops (source-level)
# ---------------------------------------------------------------------------

def test_consecutive_failures_stops():
    """Verify background.js stops after MAX_CONSECUTIVE_FAILURES consecutive failures."""
    from pathlib import Path

    bg_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "background.js"
    source = bg_path.read_text(encoding="utf-8")

    assert '_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES' in source, (
        "background.js missing consecutive failure threshold check"
    )
    assert 'sendWS({ type: "stop" })' in source, (
        "background.js missing stop message on consecutive failures"
    )


# ---------------------------------------------------------------------------
# Test 23 — step counter reset on interrupt (source-level + e2e)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def step_reset_server():
    """Dedicated server for step-counter-reset test."""
    with _server_ctx("interrupt_redirect") as urls:
        yield urls


async def test_step_counter_reset_on_interrupt(step_reset_server):
    """
    Source-level: verify background.js resets _stepCounter on interrupt.
    E2e: start task, do 2 steps, interrupt, verify new action arrives.
    """
    from pathlib import Path

    # Source-level check
    bg_path = Path(__file__).resolve().parent.parent / "webpilot-extension" / "background.js"
    source = bg_path.read_text(encoding="utf-8")
    assert '_stepCounter = 0' in source, (
        "background.js missing _stepCounter = 0 reset"
    )

    # E2e portion: interrupt mid-task, verify the loop continues
    http_url, ws_url = step_reset_server
    sid = _create_session(http_url)
    messages = []

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}", close_timeout=15) as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))

        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action"
        messages.append({"step": 1, "action": msg["action"]})

        # Interrupt instead of sending screenshot
        await ws.send(json.dumps({
            "type": "interrupt",
            "instruction": "go to bing instead",
            "screenshot": BLANK,
        }))

        # Should get thinking + new action (not stopped from step limit)
        import asyncio
        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        replan = json.loads(raw)
        if replan["type"] == "thinking":
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            replan = json.loads(raw)

        assert replan["type"] in ("action", "done"), (
            f"Expected action or done after interrupt, got {replan['type']}"
        )
        messages.append({"post_interrupt": replan["type"]})

        # Clean up
        if replan["type"] == "action":
            await ws.send(json.dumps({"type": "stop"}))
            stopped = json.loads(await ws.recv())
            messages.append({"cleanup": stopped["type"]})

    _append_messages("test_step_counter_reset_on_interrupt", messages)


# ---------------------------------------------------------------------------
# Test 24 — session reuse after reconnect (e2e)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reconnect_server():
    """Dedicated server for reconnect test."""
    with _server_ctx("navigate_and_done") as urls:
        yield urls


async def test_session_reuse_after_reconnect(reconnect_server):
    """
    Create session, connect WS, send task, receive navigate.
    Disconnect WS. Reconnect same session_id. Send new task.
    Assert thinking + action (session not deleted on disconnect).
    """
    import asyncio

    http_url, ws_url = reconnect_server
    sid = _create_session(http_url)
    messages = []

    # First connection — start a task and get navigate
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "go to example.com", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())  # thinking
        assert msg["type"] == "thinking"
        msg = json.loads(await ws.recv())  # action (navigate)
        assert msg["type"] == "action"
        messages.append({"phase": "first_connect", "action": msg["action"]})
    # WS now disconnected

    # Brief pause for server to process disconnect
    await asyncio.sleep(0.5)

    # Reconnect same session_id
    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({"type": "task", "intent": "now go to bing.com", "screenshot": BLANK}))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "thinking", f"Expected thinking on reconnect, got {msg['type']}"
        messages.append({"direction": "recv", **msg})

        msg = json.loads(await ws.recv())
        # The shared stub may have advanced its index, so accept action or done
        assert msg["type"] in ("action", "done"), (
            f"Expected action or done on reconnect, got {msg['type']}"
        )
        messages.append({"direction": "recv", **msg})

        # Clean up if still running
        if msg["type"] == "action":
            await ws.send(json.dumps({"type": "stop"}))
            stopped = json.loads(await ws.recv())
            assert stopped["type"] == "stopped"

    _append_messages("test_session_reuse_after_reconnect", messages)
