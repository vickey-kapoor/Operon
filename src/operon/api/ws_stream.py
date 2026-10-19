"""WebSocket streaming endpoint.

Publishes step events (JSON text frames) and screenshot frames (binary JPEG)
to all connected observable clients.

Usage from route handlers:
    from operon.api import ws_stream
    ws_stream.publish_event({"type": "step", ...})
    ws_stream.publish_frame(jpeg_bytes)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from typing import Any

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from operon.agent import control as _control

router = APIRouter()
logger = logging.getLogger(__name__)

# One asyncio.Queue per connected client. Items are either:
#   dict  → serialised as a JSON text frame
#   bytes → sent as a binary frame (raw JPEG)
_subscribers: list[asyncio.Queue[dict | bytes]] = []

# Optional executor reference for live frame capture (Operon agent loop path).
_executor: Any = None
# run_id of the currently active run (set/cleared by route handlers).
_active_run_id: str | None = None

# BrowserManager instance for CDP-attached screencast (command-center path).
# When set, frame streaming is driven by BrowserManager.start_screencast() and
# the _frame_pusher fallback is skipped automatically.
_browser_manager: Any = None

# Reference to the AgentLoop singleton so WS control messages can call resume_run().
_agent_loop: Any = None

# Confidence threshold, disabled rules and override hints live in
# operon.agent.control, which the loop reads; this module only writes them.


def set_executor(executor: Any) -> None:
    global _executor
    _executor = executor


def set_active_run(run_id: str | None) -> None:
    global _active_run_id
    _active_run_id = run_id


def set_browser_manager(manager: Any) -> None:
    global _browser_manager
    _browser_manager = manager


def set_agent_loop(loop: Any) -> None:
    global _agent_loop
    _agent_loop = loop


set_confidence_threshold = _control.set_confidence_threshold
get_confidence_threshold = _control.get_confidence_threshold
get_disabled_rules = _control.get_disabled_rules
set_disabled_rules = _control.set_disabled_rules
consume_override_hint = _control.consume_override_hint


def publish_event(event: dict[str, Any]) -> None:
    """Fan out a JSON step event to all connected WS clients."""
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # slow client — drop rather than block


def publish_frame(jpeg_bytes: bytes) -> None:
    """Fan out a raw JPEG screenshot frame to all connected WS clients."""
    for q in _subscribers:
        try:
            q.put_nowait(jpeg_bytes)
        except asyncio.QueueFull:
            pass


async def _sender(ws: WebSocket, q: asyncio.Queue) -> None:
    while True:
        item = await q.get()
        if item is None:
            return
        if isinstance(item, bytes):
            await ws.send_bytes(item)
        else:
            await ws.send_text(json.dumps(item))


async def _receiver(ws: WebSocket) -> None:
    """Consume control messages from observable clients."""
    async for raw in ws.iter_text():
        try:
            msg = json.loads(raw)
            await _handle_control(msg)
        except Exception:
            # One bad message must not drop the connection, but it must not vanish either.
            logger.warning("ignoring WebSocket control message that failed: %.200r", raw, exc_info=True)


async def _resume_active_run() -> None:
    # Local import: runtime.tasks imports this module at load time.
    from operon.api.runtime.tasks import resume_and_schedule
    await resume_and_schedule(_agent_loop, _active_run_id)


async def _handle_control(msg: dict) -> None:
    msg_type = msg.get("type")

    if msg_type == "inject_input":
        # Frontend → real browser input via BrowserManager.
        if _browser_manager is not None:
            await _browser_manager.inject_input(
                x_ratio=float(msg.get("x_ratio", 0.0)),
                y_ratio=float(msg.get("y_ratio", 0.0)),
                input_type=msg.get("input_type", "click"),
                text=msg.get("text", ""),
                key=msg.get("key", ""),
                button=msg.get("button", "left"),
                delta_y=int(msg.get("delta_y", 300)),
            )

    elif msg_type == "set_confidence_threshold":
        threshold = set_confidence_threshold(float(msg.get("threshold", 0.0)))
        publish_event({"type": "control_ack", "action": "set_confidence_threshold", "threshold": threshold})

    elif msg_type == "pause":
        publish_event({"type": "control_ack", "action": "pause"})

    elif msg_type == "resume":
        # Resume a HITL-paused run from the UI "Proceed" button.
        if _agent_loop is not None and _active_run_id is not None:
            try:
                await _resume_active_run()
                publish_event({"type": "control_ack", "action": "resume"})
            except Exception as exc:
                publish_event({"type": "control_ack", "action": "resume", "error": str(exc)})
        else:
            publish_event({"type": "control_ack", "action": "resume"})

    elif msg_type == "set_disabled_rules":
        rules = [str(r) for r in msg.get("rules", []) if isinstance(r, str)]
        set_disabled_rules(rules)
        publish_event({"type": "control_ack", "action": "set_disabled_rules", "rules": rules})

    elif msg_type == "override":
        # Store the correction hint and resume the paused run in one shot.
        hint = msg.get("hint", "").strip()
        if _active_run_id is not None:
            _control.set_override_hint(_active_run_id, hint)
        if _agent_loop is not None and _active_run_id is not None:
            try:
                await _resume_active_run()
            except Exception as exc:
                publish_event({"type": "control_ack", "action": "override", "hint": hint, "error": str(exc)})
                return
        publish_event({"type": "control_ack", "action": "override", "hint": hint})


async def _frame_pusher(q: asyncio.Queue) -> None:
    """Fallback frame pusher for Operon agent runs (polls live_frame_png at 10fps).

    Skipped when BrowserManager is active — in that mode, CDP Page.startScreencast
    drives frame delivery directly into publish_frame(), which fans out to all queues
    including this client's queue without needing a per-client poll loop.
    """
    while True:
        await asyncio.sleep(0.1)
        # BrowserManager is driving frames via CDP events — nothing to do here.
        if _browser_manager is not None:
            continue
        if _active_run_id and _executor and hasattr(_executor, "live_frame_png"):
            try:
                frame = await _executor.live_frame_png(_active_run_id)
                if frame:
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                pass


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    # Auth gate: the stream both exposes live frames and accepts control messages
    # (inject_input, resume, ...), so it must honor API_KEYS like the HTTP routes.
    # Browsers can't set headers on a WebSocket — the key comes via ?api_key= (or
    # an x-api-key subprotocol). No-op when API_KEYS is unset.
    from operon.api.auth import websocket_api_key_ok
    if not websocket_api_key_ok(ws):
        await ws.close(code=1008)  # 1008 = policy violation (unauthorized)
        return
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _subscribers.append(q)
    # Static pages may emit no further CDP frames after this client connects.
    # The manager clears its cached frame when stopping or changing the stream.
    if _browser_manager is not None:
        frame = _browser_manager.latest_frame
        if frame is not None:
            q.put_nowait(frame)

    frame_task = asyncio.create_task(_frame_pusher(q))
    send_task = asyncio.create_task(_sender(ws, q))
    recv_task = asyncio.create_task(_receiver(ws))

    try:
        done, pending = await asyncio.wait(
            [send_task, recv_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        frame_task.cancel()
        send_task.cancel()
        recv_task.cancel()
        if q in _subscribers:
            _subscribers.remove(q)
        # Finish cancelling child tasks even when the ASGI request's AnyIO
        # cancellation scope has already been cancelled (including TestClient).
        with anyio.CancelScope(shield=True):
            await asyncio.gather(frame_task, send_task, recv_task, return_exceptions=True)
            if (
                ws.application_state is WebSocketState.CONNECTED
                and ws.client_state is WebSocketState.CONNECTED
            ):
                # The peer can disconnect between the state check and the send.
                with suppress(WebSocketDisconnect):
                    await ws.close()
