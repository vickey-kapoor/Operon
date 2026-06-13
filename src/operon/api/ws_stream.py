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
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

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

# User-configured confidence threshold (0.0 = off, >0.0 means pause when
# PolicyDecision.confidence < threshold). Set via "set_confidence_threshold" control message.
_confidence_threshold: float = 0.0

# Override hint typed by the user during a HITL pause. Consumed once by the next
# step's policy call and cleared — see loop.py consume_override_hint usage.
_pending_override_hint: str | None = None

# Reference to the AgentLoop singleton so WS control messages can call resume_run().
_agent_loop: Any = None

# Rule names disabled by the user via the Settings UI.
_disabled_rules: set[str] = set()


def set_executor(executor: Any) -> None:
    global _executor
    _executor = executor


def get_executor() -> Any:
    """Return the executor singleton registered for the active runtime, if any."""
    return _executor


def set_active_run(run_id: str | None) -> None:
    global _active_run_id
    _active_run_id = run_id


def set_browser_manager(manager: Any) -> None:
    global _browser_manager
    _browser_manager = manager


def set_agent_loop(loop: Any) -> None:
    global _agent_loop
    _agent_loop = loop


def set_confidence_threshold(v: float) -> None:
    global _confidence_threshold
    _confidence_threshold = max(0.0, min(1.0, float(v)))


def get_confidence_threshold() -> float:
    return _confidence_threshold


def get_disabled_rules() -> frozenset[str]:
    return frozenset(_disabled_rules)


def set_disabled_rules(names: list[str]) -> None:
    global _disabled_rules
    _disabled_rules = set(names)


def consume_override_hint() -> str | None:
    """Pop and return the pending user override hint (one-shot)."""
    global _pending_override_hint
    hint = _pending_override_hint
    _pending_override_hint = None
    return hint


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
            pass


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
        set_confidence_threshold(float(msg.get("threshold", 0.0)))
        publish_event({"type": "control_ack", "action": "set_confidence_threshold", "threshold": _confidence_threshold})

    elif msg_type == "pause":
        publish_event({"type": "control_ack", "action": "pause"})

    elif msg_type == "resume":
        # Resume a HITL-paused run from the UI "Proceed" button.
        if _agent_loop is not None and _active_run_id is not None:
            try:
                await _agent_loop.resume_run(_active_run_id)
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
        global _pending_override_hint
        _pending_override_hint = hint or None
        if _agent_loop is not None and _active_run_id is not None:
            try:
                await _agent_loop.resume_run(_active_run_id)
            except Exception:
                pass
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
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _subscribers.append(q)

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
        await ws.close()
