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
import math
from contextlib import suppress
from typing import Any

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from operon.agent import control as _control
from operon.core.runtime import StreamState, current_runtime

router = APIRouter()
logger = logging.getLogger(__name__)

# What the stream serves from (subscriber queues, executors, the active run, the
# CDP browser manager) lives on the process runtime (KAN-156): ``_stream()``.
# Confidence threshold, disabled rules and override hints live in
# operon.agent.control, which the loop reads; this module only writes them.


def _stream() -> StreamState:
    return current_runtime().stream


def set_executor(executor: Any) -> None:
    """The browser loop's executor, for live frames when no CDP screencast runs."""
    _stream().browser_executor = executor


def set_desktop_executor(executor: Any) -> None:
    """The desktop loop's executor. Only one with a ``live_frame_png`` hook (the Linux
    cloud computer, KAN-108) streams; a Windows desktop shows step screenshots instead."""
    _stream().desktop_executor = executor


def set_active_run(run_id: str | None, loop: Any = None) -> None:
    """The run on screen and the loop (browser or desktop) that owns it. Control
    messages go to that loop; the browser loop is the fallback for callers that
    don't say which loop started the run."""
    stream = _stream()
    stream.active_run_id = run_id
    stream.active_loop = loop if run_id is not None else None


def active_run_id() -> str | None:
    return _stream().active_run_id


def _control_loop() -> Any:
    stream = _stream()
    return stream.active_loop if stream.active_loop is not None else current_runtime().browser_loop


async def _active_run_in_takeover() -> bool:
    """Whether the active run (if any) is under the person's control."""
    loop = _control_loop()
    run_id = _stream().active_run_id
    if run_id is None or loop is None:
        return True  # No run to fight with: the live view is just a browser.
    try:
        record = await loop.run_store.get_run(run_id)
    except Exception:  # noqa: BLE001 - refuse on doubt
        return False
    return record is None or bool(getattr(record, "in_takeover", False)) or record.status.value not in {
        "pending",
        "running",
    }


def set_browser_manager(manager: Any) -> None:
    _stream().browser_manager = manager


def set_agent_loop(loop: Any) -> None:
    """The browser agent loop (control messages fall back to it)."""
    current_runtime().browser_loop = loop


set_confidence_threshold = _control.set_confidence_threshold
get_confidence_threshold = _control.get_confidence_threshold
get_disabled_rules = _control.get_disabled_rules
set_disabled_rules = _control.set_disabled_rules
consume_override_hint = _control.consume_override_hint


def publish_event(event: dict[str, Any]) -> None:
    """Fan out a JSON step event to all connected WS clients."""
    for q in _stream().subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # slow client — drop rather than block


def publish_frame(jpeg_bytes: bytes) -> None:
    """Fan out a raw JPEG screenshot frame to all connected WS clients."""
    for q in _stream().subscribers:
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


async def _receiver(ws: WebSocket, scope: str = "control") -> None:
    """Consume control messages from observable clients. A ``view`` client only watches."""
    async for raw in ws.iter_text():
        try:
            msg = json.loads(raw)
            if scope != "control":
                publish_event({"type": "control_ack", "action": msg.get("type"), "error": "view-only stream"})
                continue
            await _handle_control(msg)
        except Exception:
            # One bad message must not drop the connection, but it must not vanish either.
            logger.warning("ignoring WebSocket control message that failed: %.200r", raw, exc_info=True)


async def _resume_active_run(*, approve_gated_action: bool = True) -> None:
    # Local import: runtime.tasks imports this module at load time.
    from operon.api.runtime.tasks import resume_and_schedule
    await resume_and_schedule(_control_loop(), _stream().active_run_id, approve_gated_action=approve_gated_action)


async def _handle_control(msg: dict) -> None:
    msg_type = msg.get("type")

    if msg_type == "inject_input":
        # Frontend → real browser input via BrowserManager. While a run is steering
        # the browser, only a person who took control may type into it (as on the
        # HTTP input route).
        if not await _active_run_in_takeover():
            publish_event({"type": "control_ack", "action": "inject_input", "error": "take control of the run first"})
            return
        manager = _stream().browser_manager
        if manager is not None:
            await manager.inject_input(
                x_ratio=float(msg.get("x_ratio", 0.0)),
                y_ratio=float(msg.get("y_ratio", 0.0)),
                input_type=msg.get("input_type", "click"),
                text=msg.get("text", ""),
                key=msg.get("key", ""),
                button=msg.get("button", "left"),
                delta_y=int(msg.get("delta_y", 300)),
            )

    elif msg_type == "set_confidence_threshold":
        raw = float(msg.get("threshold", 0.0))
        if not math.isfinite(raw):
            publish_event({"type": "control_ack", "action": "set_confidence_threshold", "error": "threshold must be a number"})
            return
        threshold = set_confidence_threshold(raw)
        publish_event({"type": "control_ack", "action": "set_confidence_threshold", "threshold": threshold})

    elif msg_type == "pause":
        # Pause the active run for real: the auto-run loop stops at the next step
        # boundary, and a step in flight ends paused instead of RUNNING.
        if _control_loop() is None or _stream().active_run_id is None:
            publish_event({"type": "control_ack", "action": "pause", "error": "no active run"})
            return
        try:
            run = await _control_loop().pause_run(_stream().active_run_id)
        except Exception as exc:
            publish_event({"type": "control_ack", "action": "pause", "error": str(exc)})
            return
        publish_event({"type": "control_ack", "action": "pause", "status": run.status.value})

    elif msg_type == "resume":
        # Resume a HITL-paused run from the UI "Proceed" button.
        if _control_loop() is not None and _stream().active_run_id is not None:
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
        if _stream().active_run_id is not None:
            _control.set_override_hint(_stream().active_run_id, hint)
        if _control_loop() is not None and _stream().active_run_id is not None:
            try:
                # A correction, not an approval: the paused action stays gated.
                await _resume_active_run(approve_gated_action=False)
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
        stream = _stream()
        run_id = stream.active_run_id
        # A desktop run streams its screen (it returns None for any run not on it).
        desktop = stream.desktop_executor
        if run_id and desktop is not None and hasattr(desktop, "live_frame_png"):
            try:
                frame = await desktop.live_frame_png(run_id)
            except Exception:
                frame = None
            if frame:
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    pass
                continue
        # BrowserManager is driving frames via CDP events — nothing to do here.
        if stream.browser_manager is not None:
            continue
        # A parallel run owns its own loop and executor; prefer it over the shared one.
        browser = stream.browser_executor
        own = getattr(stream.active_loop, "executor", None)
        if own is not None and own is not desktop and hasattr(own, "live_frame_png"):
            browser = own
        if run_id and browser and hasattr(browser, "live_frame_png"):
            try:
                frame = await browser.live_frame_png(run_id)
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
    # (inject_input, resume, ...), so it needs the API key like the HTTP routes.
    # Browsers can't set headers on a WebSocket — the key comes as an x-api-key
    # subprotocol. No-op only when OPERON_ALLOW_UNAUTHENTICATED is set.
    # The Origin gate is always on: CORS does not cover WebSockets, so without it
    # any page in a local browser could connect and drive the agent.
    # A short-lived stream token (x-stream-token, KAN-89) also opens it, with the
    # scope it was minted for: "view" clients get frames and events but no control.
    from operon.api.auth import (
        websocket_origin_ok,
        websocket_stream_scope,
        websocket_subprotocol,
    )
    scope = websocket_stream_scope(ws) if websocket_origin_ok(ws) else None
    if scope is None:
        await ws.close(code=1008)  # 1008 = policy violation (unauthorized)
        return
    await ws.accept(subprotocol=websocket_subprotocol(ws))
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _stream().subscribers.append(q)
    # Static pages may emit no further CDP frames after this client connects.
    # The manager clears its cached frame when stopping or changing the stream.
    manager = _stream().browser_manager
    if manager is not None:
        frame = manager.latest_frame
        if frame is not None:
            q.put_nowait(frame)

    frame_task = asyncio.create_task(_frame_pusher(q))
    send_task = asyncio.create_task(_sender(ws, q))
    recv_task = asyncio.create_task(_receiver(ws, scope))

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
        subscribers = _stream().subscribers
        if q in subscribers:
            subscribers.remove(q)
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
