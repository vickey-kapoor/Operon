"""Desktop Mode API router — WS-driven and REST-triggered desktop control."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.api.desktop_models import (
    DesktopConfirmMessage,
    DesktopInterruptMessage,
    DesktopSession,
    DesktopSessionStatusResponse,
    DesktopStartRequest,
    DesktopStartResponse,
    DesktopTaskMessage,
)

_DESKTOP_MODE_ENABLED = os.environ.get("DESKTOP_MODE_ENABLED", "").lower() in ("1", "true", "yes")
_MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
_ACTION_LOOP_TIMEOUT = int(os.environ.get("ACTION_LOOP_TIMEOUT", "120"))
_MAX_LOOP_STEPS = int(os.environ.get("MAX_LOOP_STEPS", "30"))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/desktop", tags=["desktop"])

# Session store: session_id → DesktopSession
_sessions: Dict[str, DesktopSession] = {}

# Shared handler — same WebPilotHandler instance configured with desktop system prompt
_desktop_handler = None
# Desktop executor (server-side mode only)
_desktop_executor = None


def init_desktop_handler(handler, executor=None) -> None:
    """Called from server.py lifespan."""
    global _desktop_handler, _desktop_executor
    _desktop_handler = handler
    _desktop_executor = executor


async def cleanup_desktop_sessions() -> None:
    """Remove sessions inactive for >30 minutes."""
    _MAX_SESSION_DURATION = int(os.environ.get("MAX_SESSION_DURATION", "1800"))
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - _MAX_SESSION_DURATION
        stale = [sid for sid, s in list(_sessions.items()) if s.last_active < cutoff]
        for sid in stale:
            _sessions.pop(sid)
            logger.info("Cleaned up stale desktop session %s", sid)


def _check_desktop_enabled():
    """Raise 503 if DESKTOP_MODE_ENABLED is not set."""
    enabled = os.environ.get("DESKTOP_MODE_ENABLED", "").lower() in ("1", "true", "yes")
    if not enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Desktop mode is not enabled. "
                "Set DESKTOP_MODE_ENABLED=true on the host running the server. "
                "Desktop mode cannot run on Cloud Run or headless containers."
            ),
        )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions")
async def create_desktop_session() -> dict:
    """Create a desktop session for the interactive WS path."""
    _check_desktop_enabled()
    if len(_sessions) >= 1000:
        raise HTTPException(status_code=503, detail="Maximum session limit reached")
    session_id = str(uuid.uuid4())
    _sessions[session_id] = DesktopSession(session_id=session_id)
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_desktop_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "deleted"}


@router.get("/sessions/{session_id}", response_model=DesktopSessionStatusResponse)
async def get_desktop_session(session_id: str) -> DesktopSessionStatusResponse:
    """Poll status of a server-side desktop task."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    return DesktopSessionStatusResponse(
        session_id=session_id,
        status=session.status,
        intent=session.intent,
        steps_taken=getattr(session, "_steps_taken", 0),
        result=getattr(session, "_result", None),
        error=getattr(session, "_error", None),
    )


@router.post("/start", response_model=DesktopStartResponse, status_code=202)
async def start_desktop_task(body: DesktopStartRequest) -> DesktopStartResponse:
    """
    Start a server-side autonomous desktop task.

    The server captures its own screenshots and executes actions directly
    against the host OS. The client polls GET /desktop/sessions/{id} for status.
    """
    _check_desktop_enabled()
    if _desktop_handler is None or _desktop_executor is None:
        raise HTTPException(status_code=503, detail="Desktop handler not initialized")

    session_id = str(uuid.uuid4())
    w, h = _desktop_executor.screen_size
    session = DesktopSession(
        session_id=session_id,
        intent=body.intent,
        screen_width=w,
        screen_height=h,
        scale_x=_desktop_executor._scale_x,
        scale_y=_desktop_executor._scale_y,
    )
    _sessions[session_id] = session

    # Run the autonomous loop in the background
    asyncio.create_task(
        _run_autonomous_loop(session, body.max_steps)
    )

    return DesktopStartResponse(
        session_id=session_id,
        status="started",
        screen_width=w,
        screen_height=h,
    )


# ---------------------------------------------------------------------------
# Server-side autonomous loop
# ---------------------------------------------------------------------------

async def _run_autonomous_loop(session: DesktopSession, max_steps: int) -> None:
    """
    Server drives the entire loop: capture → Gemini → execute → repeat.
    No WebSocket involved — results stored on the session object.
    """
    session.status = "running"
    session._steps_taken = 0
    session._result = None
    session._error = None

    try:
        await asyncio.wait_for(
            _autonomous_loop_inner(session, max_steps),
            timeout=_ACTION_LOOP_TIMEOUT,
        )
    except asyncio.TimeoutError:
        session.status = "error"
        session._error = f"Task timed out after {_ACTION_LOOP_TIMEOUT}s"
        logger.warning("Autonomous desktop loop timed out for session %s", session.session_id)
    except Exception as exc:
        session.status = "error"
        session._error = str(exc)
        logger.exception("Autonomous desktop loop error for session %s", session.session_id)


async def _autonomous_loop_inner(session: DesktopSession, max_steps: int) -> None:
    history = []
    prev_hash = b""
    retry_count = 0
    consecutive_waits = 0

    for step in range(1, max_steps + 1):
        session._steps_taken = step
        if session.abort_event.is_set():
            session.status = "idle"
            return

        screenshot_b64 = await _desktop_executor.screenshot_base64()

        # Stuck detection: identical screenshots OR repeated wait actions
        new_hash = hashlib.md5(base64.b64decode(screenshot_b64)).digest()
        if new_hash == prev_hash:
            retry_count += 1
        else:
            retry_count = 0
        prev_hash = new_hash
        stuck = retry_count >= _MAX_RETRIES or consecutive_waits >= _MAX_RETRIES
        if stuck:
            retry_count = 0
            prev_hash = b""
            consecutive_waits = 0

        try:
            action = await _desktop_handler.get_next_action_desktop(
                image_b64=screenshot_b64,
                intent=session.intent,
                history=history,
                stuck=stuck,
                screen_width=session.screen_width,
                screen_height=session.screen_height,
            )
        except Exception as exc:
            session._error = str(exc)
            session.status = "error"
            return

        # Track consecutive wait actions — treat as stuck if repeated
        if action.action == "wait":
            consecutive_waits += 1
        else:
            consecutive_waits = 0

        if action.action == "done":
            session._result = action.narration
            session.status = "done"
            return

        if action.action == "confirm_required" or action.is_irreversible:
            # Autonomous mode: auto-deny irreversible actions — safety first
            logger.warning(
                "Autonomous mode blocked irreversible action at step %d: %s",
                step, action.action_label,
            )
            session._result = f"Stopped: action requires confirmation ({action.action_label})"
            session.status = "done"
            return

        result = await _desktop_executor.execute(action)
        if not result.success:
            logger.warning(
                "Desktop action failed at step %d: %s — %s",
                step, action.action, result.error,
            )
        # Brief yield between steps
        await asyncio.sleep(0.1)

    session._result = f"Reached max steps ({max_steps}) without completing task."
    session.status = "done"


# ---------------------------------------------------------------------------
# Interactive WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def desktop_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Interactive Desktop Mode WS endpoint.

    Protocol is identical to /webpilot/ws/{session_id}.
    The client (desktop app or script) executes actions and sends back screenshots.
    """
    if session_id not in _sessions:
        await websocket.accept()
        await websocket.close(code=4404, reason="Session not found")
        return

    if _desktop_handler is None:
        await websocket.accept()
        await websocket.close(code=4503, reason="Desktop handler not initialized")
        return

    session = _sessions[session_id]
    await websocket.accept()
    logger.info("Desktop WS connected: session=%s", session_id)

    had_error = False
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 15 * 1024 * 1024:
                await websocket.send_json({"type": "error", "message": "Message too large"})
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")
            session.last_active = time.time()

            try:
                if msg_type == "task":
                    msg = DesktopTaskMessage(**data)
                    session.intent = msg.intent
                    session.history = []
                    session.status = "running"
                    session.abort_event.clear()
                    await _run_ws_action_loop(websocket, session, msg.screenshot)

                elif msg_type == "interrupt":
                    if session.status not in ("running", "thinking", "done"):
                        await websocket.send_json({"type": "stopped"})
                        continue
                    msg = DesktopInterruptMessage(**data)
                    session.status = "running"
                    await _handle_desktop_interrupt(websocket, session, msg.screenshot, msg.instruction)

                elif msg_type == "stop":
                    session.abort_event.set()
                    session.status = "idle"
                    await websocket.send_json({"type": "stopped"})

            except (ValidationError, KeyError) as exc:
                await websocket.send_json({"type": "error", "message": f"Invalid message: {exc}"})

    except WebSocketDisconnect:
        logger.info("Desktop WS disconnected: session=%s", session_id)
    except Exception:
        had_error = True
        logger.exception("Desktop WS error: session=%s", session_id)
    finally:
        if had_error:
            try:
                await websocket.send_json({"type": "error", "message": "Internal server error"})
            except Exception:
                pass


async def _run_ws_action_loop(
    websocket: WebSocket,
    session: DesktopSession,
    first_screenshot: str,
    steps_remaining: Optional[int] = None,
) -> None:
    """Wrap inner loop with hard timeout."""
    try:
        await asyncio.wait_for(
            _run_ws_action_loop_inner(websocket, session, first_screenshot, steps_remaining),
            timeout=_ACTION_LOOP_TIMEOUT,
        )
    except asyncio.TimeoutError:
        session.status = "idle"
        await websocket.send_json(
            {"type": "stopped", "narration": f"Task timed out after {_ACTION_LOOP_TIMEOUT}s."}
        )


async def _run_ws_action_loop_inner(
    websocket: WebSocket,
    session: DesktopSession,
    first_screenshot: str,
    steps_remaining: Optional[int] = None,
) -> None:
    """
    Core interactive WS action loop for Desktop Mode.

    Structurally identical to webpilot_routes._run_action_loop_inner().
    """
    if steps_remaining is None:
        steps_remaining = _MAX_LOOP_STEPS

    screenshot = first_screenshot
    _prev_hash = hashlib.md5(base64.b64decode(first_screenshot)).digest()
    retry_count = 0
    step_count = 0

    while not session.abort_event.is_set():
        if step_count >= steps_remaining:
            session.status = "idle"
            await websocket.send_json(
                {"type": "stopped", "narration": "Reached maximum number of steps."}
            )
            return
        step_count += 1

        await websocket.send_json({"type": "thinking"})

        stuck = retry_count >= _MAX_RETRIES
        if stuck:
            retry_count = 0
            _prev_hash = b""

        try:
            action = await _desktop_handler.get_next_action_desktop(
                image_b64=screenshot,
                intent=session.intent,
                history=session.history,
                stuck=stuck,
                screen_width=session.screen_width,
                screen_height=session.screen_height,
            )
        except Exception as exc:
            logger.exception("Desktop handler error: session=%s", session.session_id)
            await websocket.send_json({"type": "error", "message": str(exc)})
            session.status = "idle"
            return

        action_dict = action.model_dump()

        if action.action == "done":
            await websocket.send_json({"type": "done", **action_dict})
            session.status = "done"
            return

        if action.is_irreversible or action.action == "confirm_required":
            session.status = "awaiting_confirm"
            await websocket.send_json(
                {"type": "confirmation_required", "action": action_dict, "narration": action.narration}
            )
            raw_confirm = await websocket.receive_text()
            try:
                confirm_data = json.loads(raw_confirm)
                confirmed = DesktopConfirmMessage(**confirm_data).confirmed
            except (json.JSONDecodeError, ValidationError):
                confirmed = False
            if not confirmed:
                await websocket.send_json({"type": "stopped", "narration": "Action cancelled."})
                session.status = "idle"
                return
            session.status = "running"

        await websocket.send_json({"type": "action", **action_dict})

        # Trim history
        if len(session.history) >= 20:
            session.history = session.history[-18:]

        # Wait for next screenshot from client
        raw = await websocket.receive_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            session.status = "idle"
            return

        msg_type = data.get("type")
        if msg_type == "stop":
            session.abort_event.set()
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
            return
        elif msg_type == "interrupt":
            msg = DesktopInterruptMessage(**data)
            await _handle_desktop_interrupt(websocket, session, msg.screenshot, msg.instruction)
            return
        elif msg_type == "screenshot":
            screenshot = data["screenshot"]
            # Update screen dimensions if client reports them
            if data.get("screen_width"):
                session.screen_width = data["screen_width"]
            if data.get("screen_height"):
                session.screen_height = data["screen_height"]
            new_hash = hashlib.md5(base64.b64decode(screenshot)).digest()
            if new_hash == _prev_hash:
                retry_count += 1
            else:
                retry_count = 0
            _prev_hash = new_hash
        else:
            await websocket.send_json(
                {"type": "error", "message": f"Unexpected message type: {msg_type}"}
            )
            session.status = "idle"
            return


async def _handle_desktop_interrupt(
    websocket: WebSocket,
    session: DesktopSession,
    screenshot: str,
    instruction: str,
) -> None:
    """Handle interruption — reuses WebPilotHandler.classify_interruption_type()."""
    from src.agent.webpilot_handler import WebPilotHandler
    from src.api.webpilot_models import InterruptionType

    interrupt_type = WebPilotHandler.classify_interruption_type(instruction)

    if interrupt_type == InterruptionType.ABORT:
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json(
            {"type": "stopped", "narration": "Stopped. What would you like to do?"}
        )
        return

    original_intent = session.intent
    if interrupt_type == InterruptionType.REDIRECT:
        session.history = []
        session.intent = instruction
    else:
        session.intent = f"{session.intent} ({instruction})"

    await websocket.send_json({"type": "thinking"})

    try:
        action = await _desktop_handler.get_interruption_replan(
            screenshot, original_intent, instruction, session.history, interrupt_type,
            viewport_width=session.screen_width,
            viewport_height=session.screen_height,
        )
    except Exception as exc:
        logger.exception("Desktop interrupt replan error: session=%s", session.session_id)
        await websocket.send_json({"type": "error", "message": str(exc)})
        session.status = "idle"
        return

    action_dict = action.model_dump()
    if action.action == "done":
        await websocket.send_json({"type": "done", **action_dict})
        session.status = "done"
        return

    await websocket.send_json({"type": "action", **action_dict})

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Timeout waiting for screenshot after interrupt"})
        session.status = "idle"
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "Invalid JSON after interrupt"})
        session.status = "idle"
        return

    if data.get("type") == "screenshot":
        await _run_ws_action_loop(
            websocket, session, data["screenshot"],
            steps_remaining=max(1, _MAX_LOOP_STEPS // 2),
        )
        if session.status not in ("done", "idle"):
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
    elif data.get("type") == "stop":
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json({"type": "stopped"})
