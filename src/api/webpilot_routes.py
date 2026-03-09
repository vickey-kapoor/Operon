"""WebPilot API router — WebSocket-driven single-action browser control."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.api.webpilot_models import (
    ConfirmMessage,
    InterruptMessage,
    InterruptionType,
    StopMessage,
    TaskMessage,
    TTSRequest,
    WebPilotSession,
    ScreenshotMessage,
)
from src.agent.webpilot_handler import WebPilotHandler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webpilot", tags=["webpilot"])

_sessions: Dict[str, WebPilotSession] = {}
_handler: WebPilotHandler = None  # initialized in lifespan via init_handler()


def init_handler(handler: WebPilotHandler) -> None:
    """Inject the shared WebPilotHandler instance. Called from the server lifespan."""
    global _handler
    _handler = handler


async def cleanup_sessions() -> None:
    """Remove sessions inactive for more than 30 minutes. Run as a background task."""
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        cutoff = time.time() - 1800  # 30 minutes
        stale = [sid for sid, s in list(_sessions.items()) if s.last_active < cutoff]
        for sid in stale:
            del _sessions[sid]
            logger.info("Cleaned up stale webpilot session", extra={"session_id": sid})


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_session() -> dict:
    """Create a new WebPilot session. Returns session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = WebPilotSession(session_id=session_id)
    logger.info("Created webpilot session", extra={"session_id": session_id})
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete an existing WebPilot session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "deleted"}


@router.post("/tts")
async def tts_narration(body: TTSRequest) -> dict:
    """Generate speech audio via Gemini TTS. Returns base64 WAV audio."""
    if _handler is None:
        raise HTTPException(status_code=503, detail="Handler not initialized")
    try:
        audio_bytes = await _handler.get_narration_audio(body.text)
        return {"audio": base64.b64encode(audio_bytes).decode(), "mime_type": "audio/wav"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for real-time WebPilot agent control.

    Incoming message types:
      - {"type": "task", "intent": str, "screenshot": base64}
      - {"type": "screenshot", "screenshot": base64}
      - {"type": "interrupt", "instruction": str, "screenshot": base64}
      - {"type": "confirm", "confirmed": bool}
      - {"type": "stop"}

    Outgoing message types:
      - {"type": "thinking"}
      - {"type": "action", ...WebPilotAction fields...}
      - {"type": "confirmation_required", "action": dict, "narration": str}
      - {"type": "done", ...WebPilotAction fields...}
      - {"type": "stopped"}
      - {"type": "error", "message": str}
    """
    if session_id not in _sessions:
        await websocket.accept()
        await websocket.close(code=4404, reason="Session not found")
        return

    if _handler is None:
        await websocket.accept()
        await websocket.close(code=4503, reason="WebPilot handler not initialised")
        return

    session = _sessions[session_id]
    await websocket.accept()
    logger.info("WebSocket connected", extra={"session_id": session_id})

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")
            session.last_active = time.time()

            try:
                if msg_type == "task":
                    msg = TaskMessage(**data)
                    session.intent = msg.intent
                    session.history = []
                    session.status = "running"
                    session.abort_event.clear()
                    await _run_action_loop(websocket, session, msg.screenshot)

                elif msg_type == "interrupt":
                    msg = InterruptMessage(**data)
                    session.confirm_event.clear()
                    session.confirm_result = None
                    session.status = "running"
                    await _handle_interrupt(websocket, session, msg.screenshot, msg.instruction)

                elif msg_type == "confirm":
                    msg = ConfirmMessage(**data)
                    session.confirm_result = msg.confirmed
                    session.confirm_event.set()

                elif msg_type == "stop":
                    session.abort_event.set()
                    session.status = "idle"
                    await websocket.send_json({"type": "stopped"})

            except (ValidationError, KeyError) as exc:
                await websocket.send_json({"type": "error", "message": f"Invalid message: {exc}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", extra={"session_id": session_id})
    except Exception:
        logger.exception("WebSocket error", extra={"session_id": session_id})
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Action loop helpers
# ---------------------------------------------------------------------------


async def _run_action_loop(
    websocket: WebSocket,
    session: WebPilotSession,
    first_screenshot: str,
) -> None:
    """
    Core action loop: ask Gemini for the next action, emit it, wait for a screenshot.

    Runs until:
      - action="done" is returned
      - session.abort_event is set
      - the client sends {"type": "stop"}
      - an error occurs

    Auto-retry: tracks MD5 hashes of consecutive screenshots. After 3 identical
    screenshots, injects a "stuck" hint into the Gemini prompt to encourage a new approach.
    """
    screenshot = first_screenshot
    _prev_hash = hashlib.md5(base64.b64decode(first_screenshot)).digest()
    retry_count = 0

    while not session.abort_event.is_set():
        await websocket.send_json({"type": "thinking"})

        stuck = retry_count >= 3
        if stuck:
            retry_count = 0

        try:
            action = await _handler.get_next_action(
                screenshot, session.intent, session.history, stuck=stuck
            )
        except Exception as exc:
            logger.exception(
                "WebPilot handler error", extra={"session_id": session.session_id}
            )
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
                {
                    "type": "confirmation_required",
                    "action": action_dict,
                    "narration": action.narration,
                }
            )
            # Read the confirm response directly — the outer loop is blocked here
            # and cannot process messages, so we must receive the confirm inline.
            raw_confirm = await websocket.receive_text()
            confirm_data = json.loads(raw_confirm)
            try:
                confirm_msg = ConfirmMessage(**confirm_data)
                confirmed = confirm_msg.confirmed
            except (ValidationError, KeyError):
                confirmed = False
            if not confirmed:
                await websocket.send_json(
                    {"type": "stopped", "narration": "Action cancelled by user."}
                )
                session.status = "idle"
                return
            session.status = "running"

        await websocket.send_json({"type": "action", **action_dict})

        # Bound history to 10 turns (20 items: user + assistant pairs).
        if len(session.history) >= 20:
            session.history = session.history[-18:]

        # Wait for the next screenshot (or a control message).
        raw = await websocket.receive_text()
        data = json.loads(raw)
        msg_type = data.get("type")

        if msg_type == "stop":
            session.abort_event.set()
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
            return
        elif msg_type == "interrupt":
            msg = InterruptMessage(**data)
            await _handle_interrupt(websocket, session, msg.screenshot, msg.instruction)
            return
        elif msg_type == "screenshot":
            screenshot = data["screenshot"]
            new_hash = hashlib.md5(base64.b64decode(screenshot)).digest()
            if new_hash == _prev_hash:
                retry_count += 1
            else:
                retry_count = 0
            _prev_hash = new_hash
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Unexpected message type: {msg_type}",
                }
            )
            return


async def _handle_interrupt(
    websocket: WebSocket,
    session: WebPilotSession,
    screenshot: str,
    instruction: str,
) -> None:
    """
    Handle a mid-task interruption: classify type, update session state, replan and continue.
    """
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
        session.history = []       # fresh start — new goal
        session.intent = instruction
    elif interrupt_type == InterruptionType.REFINEMENT:
        session.intent = f"{session.intent} ({instruction})"  # merge constraint
        # keep history

    await websocket.send_json({"type": "thinking"})

    try:
        action = await _handler.get_interruption_replan(
            screenshot, original_intent, instruction, session.history, interrupt_type
        )
    except Exception as exc:
        logger.exception(
            "WebPilot interruption replan error", extra={"session_id": session.session_id}
        )
        await websocket.send_json({"type": "error", "message": str(exc)})
        session.status = "idle"
        return

    action_dict = action.model_dump()

    if action.action == "done":
        await websocket.send_json({"type": "done", **action_dict})
        session.status = "done"
        return

    await websocket.send_json({"type": "action", **action_dict})

    # Wait for the next screenshot to continue the loop (30s timeout guards against
    # the extension failing to send a screenshot after executing the replanned action).
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout waiting for screenshot after interrupt replan",
            extra={"session_id": session.session_id},
        )
        await websocket.send_json(
            {"type": "error", "message": "Timed out waiting for browser response after interrupt."}
        )
        session.status = "idle"
        return
    data = json.loads(raw)
    if data.get("type") == "screenshot":
        await _run_action_loop(websocket, session, data["screenshot"])
    elif data.get("type") == "stop":
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json({"type": "stopped"})
