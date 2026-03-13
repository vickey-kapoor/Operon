"""WebPilot API router — WebSocket-driven single-action browser control."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.api.webpilot_models import (
    ConfirmMessage,
    InterruptMessage,
    InterruptionType,
    ResumeMessage,
    StopMessage,
    TaskMessage,
    TTSRequest,
    WebPilotSession,
    ScreenshotMessage,
)
from src.agent.webpilot_handler import WebPilotHandler, LegacyWebPilotHandler

_MAX_SESSION_DURATION = int(os.environ.get("MAX_SESSION_DURATION", "1800"))
_MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Flag: True if Live API is available (google-genai client was set up during lifespan).
_live_api_client = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webpilot", tags=["webpilot"])

_sessions: Dict[str, WebPilotSession] = {}
# Shared handler — either LegacyWebPilotHandler or stub. Used for TTS and as fallback.
_handler = None


def init_handler(handler, live_client=None) -> None:
    """Inject the shared handler instance. Called from the server lifespan.

    Parameters
    ----------
    handler:
        The shared LegacyWebPilotHandler (or stub) — used for TTS and as fallback.
    live_client:
        Optional google.genai Client for creating per-session Live API handlers.
    """
    global _handler, _live_api_client
    _handler = handler
    _live_api_client = live_client


async def cleanup_sessions() -> None:
    """Remove sessions inactive for more than 30 minutes. Run as a background task."""
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        cutoff = time.time() - _MAX_SESSION_DURATION
        stale = [sid for sid, s in list(_sessions.items()) if s.last_active < cutoff]
        for sid in stale:
            session = _sessions.pop(sid)
            if session.handler:
                try:
                    await session.handler.close()
                except Exception:
                    pass
            logger.info("Cleaned up stale webpilot session", extra={"session_id": sid})


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_session() -> dict:
    """Create a new WebPilot session. Returns session_id."""
    _MAX_SESSIONS = 1000
    if len(_sessions) >= _MAX_SESSIONS:
        raise HTTPException(status_code=503, detail="Maximum session limit reached")
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


@router.get("/debug/stub_calls")
async def debug_stub_calls() -> dict:
    """Return stub call log. Only available when WEBPILOT_STUB env var is set."""
    from src.agent.webpilot_stub import WebPilotStubHandler
    if not isinstance(_handler, WebPilotStubHandler):
        raise HTTPException(status_code=404, detail="Not in stub mode")
    return {"calls": _handler.call_log}


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
            # Reject oversized messages (> 15 MB).
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
                    msg = TaskMessage(**data)
                    session.intent = msg.intent
                    session.history = []
                    session.status = "running"
                    session.abort_event.clear()
                    # Try per-session Live handler; falls back to shared Legacy handler.
                    session.handler = await _create_live_handler(msg.intent)
                    await _run_action_loop(websocket, session, msg.screenshot)

                elif msg_type == "interrupt":
                    if session.status not in ("running", "thinking", "done"):
                        await websocket.send_json({"type": "stopped", "narration": "No active task to interrupt."})
                        continue
                    msg = InterruptMessage(**data)
                    session.status = "running"
                    await _handle_interrupt(websocket, session, msg.screenshot, msg.instruction)

                elif msg_type == "stop":
                    session.abort_event.set()
                    session.status = "idle"
                    if session.handler:
                        await session.handler.close()
                        session.handler = None
                    await websocket.send_json({"type": "stopped"})

            except (ValidationError, KeyError) as exc:
                await websocket.send_json({"type": "error", "message": f"Invalid message: {exc}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", extra={"session_id": session_id})
    except Exception:
        logger.exception("WebSocket error", extra={"session_id": session_id})
    finally:
        # Close per-session Live handler on disconnect to free resources.
        if session.handler:
            try:
                await session.handler.close()
            except Exception:
                pass
            session.handler = None
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-session Live handler factory
# ---------------------------------------------------------------------------


async def _create_live_handler(intent: str) -> WebPilotHandler | None:
    """Create a per-session Live API handler and open its streaming session.

    Returns a connected ``WebPilotHandler`` on success, or ``None`` if the
    Live API is unavailable or connection fails (caller falls back to the
    shared Legacy handler).
    """
    if _live_api_client is None:
        return None
    handler = WebPilotHandler(client=_live_api_client)
    try:
        await handler.connect(intent)
        return handler
    except Exception as exc:
        logger.warning("Live API handler creation failed, using legacy fallback: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Action loop helpers
# ---------------------------------------------------------------------------


_ACTION_LOOP_TIMEOUT = int(os.environ.get("ACTION_LOOP_TIMEOUT", "120"))
_MAX_LOOP_STEPS = int(os.environ.get("MAX_LOOP_STEPS", "30"))


async def _run_action_loop(
    websocket: WebSocket,
    session: WebPilotSession,
    first_screenshot: str,
    steps_remaining: int | None = None,
) -> None:
    """
    Core action loop: ask Gemini for the next action, emit it, wait for a screenshot.

    Runs until:
      - action="done" is returned
      - session.abort_event is set
      - the client sends {"type": "stop"}
      - steps_remaining is exhausted
      - hard timeout fires (ACTION_LOOP_TIMEOUT seconds)
      - an error occurs

    Auto-retry: tracks MD5 hashes of consecutive screenshots. After 3 identical
    screenshots, injects a "stuck" hint into the Gemini prompt to encourage a new approach.
    """
    try:
        await asyncio.wait_for(
            _run_action_loop_inner(websocket, session, first_screenshot, steps_remaining),
            timeout=_ACTION_LOOP_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Action loop hard timeout (%ds) — forcing stop",
            _ACTION_LOOP_TIMEOUT,
            extra={"session_id": session.session_id},
        )
        session.status = "idle"
        await websocket.send_json(
            {"type": "stopped", "narration": f"Task timed out after {_ACTION_LOOP_TIMEOUT} seconds."}
        )


async def _run_action_loop_inner(
    websocket: WebSocket,
    session: WebPilotSession,
    first_screenshot: str,
    steps_remaining: int | None = None,
) -> None:
    """Inner implementation of the action loop (wrapped by timeout in _run_action_loop)."""
    if steps_remaining is None:
        steps_remaining = _MAX_LOOP_STEPS
    screenshot = first_screenshot
    current_url = ""
    _prev_hash = hashlib.md5(base64.b64decode(first_screenshot)).digest()
    retry_count = 0
    verification_attempts = 0
    step_count = 0

    while not session.abort_event.is_set():
        if step_count >= steps_remaining:
            logger.info(
                "Step budget exhausted (%d steps)", steps_remaining,
                extra={"session_id": session.session_id},
            )
            session.status = "idle"
            await websocket.send_json(
                {"type": "stopped", "narration": "Reached maximum number of steps."}
            )
            return
        step_count += 1
        logger.info(
            "Action loop step %d/%d starting",
            step_count, steps_remaining,
            extra={"session_id": session.session_id},
        )

        await websocket.send_json({"type": "thinking"})

        stuck = retry_count >= _MAX_RETRIES
        if stuck:
            retry_count = 0
            _prev_hash = b""  # force fresh baseline so next screenshot isn't flagged stuck

        try:
            if session.handler is not None:
                # Live API — persistent session, no manual history management
                action = await session.handler.send_screenshot_and_get_action(
                    screenshot, session.intent, stuck=stuck,
                    current_url=current_url,
                )
            else:
                # Legacy — per-call generate_content with explicit history
                action = await _handler.get_next_action(
                    screenshot, session.intent, session.history, stuck=stuck,
                    current_url=current_url,
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
            # --- Gap 2: Completion verification ---
            if verification_attempts < 2:
                verified = await _verify_completion(
                    websocket, session, screenshot, action
                )
                if not verified:
                    verification_attempts += 1
                    continue
            await websocket.send_json({"type": "done", **action_dict})
            session.status = "done"
            return

        # Reset verification attempts on non-done actions
        verification_attempts = 0

        # --- Gap 3+4: CAPTCHA / login pause ---
        if action.action in ("captcha_detected", "login_required"):
            reason = "captcha" if action.action == "captcha_detected" else "login"
            session.status = "paused"
            await websocket.send_json({
                "type": "paused",
                "reason": reason,
                "narration": action.narration,
                "action_label": action.action_label,
            })
            # Inline read — same pattern as confirmation flow
            raw_pause = await websocket.receive_text()
            pause_data = json.loads(raw_pause)
            if pause_data.get("type") == "resume":
                screenshot = pause_data["screenshot"]
                session.status = "running"
                continue
            elif pause_data.get("type") == "stop":
                session.abort_event.set()
                session.status = "idle"
                await websocket.send_json({"type": "stopped"})
                return
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Expected 'resume' or 'stop', got '{pause_data.get('type')}'"}
                )
                session.status = "idle"
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
        logger.info(
            "Waiting for next message from client",
            extra={"session_id": session.session_id},
        )
        raw = await websocket.receive_text()
        logger.info(
            "Received message type=%s size=%d",
            "?", len(raw),
            extra={"session_id": session.session_id},
        )
        data = json.loads(raw)
        msg_type = data.get("type")
        logger.info(
            "Parsed message type=%s",
            msg_type,
            extra={"session_id": session.session_id},
        )

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
            current_url = data.get("current_url", "")
            logger.info(
                "Screenshot received, b64_len=%d current_url=%s",
                len(screenshot), current_url,
                extra={"session_id": session.session_id},
            )
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


async def _verify_completion(
    websocket: WebSocket,
    session: WebPilotSession,
    screenshot: str,
    action,
) -> bool:
    """
    Ask the handler to verify that the task was actually completed.
    Returns True if verified (or on error), False if not yet done.
    """
    try:
        handler = session.handler if session.handler is not None else _handler
        verified = await handler.verify_completion(screenshot, session.intent)
        if not verified:
            logger.info(
                "Completion not verified — retrying",
                extra={"session_id": session.session_id},
            )
            await websocket.send_json({
                "type": "thinking",
                "detail": "Verifying completion...",
            })
        return verified
    except Exception as exc:
        logger.warning(
            "Completion verification error, accepting done: %s",
            exc,
            extra={"session_id": session.session_id},
        )
        return True


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
        if session.handler is not None:
            # Live API — send interruption context, then get next action
            await session.handler.send_interruption(instruction)
            action = await session.handler.send_screenshot_and_get_action(
                screenshot, session.intent
            )
        else:
            # Legacy — single replan call
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
        # Inherit remaining step budget (half of max) so recursive call can't run forever.
        await _run_action_loop(
            websocket, session, data["screenshot"],
            steps_remaining=max(1, _MAX_LOOP_STEPS // 2),
        )
        # Ensure a terminal message was sent — if the loop returned without
        # setting done/idle, force a stopped so the sidebar never hangs.
        if session.status not in ("done", "idle"):
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
    elif data.get("type") == "stop":
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json({"type": "stopped"})
