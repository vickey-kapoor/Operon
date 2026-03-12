import base64
import binascii
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.agent import adk_agent
from src import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["extension"])


class CreateSessionResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded PNG screenshot")
    task: str = Field(..., max_length=2000)


class SessionEventRequest(BaseModel):
    """Telemetry event sent from the Chrome Extension client."""

    event: str = Field(..., max_length=100, description="Event type name")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional event payload")


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def create_session() -> CreateSessionResponse:
    session_id = await adk_agent.create_session()
    return CreateSessionResponse(session_id=session_id)


@router.post("/{session_id}/step")
async def session_step(session_id: str, body: StepRequest) -> dict:
    if not await adk_agent.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id!r} not found")
    if len(body.image_b64) > 10 * 1024 * 1024:
        raise HTTPException(413, "image_b64 exceeds 10 MB limit")
    try:
        base64.b64decode(body.image_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, f"Invalid base64: {e}") from e
    plan = await adk_agent.step(session_id, body.image_b64, body.task)
    return plan.model_dump()


@router.post("/{session_id}/events", status_code=204)
async def session_event(session_id: str, body: SessionEventRequest) -> Response:
    if not await adk_agent.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id!r} not found")
    metrics.emit("ext_client_event", labels={"event": body.event})
    logger.info(
        "ext_client_event",
        extra={"session_id": session_id, "event": body.event, "data": body.data},
    )
    return Response(status_code=204)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str) -> Response:
    existed = await adk_agent.delete_session(session_id)
    if not existed:
        raise HTTPException(404, f"Session {session_id!r} not found")
    return Response(status_code=204)
