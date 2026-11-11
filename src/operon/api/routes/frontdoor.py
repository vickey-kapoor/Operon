"""Front-door routes (KAN-89): the local task page's API under ``/app/api``.

These sit behind the API key like every other route. The page itself (``GET /app``)
is served without the key because it holds no secret: it asks for the key, or reads
it from the URL fragment that ``operon serve`` prints (a fragment never reaches the
server or its logs).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import Field

from operon.api import frontdoor, stream_tokens
from operon.api.dependencies import test_safe_mode_enabled, validate_run_id
from operon.models.common import StrictModel

router = APIRouter(prefix="/app/api")


class NoteRequest(StrictModel):
    note: str | None = Field(default=None, max_length=1000)


class StreamTokenRequest(StrictModel):
    scope: str = Field(default="control", pattern=r"^(view|control)$")


class UserInputRequest(StrictModel):
    input_type: str = Field(pattern=r"^(click|dblclick|type|key|scroll)$")
    x_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    y_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    text: str = Field(default="", max_length=2000)
    key: str = Field(default="", max_length=40)
    delta_y: int = Field(default=300, ge=-5000, le=5000)


def _http(exc: frontdoor.FrontDoorError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def start(task: frontdoor.FrontDoorTask) -> dict[str, Any]:
    """Start a task in the browser or on the desktop and step it in the background."""
    if task.environment == "browser" and not test_safe_mode_enabled():
        from operon.api.dependencies import ensure_cdp_ready

        await ensure_cdp_ready(mode="observable")
    try:
        return await frontdoor.start_task(task, auto_run=not test_safe_mode_enabled())
    except ValueError as exc:  # StartUrlBlockedError and friends
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
async def get_status(run_id: str) -> dict[str, Any]:
    validate_run_id(run_id)
    try:
        return await frontdoor.run_status(run_id)
    except frontdoor.FrontDoorError as exc:
        raise _http(exc) from exc


@router.get("/runs/{run_id}/report")
async def get_report(run_id: str) -> dict[str, Any]:
    validate_run_id(run_id)
    try:
        return await frontdoor.build_report(run_id)
    except frontdoor.FrontDoorError as exc:
        raise _http(exc) from exc


async def _call(fn, run_id: str, *args) -> dict[str, Any]:
    validate_run_id(run_id)
    try:
        return await fn(run_id, *args)
    except frontdoor.FrontDoorError as exc:
        raise _http(exc) from exc


@router.post("/runs/{run_id}/approve")
async def approve(run_id: str) -> dict[str, Any]:
    return await _call(frontdoor.approve, run_id)


@router.post("/runs/{run_id}/decline")
async def decline(run_id: str, body: NoteRequest | None = None) -> dict[str, Any]:
    return await _call(frontdoor.decline, run_id, body.note if body else None)


@router.post("/runs/{run_id}/takeover")
async def takeover(run_id: str) -> dict[str, Any]:
    return await _call(frontdoor.take_over, run_id)


@router.post("/runs/{run_id}/handback")
async def handback(run_id: str, body: NoteRequest | None = None) -> dict[str, Any]:
    return await _call(frontdoor.hand_back, run_id, body.note if body else None)


@router.post("/runs/{run_id}/stop")
async def stop(run_id: str) -> dict[str, Any]:
    return await _call(frontdoor.stop, run_id)


@router.post("/runs/{run_id}/input")
async def user_input(run_id: str, body: UserInputRequest) -> dict[str, Any]:
    return await _call(frontdoor.user_input, run_id, body.model_dump())


@router.post("/stream-token")
async def stream_token(body: StreamTokenRequest | None = None) -> dict[str, Any]:
    """A short-lived token for ``/ws/stream`` (subprotocols ``["x-stream-token", token]``)."""
    token = stream_tokens.mint(body.scope if body else "control")
    return {"token": token.token, "scope": token.scope, "expires_at": token.expires_at}
