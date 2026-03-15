"""Pydantic models and session dataclass for Desktop Mode."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action model — extends WebPilotAction with desktop-specific actions
# ---------------------------------------------------------------------------

class DesktopAction(BaseModel):
    """
    Single-action response from Gemini for desktop control.

    Extends the WebPilot action schema with right_click, double_click, and move.
    The 'navigate' action type is intentionally excluded — it is browser-only.
    """
    model_config = ConfigDict(extra="ignore")

    observation: Optional[str] = None
    plan: Optional[List[str]] = None
    steps_completed: Optional[int] = None

    action: Literal[
        "click",
        "right_click",
        "double_click",
        "move",
        "type",
        "key",
        "scroll",
        "wait",
        "done",
        "confirm_required",
    ]

    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = Field(None, max_length=10000)
    direction: Optional[Literal["up", "down", "left", "right"]] = None
    duration: Optional[int] = Field(None, ge=0, le=10000)  # ms, wider range for desktop

    narration: str
    action_label: str
    is_irreversible: bool = False


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

@dataclass
class DesktopSession:
    """State for a single Desktop Mode agent session."""
    session_id: str
    intent: Optional[str] = None
    history: List = field(default_factory=list)
    status: str = "idle"          # idle | running | thinking | paused | done
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_active: float = field(default_factory=time.time)
    # Pixel coordinates of the screen at session creation time.
    screen_width: int = 1920
    screen_height: int = 1080
    # Scale factors: logical-to-physical pixel ratios (for HiDPI screens).
    scale_x: float = 1.0
    scale_y: float = 1.0


# ---------------------------------------------------------------------------
# REST request / response models
# ---------------------------------------------------------------------------

class DesktopStartRequest(BaseModel):
    """Request body for POST /desktop/start (server-side autonomous execution)."""
    intent: str = Field(..., min_length=1, max_length=2000,
                        description="High-level task for the AI agent")
    max_steps: int = Field(20, ge=1, le=50, description="Max agent loop steps")


class DesktopStartResponse(BaseModel):
    """Response from POST /desktop/start."""
    session_id: str
    status: str          # always "started"
    screen_width: int
    screen_height: int


class DesktopSessionStatusResponse(BaseModel):
    """Response from GET /desktop/sessions/{session_id}."""
    session_id: str
    status: str
    intent: Optional[str]
    steps_taken: int
    result: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# WS incoming message schemas (mirrors WebPilot protocol exactly)
# ---------------------------------------------------------------------------

class DesktopTaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["task"]
    intent: str
    screenshot: str          # base64 PNG


class DesktopScreenshotMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["screenshot"]
    screenshot: str          # base64 PNG
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None


class DesktopInterruptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["interrupt"]
    instruction: str
    screenshot: str          # base64 PNG


class DesktopConfirmMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["confirm"]
    confirmed: bool


class DesktopStopMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["stop"]
