"""Pydantic v2 models and dataclasses for the WebPilot API."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class InterruptionType(str, Enum):
    REFINEMENT = "refinement"  # constraint added, original goal preserved
    REDIRECT = "redirect"      # completely new goal
    ABORT = "abort"            # stop everything


class WebPilotAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    observation: Optional[str] = None
    plan: Optional[List[str]] = None
    steps_completed: Optional[int] = None
    action: Literal["click", "right_click", "double_click", "move", "type", "scroll", "wait", "navigate", "key", "done", "confirm_required", "captcha_detected", "login_required"]
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    url: Optional[str] = None
    direction: Optional[Literal["up", "down"]] = None
    duration: Optional[int] = None
    narration: str
    action_label: str
    is_irreversible: bool = False


@dataclass
class WebPilotSession:
    session_id: str
    intent: Optional[str] = None
    history: List = field(default_factory=list)
    status: str = "idle"
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Note: confirm flow is handled inline in _run_action_loop via direct
    # websocket.receive_text() — the outer loop is blocked while waiting,
    # so confirm_event/confirm_result were dead code and have been removed.
    last_active: float = field(default_factory=time.time)


# WS incoming message schemas
class TaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["task"]
    intent: str
    screenshot: str  # base64


class ScreenshotMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["screenshot"]
    screenshot: str  # base64


class InterruptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["interrupt"]
    instruction: str
    screenshot: str  # base64


class ConfirmMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["confirm"]
    confirmed: bool


class StopMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["stop"]


class ResumeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["resume"]
    screenshot: str  # base64


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=5000)
