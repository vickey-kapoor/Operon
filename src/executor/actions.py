"""Action type definitions and schemas using Pydantic."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    """Enumeration of all supported browser actions."""

    CLICK = "click"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    DONE = "done"


class Action(BaseModel):
    """A single action to be executed by the browser executor."""

    type: ActionType = Field(..., description="The type of action to perform")
    coordinate: Optional[List[int]] = Field(
        None,
        description="[x, y] pixel coordinates from top-left of 1280x800 viewport",
        min_length=2,
        max_length=2,
    )
    text: Optional[str] = Field(
        None,
        description="Text to type (for TYPE action)",
        max_length=10000,
    )
    key: Optional[str] = Field(
        None,
        description="Key to press (for KEY action), e.g. 'Enter', 'Tab', 'Escape'",
    )
    scroll_direction: Optional[str] = Field(
        None,
        description="Direction to scroll: up, down, left, right",
    )
    scroll_amount: Optional[int] = Field(
        None,
        description="Number of scroll units (default 3)",
        ge=1,
        le=20,
    )
    url: Optional[str] = Field(
        None,
        description="URL to navigate to (for NAVIGATE action)",
    )
    duration_ms: Optional[int] = Field(
        None,
        description="Duration to wait in milliseconds (for WAIT action)",
        ge=0,
        le=5000,
    )
    description: str = Field(
        "",
        description="Human-readable description of what this action does",
    )

    model_config = ConfigDict(use_enum_values=True)


class ActionResult(BaseModel):
    """Result of executing a single action."""

    success: bool = Field(..., description="Whether the action completed successfully")
    screenshot: Optional[str] = Field(
        None,
        description="Base64-encoded PNG screenshot taken after the action (if applicable)",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the action failed",
    )
    action_type: Optional[str] = Field(
        None,
        description="The type of action that was executed",
    )
