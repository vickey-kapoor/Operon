"""Schemas for policy decisions and agent actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from src.models.common import StrictModel
from src.models.selector import TargetSelectionContext


class ActionType(StrEnum):
    """Supported browser actions for the MVP workflow."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    PRESS_KEY = "press_key"
    NAVIGATE = "navigate"
    WAIT = "wait"
    WAIT_FOR_USER = "wait_for_user"
    STOP = "stop"


class AgentAction(StrictModel):
    """A single execution-focused browser action selected by policy."""

    action_type: ActionType
    selector: str | None = Field(default=None, min_length=1)
    target_element_id: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    key: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1)
    wait_ms: int | None = Field(default=None, ge=1, le=30000)
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    target_context: TargetSelectionContext | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "AgentAction":
        """Ensure action payloads contain only the fields valid for the action type."""
        if self.action_type is ActionType.CLICK:
            if self.selector is None and (self.x is None or self.y is None) and self.target_element_id is None:
                raise ValueError("click requires selector, coordinates, or target_element_id")
            if self.text is not None or self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("click cannot include text, key, url, or wait_ms")
        elif self.action_type is ActionType.TYPE:
            if self.text is None:
                raise ValueError("type requires text")
            if self.selector is None and self.target_element_id is None and (self.x is None or self.y is None):
                raise ValueError("type requires selector, coordinates, or target_element_id")
            if self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("type cannot include key, url, or wait_ms")
        elif self.action_type is ActionType.SELECT:
            if self.text is None:
                raise ValueError("select requires text")
            if self.selector is None and self.target_element_id is None and (self.x is None or self.y is None):
                raise ValueError("select requires selector, coordinates, or target_element_id")
            if self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("select cannot include key, url, or wait_ms")
        elif self.action_type is ActionType.PRESS_KEY:
            if self.key is None:
                raise ValueError("press_key requires key")
            if self.selector is not None or self.text is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("press_key cannot include selector, text, url, or wait_ms")
        elif self.action_type is ActionType.NAVIGATE:
            if self.url is None:
                raise ValueError("navigate requires url")
            if self.selector is not None or self.text is not None or self.key is not None or self.wait_ms is not None:
                raise ValueError("navigate cannot include selector, text, key, or wait_ms")
        elif self.action_type is ActionType.WAIT:
            if self.wait_ms is None:
                raise ValueError("wait requires wait_ms")
            if self.selector is not None or self.text is not None or self.key is not None or self.url is not None:
                raise ValueError("wait cannot include selector, text, key, or url")
        elif self.action_type is ActionType.WAIT_FOR_USER:
            if self.text is None:
                raise ValueError("wait_for_user requires text (reason for user)")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.key, self.url, self.wait_ms, self.x, self.y, self.target_context)
            ):
                raise ValueError("wait_for_user cannot include action payload fields other than text")
        elif self.action_type is ActionType.STOP:
            if any(
                value is not None
                for value in (
                    self.selector,
                    self.target_element_id,
                    self.text,
                    self.key,
                    self.url,
                    self.wait_ms,
                    self.x,
                    self.y,
                    self.target_context,
                )
            ):
                raise ValueError("stop cannot include action payload fields")
        return self


class PolicyDecision(StrictModel):
    """The next action decision chosen from current perception plus agent state."""

    action: AgentAction
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    active_subgoal: str = Field(min_length=1)
