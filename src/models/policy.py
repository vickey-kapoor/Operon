"""Schemas for policy decisions and agent actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from src.models.common import StrictModel
from src.models.selector import TargetSelectionContext


class ActionType(StrEnum):
    """Supported actions for browser and desktop automation."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SELECT = "select"
    PRESS_KEY = "press_key"
    NAVIGATE = "navigate"
    WAIT = "wait"
    WAIT_FOR_USER = "wait_for_user"
    STOP = "stop"
    LAUNCH_APP = "launch_app"
    HOTKEY = "hotkey"
    DRAG = "drag"
    SCROLL = "scroll"
    HOVER = "hover"
    READ_CLIPBOARD = "read_clipboard"
    WRITE_CLIPBOARD = "write_clipboard"
    SCREENSHOT_REGION = "screenshot_region"


class AgentAction(StrictModel):
    """A single execution-focused action selected by policy."""

    action_type: ActionType
    selector: str | None = Field(default=None, min_length=1)
    target_element_id: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    key: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1)
    wait_ms: int | None = Field(default=None, ge=1, le=30000)
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    x_end: int | None = Field(default=None, ge=0)
    y_end: int | None = Field(default=None, ge=0)
    scroll_amount: int | None = Field(default=None)
    target_context: TargetSelectionContext | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "AgentAction":  # noqa: C901
        """Ensure action payloads contain only the fields valid for the action type."""
        _new_fields = (self.x_end, self.y_end, self.scroll_amount)

        if self.action_type is ActionType.CLICK:
            if self.selector is None and (self.x is None or self.y is None) and self.target_element_id is None:
                raise ValueError("click requires selector, coordinates, or target_element_id")
            if self.text is not None or self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("click cannot include text, key, url, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError("click cannot include x_end, y_end, or scroll_amount")

        elif self.action_type in (ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
            label = self.action_type.value
            if self.selector is None and (self.x is None or self.y is None) and self.target_element_id is None:
                raise ValueError(f"{label} requires selector, coordinates, or target_element_id")
            if self.text is not None or self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError(f"{label} cannot include text, key, url, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError(f"{label} cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.TYPE:
            if self.text is None:
                raise ValueError("type requires text")
            if self.selector is None and self.target_element_id is None and (self.x is None or self.y is None):
                raise ValueError("type requires selector, coordinates, or target_element_id")
            if self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("type cannot include key, url, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError("type cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.SELECT:
            if self.text is None:
                raise ValueError("select requires text")
            if self.selector is None and self.target_element_id is None and (self.x is None or self.y is None):
                raise ValueError("select requires selector, coordinates, or target_element_id")
            if self.key is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("select cannot include key, url, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError("select cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.PRESS_KEY:
            if self.key is None:
                raise ValueError("press_key requires key")
            if self.selector is not None or self.text is not None or self.url is not None or self.wait_ms is not None:
                raise ValueError("press_key cannot include selector, text, url, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError("press_key cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.NAVIGATE:
            if self.url is None:
                raise ValueError("navigate requires url")
            if self.selector is not None or self.text is not None or self.key is not None or self.wait_ms is not None:
                raise ValueError("navigate cannot include selector, text, key, or wait_ms")
            if any(v is not None for v in _new_fields):
                raise ValueError("navigate cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.WAIT:
            if self.wait_ms is None:
                raise ValueError("wait requires wait_ms")
            if self.selector is not None or self.text is not None or self.key is not None or self.url is not None:
                raise ValueError("wait cannot include selector, text, key, or url")
            if any(v is not None for v in _new_fields):
                raise ValueError("wait cannot include x_end, y_end, or scroll_amount")

        elif self.action_type is ActionType.WAIT_FOR_USER:
            if self.text is None:
                raise ValueError("wait_for_user requires text (reason for user)")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.key, self.url, self.wait_ms, self.x, self.y, self.target_context, *_new_fields)
            ):
                raise ValueError("wait_for_user cannot include action payload fields other than text")

        elif self.action_type is ActionType.STOP:
            if any(
                value is not None
                for value in (
                    self.selector, self.target_element_id, self.text, self.key,
                    self.url, self.wait_ms, self.x, self.y, self.target_context,
                    *_new_fields,
                )
            ):
                raise ValueError("stop cannot include action payload fields")

        elif self.action_type is ActionType.LAUNCH_APP:
            if self.text is None:
                raise ValueError("launch_app requires text (application name)")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.key, self.url, self.wait_ms, self.x, self.y, *_new_fields)
            ):
                raise ValueError("launch_app cannot include selector, target, key, url, wait_ms, coordinates, or desktop fields")

        elif self.action_type is ActionType.HOTKEY:
            if self.key is None:
                raise ValueError("hotkey requires key (e.g. 'win+r', 'ctrl+c')")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.text, self.url, self.wait_ms, *_new_fields)
            ):
                raise ValueError("hotkey cannot include selector, target, text, url, wait_ms, or desktop fields")

        elif self.action_type is ActionType.DRAG:
            if self.x is None or self.y is None or self.x_end is None or self.y_end is None:
                raise ValueError("drag requires x, y, x_end, y_end")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.text, self.key, self.url, self.wait_ms, self.scroll_amount)
            ):
                raise ValueError("drag cannot include selector, target, text, key, url, wait_ms, or scroll_amount")

        elif self.action_type is ActionType.SCROLL:
            if self.scroll_amount is None:
                raise ValueError("scroll requires scroll_amount")
            if self.scroll_amount == 0:
                raise ValueError("scroll_amount must not be zero")
            if self.x is None or self.y is None:
                raise ValueError("scroll requires x, y coordinates")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.text, self.key, self.url, self.wait_ms, self.x_end, self.y_end)
            ):
                raise ValueError("scroll cannot include selector, target, text, key, url, wait_ms, x_end, or y_end")

        elif self.action_type is ActionType.HOVER:
            if self.x is None or self.y is None:
                raise ValueError("hover requires x, y coordinates")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.text, self.key, self.url, self.wait_ms, *_new_fields)
            ):
                raise ValueError("hover cannot include selector, target, text, key, url, wait_ms, or desktop fields")

        elif self.action_type is ActionType.READ_CLIPBOARD:
            if any(
                value is not None
                for value in (
                    self.selector, self.target_element_id, self.text, self.key,
                    self.url, self.wait_ms, self.x, self.y, self.target_context,
                    *_new_fields,
                )
            ):
                raise ValueError("read_clipboard cannot include any action payload fields")

        elif self.action_type is ActionType.WRITE_CLIPBOARD:
            if self.text is None:
                raise ValueError("write_clipboard requires text")
            if any(
                value is not None
                for value in (
                    self.selector, self.target_element_id, self.key, self.url,
                    self.wait_ms, self.x, self.y, self.target_context, *_new_fields,
                )
            ):
                raise ValueError("write_clipboard cannot include fields other than text")

        elif self.action_type is ActionType.SCREENSHOT_REGION:
            if self.x is None or self.y is None or self.x_end is None or self.y_end is None:
                raise ValueError("screenshot_region requires x, y, x_end, y_end")
            if any(
                value is not None
                for value in (self.selector, self.target_element_id, self.text, self.key, self.url, self.wait_ms, self.scroll_amount)
            ):
                raise ValueError("screenshot_region cannot include selector, target, text, key, url, wait_ms, or scroll_amount")

        return self


class PolicyDecision(StrictModel):
    """The next action decision chosen from current perception plus agent state."""

    action: AgentAction
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    active_subgoal: str = Field(min_length=1)
