"""Schemas for the perception stage of the loop."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from src.models.common import StrictModel


class UIElementType(StrEnum):
    """Visible browser element types relevant to the Gmail draft workflow."""

    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    DIALOG = "dialog"
    ICON = "icon"
    UNKNOWN = "unknown"


class PageHint(StrEnum):
    """Supported high-level page classifications for the Gmail benchmark."""

    GOOGLE_SIGN_IN = "google_sign_in"
    GMAIL_INBOX = "gmail_inbox"
    GMAIL_COMPOSE = "gmail_compose"
    GMAIL_MESSAGE_VIEW = "gmail_message_view"
    UNKNOWN = "unknown"


class UIElement(StrictModel):
    """A typed visible element extracted from the current browser frame."""

    element_id: str = Field(min_length=1)
    element_type: UIElementType
    label: str = Field(min_length=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    is_interactable: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ScreenPerception(StrictModel):
    """Typed understanding of the current screen state."""

    summary: str = Field(min_length=1)
    page_hint: PageHint
    visible_elements: list[UIElement] = Field(default_factory=list)
    focused_element_id: str | None = Field(default=None, min_length=1)
    capture_artifact_path: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
