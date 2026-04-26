"""Schemas for the perception stage of the loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.models.common import StrictModel


class UIElementType(StrEnum):
    """Visible element types for browser and desktop automation."""

    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    DIALOG = "dialog"
    ICON = "icon"
    WINDOW = "window"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "UIElementType":
        if not isinstance(value, str):
            return None  # type: ignore[return-value]
        obj = str.__new__(cls, value)
        obj._name_ = value
        obj._value_ = value
        return obj


class PageHint(StrEnum):
    """Common page classifications. Accepts arbitrary snake_case values from the LLM."""

    FORM_PAGE = "form_page"
    FORM_SUCCESS = "form_success"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "PageHint":
        if not isinstance(value, str):
            return None  # type: ignore[return-value]
        obj = str.__new__(cls, value)
        obj._name_ = value
        obj._value_ = value
        return obj


class UIElementNameSource(StrEnum):
    """Ordered semantic name sources used during canonicalization."""

    LABEL = "label"
    TEXT = "text"
    PLACEHOLDER = "placeholder"
    NAME = "name"
    ROLE = "role"
    SYNTHETIC = "synthetic"


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return value  # let strict validation reject non-string values later
    normalized = value.strip()
    return normalized or None


def _canonical_name_fields(payload: dict[str, Any]) -> tuple[str, UIElementNameSource]:
    candidates = (
        ("label", UIElementNameSource.LABEL),
        ("text", UIElementNameSource.TEXT),
        ("placeholder", UIElementNameSource.PLACEHOLDER),
        ("name", UIElementNameSource.NAME),
        ("role", UIElementNameSource.ROLE),
    )
    for field_name, source in candidates:
        value = payload.get(field_name)
        if isinstance(value, str) and value:
            return value, source

    element_type = payload.get("element_type")
    if isinstance(element_type, str) and element_type:
        return f"unlabeled_{element_type}", UIElementNameSource.SYNTHETIC

    if isinstance(element_type, UIElementType):
        return f"unlabeled_{element_type.value}", UIElementNameSource.SYNTHETIC

    element_id = payload.get("element_id")
    if isinstance(element_id, str) and element_id:
        return f"unlabeled_{element_id}", UIElementNameSource.SYNTHETIC

    return "unlabeled_unknown", UIElementNameSource.SYNTHETIC


def _usable_for_targeting(payload: dict[str, Any], source: UIElementNameSource) -> bool:
    if payload.get("is_interactable") is not True:
        return False
    return source in {
        UIElementNameSource.LABEL,
        UIElementNameSource.TEXT,
        UIElementNameSource.PLACEHOLDER,
        UIElementNameSource.NAME,
    }


class RawUIElement(StrictModel):
    """Strict model-facing element schema with tolerant weak semantic fields."""

    element_id: str = Field(min_length=1)
    element_type: UIElementType
    label: str | None = None
    text: str | None = None
    placeholder: str | None = None
    name: str | None = None
    role: str | None = None
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    is_interactable: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("label", "text", "placeholder", "name", "role", mode="before")
    @classmethod
    def _normalize_weak_semantic_fields(cls, value: object) -> object:
        return _normalize_optional_text(value)


class UIElement(StrictModel):
    """Canonical internal element state used after perception normalization."""

    element_id: str = Field(min_length=1)
    element_type: UIElementType
    label: str | None = None
    text: str | None = None
    placeholder: str | None = None
    name: str | None = None
    role: str | None = None
    primary_name: str = Field(min_length=1)
    name_source: UIElementNameSource
    is_unlabeled: bool
    usable_for_targeting: bool
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    is_interactable: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _populate_canonical_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        for field_name in ("label", "text", "placeholder", "name", "role"):
            normalized[field_name] = _normalize_optional_text(normalized.get(field_name))

        primary_name, name_source = _canonical_name_fields(normalized)
        normalized["primary_name"] = primary_name
        normalized["name_source"] = name_source
        normalized["is_unlabeled"] = name_source is UIElementNameSource.SYNTHETIC
        normalized["usable_for_targeting"] = _usable_for_targeting(normalized, name_source)
        return normalized


class RawScreenPerception(StrictModel):
    """Strict model-facing screen perception with tolerant weak semantic fields."""

    summary: str = Field(min_length=1)
    page_hint: PageHint
    visible_elements: list[RawUIElement] = Field(default_factory=list)
    focused_element_id: str | None = Field(default=None, min_length=1)
    capture_artifact_path: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Virtual-desktop origin of the monitor this perception was captured from.
    # Coords in visible_elements are monitor-local; add this offset to get
    # virtual-desktop coords for pyautogui. Defaults to (0, 0) (primary display).
    monitor_origin: tuple[int, int] = Field(default=(0, 0))


class ScreenPerception(StrictModel):
    """Canonical typed understanding of the current screen state."""

    summary: str = Field(min_length=1)
    page_hint: PageHint
    visible_elements: list[UIElement] = Field(default_factory=list)
    focused_element_id: str | None = Field(default=None, min_length=1)
    capture_artifact_path: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    monitor_origin: tuple[int, int] = Field(default=(0, 0))
    # Set to True when Gemini returned zero elements (blank/loading frame).
    # The loop uses this as the trigger for liveness retries rather than a hard failure.
    is_empty_frame: bool = False
    # Stamped with the number of zero-element liveness retries that occurred before
    # a usable frame was captured. Appears in run.jsonl for observability.
    liveness_retries: int = 0
