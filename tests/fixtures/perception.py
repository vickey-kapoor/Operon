"""Shared perception builders."""

from __future__ import annotations

from operon.models.perception import ScreenPerception, UIElement, UIElementType


def ui_element(
    element_id: str = "button-1",
    *,
    label: str = "Submit",
    element_type: UIElementType = UIElementType.BUTTON,
    x: int = 100,
    y: int = 100,
    width: int = 80,
    height: int = 30,
    is_interactable: bool = True,
) -> UIElement:
    return UIElement(
        element_id=element_id,
        element_type=element_type,
        label=label,
        x=x,
        y=y,
        width=width,
        height=height,
        is_interactable=is_interactable,
        confidence=1.0,
    )


def screen_perception(
    *,
    summary: str = "Test screen",
    page_hint: str = "unknown",
    capture_artifact_path: str = "runs/run-1/step_1/before.png",
    visible_elements: list[UIElement] | None = None,
) -> ScreenPerception:
    return ScreenPerception(
        summary=summary,
        page_hint=page_hint,
        capture_artifact_path=capture_artifact_path,
        visible_elements=visible_elements or [],
    )

