"""Perception service interface for typed screen understanding."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image
from pydantic import ValidationError

from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory, StopReason
from operon.models.perception import (
    PageHint,
    RawScreenPerception,
    RawUIElement,
    ScreenPerception,
    UIElement,
    UIElementNameSource,
    UIElementType,
)
from operon.models.state import AgentState

logger = logging.getLogger(__name__)

_MIN_INTERACTIVE_ELEMENTS = 1
_MIN_TARGETABLE_CANDIDATES = 1
_MIN_FORM_INTERACTIVE_ELEMENTS = 2
_MIN_FORM_TEXT_ELEMENTS = 1
_LABEL_VERTICAL_GAP_PX = 72
_LABEL_HORIZONTAL_GAP_PX = 180
_LABEL_ALIGNMENT_TOLERANCE_PX = 96


class PerceptionError(RuntimeError):
    """Raised when Gemini perception output cannot be parsed into strict schemas."""


class PerceptionLowQualityError(PerceptionError):
    """Raised when perception output is structurally valid but semantically unusable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.failure_category = FailureCategory.PERCEPTION_LOW_QUALITY
        self.stop_reason = StopReason.PERCEPTION_LOW_QUALITY


class PerceptionService(ABC):
    @abstractmethod
    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        """Convert a captured frame into a typed screen perception."""


_ELEMENT_FIELDS = frozenset(RawUIElement.model_fields.keys())


def _normalize_visible_elements(parsed: dict[str, Any]) -> None:
    """Fix common Gemini output quirks in visible_elements before validation."""
    elements = parsed.get("visible_elements")
    if not isinstance(elements, list):
        return
    for element in elements:
        if not isinstance(element, dict):
            continue
        # Fix: Gemini sometimes emits "element_N": "element_N" instead of "element_id"
        if "element_id" not in element:
            for key in list(element.keys()):
                if key.startswith("element_") and key not in _ELEMENT_FIELDS:
                    element["element_id"] = element.pop(key)
                    break
        # Fix: Gemini occasionally emits a numeric key like "10": 40 instead of "y": 10
        # (the y-value leaks into the key name). Recover y from it when y is missing.
        if "y" not in element:
            for key in list(element.keys()):
                if key not in _ELEMENT_FIELDS and key.lstrip("-").isdigit():
                    element["y"] = int(key)
                    del element[key]
                    break
        # Drop any extra keys not in the schema to tolerate minor hallucinations
        extra_keys = [k for k in element if k not in _ELEMENT_FIELDS]
        for key in extra_keys:
            del element[key]


def _fix_spaced_json(text: str) -> str:
    """Collapse space-separated single characters in malformed Gemini output.

    Gemini occasionally returns JSON with spaces between every character, e.g.
    '"s u m m a r y": "s e a r c h _ p a g e"'. This iteratively collapses
    adjacent single-char pairs until the string is stable.
    """
    if '"s u m m a r y"' not in text:
        return text
    logger.warning("Detected spaced-character Gemini output; collapsing before parse.")
    pattern = re.compile(r'([a-zA-Z0-9_]) ([a-zA-Z0-9_])')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r'\1\2', text)
    return text


def _check_coord_bounds(perception: "ScreenPerception", screenshot_path: str) -> None:
    """Assert that every returned element coordinate lies within the native screenshot
    dimensions, then log native dims alongside the coord ranges found.

    This is the runtime guard for the coordinate-space contract documented in
    the JSON backend prompts (prompts/*_combined_prompt.txt). A violation means a model upgrade or prompt
    edit broke the raw-pixel assumption — coords will be silently wrong at
    execution time without this check.

    Logs a WARNING on violation (does not raise) so a single bad element does
    not abort the whole step; the log makes the divergence immediately debuggable.
    """
    try:
        with Image.open(screenshot_path) as img:
            native_w, native_h = img.size
    except Exception as exc:
        logger.debug("coord_bounds_check: could not read %s: %s", screenshot_path, exc)
        return

    elements = perception.visible_elements
    if not elements:
        logger.debug("coord_bounds native=%dx%d  elements=0", native_w, native_h)
        return

    xs = [e.x for e in elements]
    ys = [e.y for e in elements]
    x_max = max(e.x + e.width for e in elements)
    y_max = max(e.y + e.height for e in elements)

    ox, oy = perception.monitor_origin
    # Log native dims, monitor origin, and coord ranges on every parse.
    # If DPI scaling or a model change ever shifts the coordinate space,
    # this line makes the divergence immediately visible in the log stream.
    logger.debug(
        "coord_bounds native=%dx%d  origin=(%d,%d)  x=[%d..%d]  y=[%d..%d]  elements=%d",
        native_w, native_h, ox, oy, min(xs), x_max, min(ys), y_max, len(elements),
    )

    violations = [
        e for e in elements
        if e.x < 0 or e.y < 0 or e.x + e.width > native_w or e.y + e.height > native_h
    ]
    if violations:
        ids = [e.element_id for e in violations]
        logger.warning(
            "coord_bounds VIOLATION native=%dx%d origin=(%d,%d) — %d element(s) exceed "
            "screen bounds: %s. Check the combined prompt's coordinate-space contract.",
            native_w, native_h, ox, oy, len(violations), ids,
        )


def parse_perception_output(raw_output: str, screenshot_path: str, scale_ratio: float = 1.0) -> ScreenPerception:
    cleaned = _fix_spaced_json(_strip_json_fence(raw_output))
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise PerceptionError("Gemini perception output was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise PerceptionError("Gemini perception output must be a JSON object.")

    page_hint = parsed.get("page_hint")
    if page_hint is None and "page_hint" not in parsed:
        parsed["page_hint"] = _fallback_page_hint_from_summary(parsed.get("summary"))
    parsed["capture_artifact_path"] = screenshot_path
    _normalize_visible_elements(parsed)
    try:
        raw_perception = RawScreenPerception.model_validate(parsed)
    except ValidationError as exc:
        raise PerceptionError("Gemini perception output did not match the strict schema.") from exc
    perception = _apply_weak_canonicalization(_canonicalize_perception(raw_perception))
    if scale_ratio != 1.0:
        perception = _upscale_element_coords(perception, scale_ratio)
    _check_coord_bounds(perception, screenshot_path)
    return perception


def _upscale_element_coords(perception: ScreenPerception, scale_ratio: float) -> ScreenPerception:
    """Invert the downscaling applied before sending the screenshot to Gemini.

    Gemini returns coordinates in the downscaled image's pixel space. Multiplying
    by 1/scale_ratio restores them to native screenshot pixel space so pyautogui
    and Playwright click at the correct position on the actual screen.
    """
    inverse = 1.0 / scale_ratio
    upscaled = [
        element.model_copy(update={
            "x": round(element.x * inverse),
            "y": round(element.y * inverse),
            "width": max(1, round(element.width * inverse)),
            "height": max(1, round(element.height * inverse)),
        })
        for element in perception.visible_elements
    ]
    return perception.model_copy(update={"visible_elements": upscaled})



def _fallback_page_hint_from_summary(summary: object) -> PageHint:
    # Only classify generic terminal states — everything else is left to the LLM.
    if not isinstance(summary, str):
        return PageHint.UNKNOWN
    lowered = summary.lower()
    if "thank you" in lowered or "form submitted" in lowered or "successfully submitted" in lowered:
        return PageHint.FORM_SUCCESS
    return PageHint.UNKNOWN



def _canonicalize_perception(raw_perception: RawScreenPerception) -> ScreenPerception:
    return ScreenPerception(
        summary=raw_perception.summary,
        page_hint=raw_perception.page_hint,
        visible_elements=[_canonicalize_visible_element(element) for element in raw_perception.visible_elements],
        focused_element_id=raw_perception.focused_element_id,
        capture_artifact_path=raw_perception.capture_artifact_path,
        confidence=raw_perception.confidence,
        monitor_origin=raw_perception.monitor_origin,
    )


def _canonicalize_visible_element(element) -> UIElement:
    return UIElement.model_validate(element.model_dump())


def _apply_weak_canonicalization(perception: ScreenPerception) -> ScreenPerception:
    text_candidates = _label_text_candidates(perception.visible_elements)
    canonicalized: list[UIElement] = []
    for element in perception.visible_elements:
        if not element.is_unlabeled or not element.is_interactable or element.element_type not in {
            UIElementType.INPUT,
            UIElementType.BUTTON,
            UIElementType.LINK,
        }:
            canonicalized.append(element)
            continue
        inferred_label = _infer_nearby_label(element, text_candidates)
        if inferred_label is None:
            canonicalized.append(element)
            continue
        canonicalized.append(
            element.model_copy(
                update={
                    "label": inferred_label,
                    "primary_name": inferred_label,
                    "name_source": UIElementNameSource.LABEL,
                    "is_unlabeled": False,
                    "usable_for_targeting": True,
                }
            )
        )
    return perception.model_copy(update={"visible_elements": canonicalized})


def _low_quality_reason(perception: ScreenPerception, *, quality_metrics: dict[str, float | int | bool] | None = None) -> str | None:
    elements = perception.visible_elements
    metrics = quality_metrics or _quality_metrics(perception)
    if not elements:
        return "no visible elements were returned"

    if (
        float(metrics["unlabeled_ratio"]) > 0.5
        and int(metrics["candidate_count"]) == 0
        and int(metrics["interactive_count"]) < _MIN_INTERACTIVE_ELEMENTS
    ):
        return "more than 50% of visible elements are unlabeled"

    usable_elements = [element for element in elements if element.usable_for_targeting]
    if not usable_elements and int(metrics["interactive_count"]) < _MIN_INTERACTIVE_ELEMENTS:
        return "zero usable_for_targeting elements"

    selector_candidates = [
        element
        for element in usable_elements
        if element.element_type in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}
    ]
    if len(selector_candidates) < _MIN_TARGETABLE_CANDIDATES and int(metrics["interactive_count"]) < _MIN_INTERACTIVE_ELEMENTS:
        return "no candidate passes selector threshold"

    if perception.page_hint is PageHint.FORM_PAGE:
        if _form_page_minimally_viable(metrics):
            return None

    if int(metrics["interactive_count"]) < _MIN_INTERACTIVE_ELEMENTS:
        return "no interactive elements were returned"
    if int(metrics["candidate_count"]) < _MIN_TARGETABLE_CANDIDATES:
        return "zero usable candidates after salvage"

    return None


def _quality_metrics(perception: ScreenPerception, *, salvage_mode: bool = False) -> dict[str, float | int | bool]:
    elements = perception.visible_elements
    total = len(elements)
    unlabeled_count = sum(1 for element in elements if element.is_unlabeled)
    usable_elements = [element for element in elements if element.usable_for_targeting]
    selector_candidates = [
        element
        for element in usable_elements
        if element.element_type in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}
    ]
    interactive_count = sum(
        1
        for element in elements
        if element.is_interactable and element.element_type in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}
    )
    text_count = sum(
        1
        for element in elements
        if element.element_type is UIElementType.TEXT and bool(element.primary_name.strip())
    )
    labeled_interactive_count = sum(
        1
        for element in elements
        if element.is_interactable
        and element.element_type in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}
        and not element.is_unlabeled
    )
    unlabeled_interactive_count = max(interactive_count - labeled_interactive_count, 0)
    spatially_groundable_interactive_count = sum(
        1
        for element in elements
        if element.is_interactable
        and element.element_type in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}
        and _infer_nearby_label(element, _label_text_candidates(elements)) is not None
    )
    return {
        "total_elements": total,
        "labeled_elements": total - unlabeled_count,
        "unlabeled_elements": unlabeled_count,
        "unlabeled_ratio": (unlabeled_count / total) if total else 0.0,
        "usable_count": len(usable_elements),
        "candidate_count": len(selector_candidates),
        "interactive_count": interactive_count,
        "text_count": text_count,
        "labeled_interactive_count": labeled_interactive_count,
        "unlabeled_interactive_count": unlabeled_interactive_count,
        "spatially_groundable_interactive_count": spatially_groundable_interactive_count,
        "salvage_mode": salvage_mode,
    }


def _form_page_minimally_viable(metrics: dict[str, float | int | bool]) -> bool:
    return (
        int(metrics["interactive_count"]) >= _MIN_FORM_INTERACTIVE_ELEMENTS
        and int(metrics["text_count"]) >= _MIN_FORM_TEXT_ELEMENTS
        and (
            int(metrics["candidate_count"]) >= _MIN_TARGETABLE_CANDIDATES
            or int(metrics["spatially_groundable_interactive_count"]) >= 1
        )
    )


def _label_text_candidates(elements: list[UIElement]) -> list[UIElement]:
    return [
        element
        for element in elements
        if element.element_type is UIElementType.TEXT
        and not element.is_interactable
        and bool(element.primary_name.strip())
    ]


def _infer_nearby_label(target: UIElement, text_candidates: list[UIElement]) -> str | None:
    ranked: list[tuple[float, UIElement]] = []
    for candidate in text_candidates:
        above_gap = target.y - (candidate.y + candidate.height)
        left_gap = target.x - (candidate.x + candidate.width)
        horizontal_offset = abs((candidate.x + candidate.width // 2) - (target.x + target.width // 2))
        vertical_offset = abs((candidate.y + candidate.height // 2) - (target.y + target.height // 2))

        aligned_above = 0 <= above_gap <= _LABEL_VERTICAL_GAP_PX and horizontal_offset <= _LABEL_ALIGNMENT_TOLERANCE_PX
        aligned_left = 0 <= left_gap <= _LABEL_HORIZONTAL_GAP_PX and vertical_offset <= _LABEL_ALIGNMENT_TOLERANCE_PX
        if not aligned_above and not aligned_left:
            continue

        distance_score = above_gap if aligned_above else left_gap
        ranked.append((distance_score + horizontal_offset + vertical_offset, candidate))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].element_id))
    return ranked[0][1].primary_name


def _strip_json_fence(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


