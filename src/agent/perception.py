"""Perception service interface for typed screen understanding."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, StopReason
from src.models.logs import ModelDebugArtifacts
from src.models.perception import (
    PageHint,
    RawScreenPerception,
    RawUIElement,
    ScreenPerception,
    UIElement,
    UIElementNameSource,
    UIElementType,
)
from src.models.state import AgentState
from src.store.background_writer import bg_writer

logger = logging.getLogger(__name__)

_MIN_INTERACTIVE_ELEMENTS = 1
_MIN_TARGETABLE_CANDIDATES = 1
_MIN_FORM_INTERACTIVE_ELEMENTS = 2
_MIN_FORM_TEXT_ELEMENTS = 1
_LABEL_VERTICAL_GAP_PX = 72
_LABEL_HORIZONTAL_GAP_PX = 180
_LABEL_ALIGNMENT_TOLERANCE_PX = 96
_SALVAGE_CONFIDENCE_CAP = 0.49


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


class GeminiPerceptionService(PerceptionService):
    """Gemini-backed perception implementation for strict screen understanding."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_path: Path | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.prompt_path = prompt_path or Path(__file__).resolve().parents[2] / "prompts" / "perception_prompt.txt"
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._max_semantic_retries = 1

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        step_dir = Path(screenshot.artifact_path).resolve().parent
        debug_artifacts = self._artifact_paths(step_dir)
        retry_log_lines: list[str] = []

        for attempt in range(self._max_semantic_retries + 1):
            prompt = self._render_prompt(state, semantic_retry=attempt > 0)
            bg_writer.enqueue(debug_artifacts.prompt_artifact_path, prompt)

            try:
                raw_output = await self.gemini_client.generate_perception(prompt, screenshot.artifact_path)
            except GeminiClientError:
                raise

            bg_writer.enqueue(debug_artifacts.raw_response_artifact_path, raw_output)
            self._last_debug_artifacts = ModelDebugArtifacts(
                prompt_artifact_path=str(debug_artifacts.prompt_artifact_path),
                raw_response_artifact_path=str(debug_artifacts.raw_response_artifact_path),
                parsed_artifact_path=str(debug_artifacts.parsed_artifact_path),
                retry_log_artifact_path=str(debug_artifacts.retry_log_artifact_path),
                diagnostics_artifact_path=str(debug_artifacts.diagnostics_artifact_path),
            )
            perception = parse_perception_output(raw_output, screenshot.artifact_path)
            quality_metrics = _quality_metrics(perception)
            low_quality_reason = _low_quality_reason(perception, quality_metrics=quality_metrics)
            if low_quality_reason is None:
                bg_writer.enqueue(debug_artifacts.parsed_artifact_path, perception.model_dump_json())
                _write_diagnostics_artifact(
                    debug_artifacts=debug_artifacts,
                    perception=perception,
                    quality_metrics=quality_metrics,
                    quality_gate_reason=None,
                    salvage_attempted=False,
                    salvage_reason=None,
                    salvage_metrics=None,
                    final_decision="accepted",
                )
                if retry_log_lines:
                    bg_writer.enqueue(debug_artifacts.retry_log_artifact_path, "\n".join(retry_log_lines))
                return perception

            retry_log_lines.append(_format_quality_log_line(attempt + 1, low_quality_reason, quality_metrics, salvage_mode=False))
            bg_writer.enqueue(debug_artifacts.retry_log_artifact_path, "\n".join(retry_log_lines))
            logger.warning("Retrying perception after low-quality output (%s).", low_quality_reason)
            if attempt >= self._max_semantic_retries:
                salvaged = _salvage_perception(perception)
                salvaged_metrics = _quality_metrics(salvaged, salvage_mode=True)
                salvage_reason = _low_quality_reason(salvaged, quality_metrics=salvaged_metrics)
                retry_log_lines.append(
                    _format_quality_log_line(
                        attempt + 1,
                        salvage_reason or "salvage_succeeded",
                        salvaged_metrics,
                        salvage_mode=True,
                    )
                )
                bg_writer.enqueue(debug_artifacts.retry_log_artifact_path, "\n".join(retry_log_lines))
                bg_writer.enqueue(debug_artifacts.parsed_artifact_path, salvaged.model_dump_json())
                _write_diagnostics_artifact(
                    debug_artifacts=debug_artifacts,
                    perception=perception,
                    quality_metrics=quality_metrics,
                    quality_gate_reason=low_quality_reason,
                    salvage_attempted=True,
                    salvage_reason=salvage_reason,
                    salvage_metrics=salvaged_metrics,
                    final_decision="salvaged" if salvage_reason is None else "aborted_low_quality",
                    salvaged_perception=salvaged,
                )
                if salvage_reason is None:
                    return salvaged
                raise PerceptionLowQualityError(f"Gemini perception output was low quality: {salvage_reason}")

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def _render_prompt(self, state: AgentState, *, semantic_retry: bool = False) -> str:
        previous_summary = state.observation_history[-1].summary if state.observation_history else "none"
        prompt = self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            previous_summary=previous_summary,
        )
        if semantic_retry:
            prompt = (
                f"{prompt}\n\nRetry instructions:\n"
                "Do not emit empty strings. Use null for missing fields.\n"
                "Ensure all interactive elements have labels or placeholders when possible.\n"
            )
        return prompt

    @staticmethod
    def _artifact_paths(step_dir: Path) -> "_StageArtifactPaths":
        step_dir.mkdir(parents=True, exist_ok=True)
        return _StageArtifactPaths(
            prompt_artifact_path=step_dir / "perception_prompt.txt",
            raw_response_artifact_path=step_dir / "perception_raw.txt",
            parsed_artifact_path=step_dir / "perception_parsed.json",
            retry_log_artifact_path=step_dir / "perception_retry_log.txt",
            diagnostics_artifact_path=step_dir / "perception_diagnostics.json",
        )


class _StageArtifactPaths:
    """Filesystem paths used to persist one model-backed stage's artifacts."""

    def __init__(
        self,
        prompt_artifact_path: Path,
        raw_response_artifact_path: Path,
        parsed_artifact_path: Path,
        retry_log_artifact_path: Path,
        diagnostics_artifact_path: Path,
    ) -> None:
        self.prompt_artifact_path = prompt_artifact_path
        self.raw_response_artifact_path = raw_response_artifact_path
        self.parsed_artifact_path = parsed_artifact_path
        self.retry_log_artifact_path = retry_log_artifact_path
        self.diagnostics_artifact_path = diagnostics_artifact_path



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
        # Drop any extra keys not in the schema to tolerate minor hallucinations
        extra_keys = [k for k in element if k not in _ELEMENT_FIELDS]
        for key in extra_keys:
            del element[key]


def parse_perception_output(raw_output: str, screenshot_path: str) -> ScreenPerception:
    cleaned = _strip_json_fence(raw_output)
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
    return _apply_weak_canonicalization(_canonicalize_perception(raw_perception))



def _fallback_page_hint_from_summary(summary: object) -> PageHint:
    if not isinstance(summary, str):
        return PageHint.UNKNOWN
    lowered = summary.lower()
    if "thank you" in lowered or "form submitted" in lowered or "successfully submitted" in lowered:
        return PageHint.FORM_SUCCESS
    if all(token in lowered for token in ("name", "email")) and ("message" in lowered or "submit" in lowered):
        return PageHint.FORM_PAGE
    if "sign in" in lowered or "google account" in lowered or "email or phone" in lowered:
        return PageHint.GOOGLE_SIGN_IN
    if "compose" in lowered or "draft" in lowered:
        return PageHint.GMAIL_COMPOSE
    if "inbox" in lowered:
        return PageHint.GMAIL_INBOX
    if "message" in lowered or "conversation" in lowered:
        return PageHint.GMAIL_MESSAGE_VIEW
    if "wikipedia" in lowered or "article" in lowered or "wiki" in lowered:
        return PageHint("article_page")
    if "search" in lowered:
        return PageHint("search_results")
    return PageHint.UNKNOWN



def _canonicalize_perception(raw_perception: RawScreenPerception) -> ScreenPerception:
    return ScreenPerception(
        summary=raw_perception.summary,
        page_hint=raw_perception.page_hint,
        visible_elements=[_canonicalize_visible_element(element) for element in raw_perception.visible_elements],
        focused_element_id=raw_perception.focused_element_id,
        capture_artifact_path=raw_perception.capture_artifact_path,
        confidence=raw_perception.confidence,
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


def _format_quality_log_line(
    attempt: int,
    reason: str,
    quality_metrics: dict[str, float | int | bool],
    *,
    salvage_mode: bool,
) -> str:
    return (
        f"attempt={attempt} reason={reason} "
        f"unlabeled_pct={float(quality_metrics['unlabeled_ratio']) * 100:.1f} "
        f"interactive_count={int(quality_metrics['interactive_count'])} "
        f"text_count={int(quality_metrics['text_count'])} "
        f"usable_count={int(quality_metrics['usable_count'])} "
        f"candidate_count={int(quality_metrics['candidate_count'])} "
        f"salvage_mode={str(salvage_mode).lower()}"
    )


def _salvage_perception(perception: ScreenPerception) -> ScreenPerception:
    text_candidates = _label_text_candidates(perception.visible_elements)
    salvaged_elements: list[UIElement] = []
    for element in perception.visible_elements:
        if not element.is_interactable or element.element_type not in {UIElementType.INPUT, UIElementType.BUTTON, UIElementType.LINK}:
            salvaged_elements.append(element)
            continue

        updates: dict[str, object] = {}
        inferred_label = None if not element.is_unlabeled else _infer_nearby_label(element, text_candidates)
        if inferred_label is not None:
            updates["label"] = inferred_label
            updates["primary_name"] = inferred_label
            updates["name_source"] = UIElementNameSource.LABEL
            updates["is_unlabeled"] = False
            updates["usable_for_targeting"] = True
            updates["confidence"] = min(element.confidence, _SALVAGE_CONFIDENCE_CAP)
        elif not element.usable_for_targeting:
            updates["usable_for_targeting"] = True
            updates["confidence"] = min(element.confidence, _SALVAGE_CONFIDENCE_CAP)
        salvaged_elements.append(element.model_copy(update=updates) if updates else element)

    salvaged = perception.model_copy(update={"visible_elements": salvaged_elements})
    salvaged_metrics = _quality_metrics(salvaged, salvage_mode=True)
    if _form_page_minimally_viable(salvaged_metrics) or salvaged_metrics["candidate_count"] >= _MIN_TARGETABLE_CANDIDATES:
        return salvaged
    return perception


def _form_page_minimally_viable(metrics: dict[str, float | int | bool]) -> bool:
    return (
        int(metrics["interactive_count"]) >= _MIN_FORM_INTERACTIVE_ELEMENTS
        and int(metrics["text_count"]) >= _MIN_FORM_TEXT_ELEMENTS
        and (
            int(metrics["candidate_count"]) >= _MIN_TARGETABLE_CANDIDATES
            or int(metrics["spatially_groundable_interactive_count"]) >= 1
        )
    )


def _write_diagnostics_artifact(
    *,
    debug_artifacts: "_StageArtifactPaths",
    perception: ScreenPerception,
    quality_metrics: dict[str, float | int | bool],
    quality_gate_reason: str | None,
    salvage_attempted: bool,
    salvage_reason: str | None,
    salvage_metrics: dict[str, float | int | bool] | None,
    final_decision: str,
    salvaged_perception: ScreenPerception | None = None,
) -> None:
    payload: dict[str, Any] = {
        "summary": perception.summary,
        "page_hint": perception.page_hint.value,
        "quality_gate_reason": quality_gate_reason,
        "salvage_attempted": salvage_attempted,
        "salvage_reason": salvage_reason,
        "final_decision": final_decision,
        "raw_response_artifact_path": str(debug_artifacts.raw_response_artifact_path),
        "parsed_artifact_path": str(debug_artifacts.parsed_artifact_path),
        "retry_log_artifact_path": str(debug_artifacts.retry_log_artifact_path),
        "quality_metrics": quality_metrics,
        "normalized_raw_perception_summary": {
            "summary": perception.summary,
            "page_hint": perception.page_hint.value,
            "focused_element_id": perception.focused_element_id,
            "element_count": len(perception.visible_elements),
        },
        "salvage_result": None,
    }
    if salvaged_perception is not None and salvage_metrics is not None:
        payload["salvage_result"] = {
            "summary": salvaged_perception.summary,
            "page_hint": salvaged_perception.page_hint.value,
            "focused_element_id": salvaged_perception.focused_element_id,
            "element_count": len(salvaged_perception.visible_elements),
            "quality_metrics": salvage_metrics,
        }
    bg_writer.enqueue(debug_artifacts.diagnostics_artifact_path, json.dumps(payload))


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
