"""Perception service interface for typed screen understanding."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.capture import CaptureFrame
from src.models.logs import ModelDebugArtifacts
from src.models.perception import PageHint, ScreenPerception
from src.models.state import AgentState


class PerceptionError(RuntimeError):
    """Raised when Gemini perception output cannot be parsed into strict schemas."""


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

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        prompt = self._render_prompt(state)
        step_dir = Path(screenshot.artifact_path).resolve().parent
        debug_artifacts = self._artifact_paths(step_dir)
        debug_artifacts.prompt_artifact_path.write_text(prompt, encoding="utf-8")

        try:
            raw_output = await self.gemini_client.generate_perception(prompt, screenshot.artifact_path)
        except GeminiClientError:
            raise

        debug_artifacts.raw_response_artifact_path.write_text(raw_output, encoding="utf-8")
        perception = parse_perception_output(raw_output, screenshot.artifact_path)
        debug_artifacts.parsed_artifact_path.write_text(perception.model_dump_json(indent=2), encoding="utf-8")
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(debug_artifacts.prompt_artifact_path),
            raw_response_artifact_path=str(debug_artifacts.raw_response_artifact_path),
            parsed_artifact_path=str(debug_artifacts.parsed_artifact_path),
        )
        return perception

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def _render_prompt(self, state: AgentState) -> str:
        previous_summary = state.observation_history[-1].summary if state.observation_history else "none"
        return self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            previous_summary=previous_summary,
        )

    @staticmethod
    def _artifact_paths(step_dir: Path) -> "_StageArtifactPaths":
        step_dir.mkdir(parents=True, exist_ok=True)
        return _StageArtifactPaths(
            prompt_artifact_path=step_dir / "perception_prompt.txt",
            raw_response_artifact_path=step_dir / "perception_raw.txt",
            parsed_artifact_path=step_dir / "perception_parsed.json",
        )


class _StageArtifactPaths:
    """Filesystem paths used to persist one model-backed stage's artifacts."""

    def __init__(self, prompt_artifact_path: Path, raw_response_artifact_path: Path, parsed_artifact_path: Path) -> None:
        self.prompt_artifact_path = prompt_artifact_path
        self.raw_response_artifact_path = raw_response_artifact_path
        self.parsed_artifact_path = parsed_artifact_path



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
    try:
        return ScreenPerception.model_validate(parsed)
    except ValidationError as exc:
        raise PerceptionError("Gemini perception output did not match the strict schema.") from exc



def _fallback_page_hint_from_summary(summary: object) -> PageHint:
    if not isinstance(summary, str):
        return PageHint.UNKNOWN
    lowered = summary.lower()
    if "sign in" in lowered or "google account" in lowered or "email or phone" in lowered:
        return PageHint.GOOGLE_SIGN_IN
    if "compose" in lowered or "draft" in lowered:
        return PageHint.GMAIL_COMPOSE
    if "inbox" in lowered:
        return PageHint.GMAIL_INBOX
    if "message" in lowered or "conversation" in lowered:
        return PageHint.GMAIL_MESSAGE_VIEW
    return PageHint.UNKNOWN



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
