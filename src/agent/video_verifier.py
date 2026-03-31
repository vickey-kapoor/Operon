"""Video-based action verification using Gemini multimodal."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.perception import _strip_json_fence
from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.common import StrictModel
from src.models.policy import AgentAction

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "video_verification_prompt.txt"


class VideoVerificationResult(StrictModel):
    """Structured result from Gemini video verification."""

    did_it_work: bool
    what_happened: str
    suggested_next_action: str


class VideoVerifier:
    """Send a screen recording to Gemini and parse the verification result."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client
        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    async def verify_action(
        self,
        *,
        video_path: Path,
        action: AgentAction,
        intent: str,
    ) -> VideoVerificationResult:
        """Send the video to Gemini and return structured verification."""
        prompt = self._render_prompt(action, intent)
        try:
            raw = await self.gemini_client.generate_video_verification(prompt, str(video_path))
        except GeminiClientError:
            logger.warning("Video verification Gemini call failed", exc_info=True)
            return VideoVerificationResult(
                did_it_work=False,
                what_happened="Video verification call failed",
                suggested_next_action="retry_same_step",
            )
        return self._parse_response(raw)

    def _render_prompt(self, action: AgentAction, intent: str) -> str:
        detail_parts: list[str] = []
        if action.target_element_id:
            detail_parts.append(f"target={action.target_element_id}")
        if action.text:
            detail_parts.append(f"text={action.text!r}")
        if action.key:
            detail_parts.append(f"key={action.key}")
        if action.x is not None and action.y is not None:
            detail_parts.append(f"at ({action.x}, {action.y})")
        return self._prompt_template.format(
            action_type=action.action_type.value,
            action_detail=", ".join(detail_parts) or "no additional details",
            intent=intent,
        )

    @staticmethod
    def _parse_response(raw: str) -> VideoVerificationResult:
        cleaned = _strip_json_fence(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Video verification response was not valid JSON: %s", raw[:200])
            return VideoVerificationResult(
                did_it_work=False,
                what_happened="Unparseable verification response",
                suggested_next_action="retry_same_step",
            )
        return VideoVerificationResult(
            did_it_work=bool(parsed.get("did_it_work", False)),
            what_happened=str(parsed.get("what_happened", "unknown")),
            suggested_next_action=str(parsed.get("suggested_next_action", "continue")),
        )
