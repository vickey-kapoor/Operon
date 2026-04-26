"""Video-based action verification using temporal saliency + Gemini multimodal."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import cv2

from src.agent.perception import _strip_json_fence
from src.agent.screen_diff import TemporalSaliencyResult, compute_temporal_saliency
from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.common import StrictModel
from src.models.policy import AgentAction

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "video_verification_prompt.txt"

# Confidence thresholds for interpreting combined saliency + Gemini results.
_CONFIDENCE_SUCCESS = 0.5   # ≥ this → action succeeded
_CONFIDENCE_UNCERTAIN = 0.2  # ≥ this → still loading / uncertain

# Max frames sampled from the video for saliency (evenly spaced).
_MAX_SALIENCY_FRAMES = 24


class VideoVerificationResult(StrictModel):
    """Structured result from video verification."""

    did_it_work: bool
    what_happened: str
    suggested_next_action: str
    confidence_score: float = 0.5  # 0.0–1.0 derived from temporal saliency
    motion_class: str = "unknown"  # "hung" | "spinner" | "progressing" | "unknown"


class VideoVerifier:
    """Verify an action using pixel-velocity temporal saliency and Gemini video analysis."""

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
        """Compute temporal saliency from the video, then optionally ask Gemini.

        The confidence score reflects pixel velocity:
        - hung (≈0.0): no motion → action had no effect
        - spinner (≈0.35): periodic motion → still loading
        - progressing (0.55–0.9): aperiodic motion → real progress
        """
        # Extract sampled frames and compute saliency; release buffer immediately.
        frames = await asyncio.to_thread(self._extract_frames, video_path)
        saliency: TemporalSaliencyResult | None = None
        if len(frames) >= 2:
            saliency = compute_temporal_saliency(frames)
            logger.info(
                "temporal_saliency: class=%s velocity=%.5f variance=%.7f confidence=%.2f",
                saliency.motion_class, saliency.velocity_mean,
                saliency.velocity_variance, saliency.confidence,
            )
        else:
            logger.debug("temporal_saliency: skipped — fewer than 2 frames decoded")
        frames.clear()

        # Short-circuit only when we have clear evidence of a hung screen.
        if saliency is not None and saliency.motion_class == "hung":
            return VideoVerificationResult(
                did_it_work=False,
                what_happened="Screen showed no motion; app appears hung",
                suggested_next_action="retry_same_step",
                confidence_score=0.0,
                motion_class="hung",
            )

        prompt = self._render_prompt(action, intent)
        try:
            raw = await self.gemini_client.generate_video_verification(prompt, str(video_path))
        except GeminiClientError:
            logger.warning("Video verification Gemini call failed", exc_info=True)
            return VideoVerificationResult(
                did_it_work=False,
                what_happened="Video verification call failed",
                suggested_next_action="retry_same_step",
                confidence_score=saliency.confidence if saliency is not None else 0.0,
                motion_class=saliency.motion_class if saliency is not None else "unknown",
            )

        gemini_result = self._parse_response(raw)
        if saliency is not None:
            confidence = self._compute_confidence(saliency, gemini_result.did_it_work)
            motion_class = saliency.motion_class
        else:
            confidence = 0.9 if gemini_result.did_it_work else 0.1
            motion_class = "unknown"

        return VideoVerificationResult(
            did_it_work=gemini_result.did_it_work,
            what_happened=gemini_result.what_happened,
            suggested_next_action=gemini_result.suggested_next_action,
            confidence_score=confidence,
            motion_class=motion_class,
        )

    @staticmethod
    def _extract_frames(video_path: Path, max_frames: int = _MAX_SALIENCY_FRAMES) -> list:
        """Sample up to max_frames evenly spaced BGR frames from the MP4."""
        frames: list = []
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            step = max(1, total // max_frames)
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frames.append(frame)
                idx += 1
        finally:
            cap.release()
        return frames

    @staticmethod
    def _compute_confidence(saliency: TemporalSaliencyResult, gemini_ok: bool) -> float:
        """Combine saliency motion class with Gemini's semantic verdict."""
        if saliency.motion_class == "hung":
            return 0.0
        if saliency.motion_class == "spinner":
            # Loading state: don't trust Gemini's optimism; cap at spinner band.
            return 0.35
        # progressing: Gemini is the semantic tiebreaker.
        return 0.9 if gemini_ok else 0.2

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
