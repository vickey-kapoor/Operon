"""Gemini multimodal client for UI screenshot analysis."""

import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import List, Optional, Union

from google import genai
from google.genai import types
from PIL import Image
from src import metrics, tracing

logger = logging.getLogger(__name__)


class VisionUnavailableError(Exception):
    """Raised when the Gemini API fails all retry attempts."""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are UI Navigator, an AI agent that controls web browsers and computer \
interfaces by analyzing screenshots.

Your task: Given a screenshot of the current UI state and a user's goal, \
determine the next actions to take.

IMPORTANT: Always analyze the screenshot carefully before deciding actions. \
Look for:
- Visible text, buttons, links, forms, menus
- Current URL or page title if visible
- Loading states or errors
- Interactive elements

Respond with a JSON object:
{
  "observation": "Detailed description of what you see on screen",
  "reasoning": "Step-by-step thinking about what actions to take",
  "actions": [
    // One or more actions from the list below
  ],
  "done": false,
  "result": null
}

Available actions:
{"type": "click", "coordinate": [x, y], "description": "..."}
{"type": "type", "text": "...", "description": "..."}
{"type": "key", "key": "Enter|Tab|Escape|F5|...", "description": "..."}
{"type": "scroll", "coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": 3, "description": "..."}
{"type": "navigate", "url": "https://...", "description": "..."}
{"type": "wait", "duration_ms": 1000, "description": "..."}
{"type": "screenshot", "description": "Take screenshot to observe current state"}
{"type": "done", "description": "Task complete"}

Rules:
- Coordinates are [x, y] pixels from top-left of the 1280x800 viewport
- Only use coordinates you can see in the screenshot
- If unsure of exact position, use screenshot action to get a fresh view
- Navigate action opens a URL in the browser
- Set done=true only when the task is fully accomplished
- Respond ONLY with the JSON object — no markdown fences, no extra prose
- If the page is blank or white, immediately navigate to the most appropriate website for the task using your knowledge of the web — do not wait or take a screenshot first
- Avoid google.com for searches — it blocks automated browsers with CAPTCHAs. Use your best judgement to pick an alternative (e.g. bing.com, duckduckgo.com, or navigate directly to the best site for the task)
- If you hit a CAPTCHA or bot-detection wall, navigate directly to the target website instead of searching
- Be decisive: pick the best site for the task from your training knowledge and go there directly rather than using a search engine when possible
- For flight/hotel booking forms: fill fields one at a time, take a screenshot after each interaction to verify the state before continuing
- For date pickers: click the field first, wait for the calendar to open, then click the correct date cell
- If a site blocks you or shows a CAPTCHA, immediately navigate to a different site that offers the same service
"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GeminiVisionClient:
    """
    Wraps the Google Generative AI SDK to provide multimodal screenshot analysis.

    The client accepts PIL Images (or base64 strings), sends them to
    ``gemini-2.0-flash``, and returns the model's structured JSON response
    as a plain string (further parsing is handled by ActionPlanner).
    """

    MODEL_NAME = "gemini-2.5-flash"
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # seconds; doubled on each retry

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Gemini API key is required. Set the GOOGLE_API_KEY environment "
                "variable or pass api_key= explicitly."
            )
        self._client = genai.Client(api_key=resolved_key)
        self.model_name = model or self.MODEL_NAME

        self._generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,          # Low temperature for deterministic action plans
            top_p=0.95,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        logger.info("GeminiVisionClient initialised with model '%s'", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_screen(
        self,
        image: Union[Image.Image, str],
        task: str,
        history: Optional[List[types.Content]] = None,
    ) -> str:
        """
        Analyse a screenshot and return the model's raw JSON string response.

        Parameters
        ----------
        image:
            Either a PIL Image object or a base64-encoded PNG/JPEG string.
        task:
            The user's high-level navigation goal.
        history:
            Optional list of prior conversation turns as ``types.Content`` objects.

        Returns
        -------
        str
            The raw text produced by the model (should be a JSON object).
        """
        pil_image = self._ensure_pil_image(image)
        user_turn = self._build_user_turn(pil_image, task)

        # Run blocking SDK call in a thread so we don't block the event loop.
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._call_with_retry,
            user_turn,
            history,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_pil_image(image: Union[Image.Image, str]) -> Image.Image:
        """Convert base64 input to PIL Image if necessary."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            img_bytes = base64.b64decode(image)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        raise TypeError(
            f"image must be PIL.Image.Image or base64 str, got {type(image).__name__}"
        )

    @staticmethod
    def _pil_to_bytes(image: Image.Image) -> bytes:
        """Encode a PIL Image as PNG bytes."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _build_user_turn(self, image: Image.Image, task: str) -> types.Content:
        """Construct the user Content turn for the Gemini API."""
        img_bytes = self._pil_to_bytes(image)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        text_part = types.Part.from_text(
            text=(
                f"User task: {task}\n\n"
                "Analyze the screenshot above and respond with the next action plan "
                "as a JSON object matching the schema in your instructions."
            )
        )
        return types.Content(role="user", parts=[image_part, text_part])

    def _call_with_retry(
        self,
        user_turn: types.Content,
        history: Optional[List[types.Content]],
    ) -> str:
        """
        Call the Gemini API with exponential back-off retry on transient errors.
        """
        attempt = 0
        backoff = self.RETRY_BACKOFF

        while True:
            attempt += 1
            try:
                if history:
                    contents: list = history + [user_turn]
                else:
                    contents = [user_turn]

                t0 = time.time()
                with tracing.span("gemini_call", {"model": self.model_name, "attempt": str(attempt)}):
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=self._generation_config,
                    )
                gemini_ms = int((time.time() - t0) * 1000)

                text = response.text
                if not text:
                    raise ValueError("Gemini returned an empty response")

                metrics.emit("gemini_latency_ms", gemini_ms, {"model": self.model_name})
                logger.debug(
                    "Gemini response received",
                    extra={
                        "attempt": attempt,
                        "chars": len(text),
                        "gemini_latency_ms": gemini_ms,
                        "model": self.model_name,
                    },
                )
                return text

            except Exception as exc:
                if attempt > self.MAX_RETRIES:
                    logger.error(
                        "Gemini API call failed after %d attempts: %s",
                        attempt,
                        exc,
                        extra={"model": self.model_name},
                    )
                    raise VisionUnavailableError(
                        f"Gemini API unavailable after {self.MAX_RETRIES} retries: {exc}"
                    ) from exc

                logger.warning(
                    "Gemini API attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
