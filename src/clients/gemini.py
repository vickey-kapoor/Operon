"""Gemini client interface and thin HTTP implementation."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from src.models.usage import ModelUsage, estimate_usage_cost

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised when Gemini requests or responses are invalid."""


class GeminiClient(ABC):
    """Typed interface for the only external model dependency."""

    @abstractmethod
    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        """Generate JSON-only perception output for the provided screenshot."""

    @abstractmethod
    async def generate_policy(self, prompt: str) -> str:
        """Generate JSON-only policy output for the provided state and perception prompt."""

    @abstractmethod
    async def generate_video_verification(self, prompt: str, video_path: str) -> str:
        """Send a video clip with a verification prompt and return raw JSON."""

    def latest_usage(self) -> ModelUsage | None:
        """Return provider-reported usage for the most recent call, when available."""
        return None

    def latest_perception_scale_ratio(self) -> float:
        """Return the scale ratio applied to the most recent perception screenshot. 1.0 = no resize."""
        return 1.0


class GeminiHttpClient(GeminiClient):
    """Async HTTP client for Gemini with connection pooling via httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_base_url = api_base_url or os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._client: httpx.AsyncClient | None = None
        self._last_usage: ModelUsage | None = None
        self._last_perception_scale_ratio: float = 1.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a persistent async client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                http2=True,
            )
        return self._client

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        """Send the screenshot and prompt to Gemini and return raw JSON text."""
        image_bytes, scale_ratio = await asyncio.to_thread(_optimize_screenshot, Path(screenshot_path))
        self._last_perception_scale_ratio = scale_ratio
        payload = self._build_perception_payload(prompt=prompt, image_bytes=image_bytes, mime_type="image/jpeg")
        return await self._post_json_payload(payload, request_kind="image")

    def latest_perception_scale_ratio(self) -> float:
        return self._last_perception_scale_ratio

    async def generate_policy(self, prompt: str) -> str:
        """Send the policy prompt to Gemini and return raw JSON text."""
        payload = self._build_text_payload(prompt=prompt)
        return await self._post_json_payload(payload, request_kind="text")

    async def generate_video_verification(self, prompt: str, video_path: str) -> str:
        """Send a video clip with verification prompt and return raw JSON."""
        video_bytes = await asyncio.to_thread(Path(video_path).read_bytes)
        payload = self._build_perception_payload(
            prompt=prompt, image_bytes=video_bytes, mime_type="video/mp4",
        )
        return await self._post_json_payload(payload, request_kind="video")

    def latest_usage(self) -> ModelUsage | None:
        return self._last_usage

    async def _post_json_payload(self, payload: dict[str, Any], *, request_kind: str) -> str:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiClientError("GEMINI_API_KEY is not configured.")

        url = f"{self.api_base_url}/{self.model}:generateContent?key={api_key}"
        client = await self._get_client()
        attempts = self.max_retries + 1
        last_error: GeminiClientError | None = None
        # Pre-serialize once — avoids re-encoding base64 payload on retries
        payload_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(
                    url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    detail = response.text
                    retryable = self._is_retryable_http_error(response.status_code)
                    last_error = GeminiClientError(f"Gemini HTTP error {response.status_code}: {detail}")
                    if not retryable or attempt >= attempts:
                        raise last_error
                    self._log_retry(attempt, attempts, f"http {response.status_code}")
                else:
                    response_body = response.text
                    break
            except httpx.TimeoutException as exc:
                last_error = GeminiClientError(f"Gemini request timed out after {self.timeout_seconds} seconds.")
                if attempt >= attempts:
                    raise last_error from exc
                self._log_retry(attempt, attempts, f"timeout after {self.timeout_seconds} seconds")
            except httpx.ConnectError as exc:
                last_error = GeminiClientError(f"Gemini connection failed: {exc}")
                if attempt >= attempts:
                    raise last_error from exc
                self._log_retry(attempt, attempts, f"connection error: {exc}")
            except GeminiClientError:
                raise

            await asyncio.sleep(self.retry_backoff_seconds * attempt)
        else:
            if last_error is not None:
                raise last_error
            raise GeminiClientError("Gemini request failed before receiving a response.")

        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise GeminiClientError("Gemini response was not valid JSON.") from exc
        self._last_usage = _extract_usage(payload=response_payload, model=self.model, request_kind=request_kind)
        return self.extract_text(response_payload)

    @staticmethod
    def _is_retryable_http_error(status_code: int) -> bool:
        return status_code == 408 or status_code == 429 or 500 <= status_code < 600

    def _log_retry(self, attempt: int, attempts: int, reason: str) -> None:
        logger.warning(
            "Retrying Gemini request after %s (%s/%s).",
            reason,
            attempt,
            attempts,
        )

    @staticmethod
    def _build_perception_payload(prompt: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
        """Build the minimal Gemini payload for screenshot perception."""
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }

    @staticmethod
    def _build_text_payload(prompt: str) -> dict[str, Any]:
        """Build the minimal Gemini payload for JSON-only text generation."""
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }

    @staticmethod
    def extract_text(payload: dict[str, Any]) -> str:
        """Extract the first text part from a Gemini generateContent response."""
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GeminiClientError("Gemini response did not include any candidates.")

        first_candidate = candidates[0]
        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise GeminiClientError("Gemini response candidate content was missing.")

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise GeminiClientError("Gemini response candidate parts were missing.")

        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                return part["text"]

        raise GeminiClientError("Gemini response did not include a text part.")


_MAX_SCREENSHOT_WIDTH = 1280
_JPEG_QUALITY = 92


def _optimize_screenshot(path: Path) -> tuple[bytes, float]:
    """Resize to max 1280px wide and convert to JPEG. Returns (bytes, scale_ratio).

    scale_ratio < 1.0 when the image was downscaled. Callers must invert this ratio
    to restore Gemini's returned coordinates back to native pixel space before use.
    """
    try:
        img = Image.open(path)
        scale_ratio = 1.0
        if img.width > _MAX_SCREENSHOT_WIDTH:
            scale_ratio = _MAX_SCREENSHOT_WIDTH / img.width
            new_size = (int(img.width * scale_ratio), int(img.height * scale_ratio))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), scale_ratio
    except Exception:
        return path.read_bytes(), 1.0


class PlaceholderGeminiClient(GeminiClient):
    """Placeholder Gemini client kept for non-perception tests and boundaries."""

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        """Placeholder model interface; external API calls are intentionally disabled."""
        raise NotImplementedError("Gemini perception integration is not configured for this client.")

    async def generate_policy(self, prompt: str) -> str:
        """Placeholder model interface; external API calls are intentionally disabled."""
        raise NotImplementedError("Gemini policy integration is not configured for this client.")

    async def generate_video_verification(self, prompt: str, video_path: str) -> str:
        """Placeholder model interface; external API calls are intentionally disabled."""
        raise NotImplementedError("Gemini video verification is not configured for this client.")


def _extract_usage(*, payload: dict[str, Any], model: str, request_kind: str) -> ModelUsage | None:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    input_tokens = _as_int(usage.get("promptTokenCount"))
    output_tokens = _as_int(usage.get("candidatesTokenCount"))
    total_tokens = _as_int(usage.get("totalTokenCount"))
    cache_creation_input_tokens = _as_int(usage.get("cacheCreationInputTokens"))
    cache_read_input_tokens = _as_int(usage.get("cachedContentTokenCount"))
    thoughts_tokens = _as_int(usage.get("thoughtsTokenCount"))
    input_cost, output_cost, total_cost = estimate_usage_cost(
        provider="gemini",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return ModelUsage(
        provider="gemini",
        model=model,
        request_kind=request_kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        thoughts_tokens=thoughts_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        estimated_cost_usd=total_cost,
    )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
