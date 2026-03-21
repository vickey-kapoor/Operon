"""Gemini client interface and thin HTTP implementation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import error, request

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


class GeminiHttpClient(GeminiClient):
    """Thin REST client for Gemini perception and policy calls."""

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

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        """Send the screenshot and prompt to Gemini and return raw JSON text."""
        return await asyncio.to_thread(self._generate_perception_sync, prompt, screenshot_path)

    async def generate_policy(self, prompt: str) -> str:
        """Send the policy prompt to Gemini and return raw JSON text."""
        return await asyncio.to_thread(self._generate_text_sync, prompt)

    def _generate_perception_sync(self, prompt: str, screenshot_path: str) -> str:
        image_bytes = Path(screenshot_path).read_bytes()
        payload = self._build_perception_payload(prompt=prompt, image_bytes=image_bytes)
        return self._post_json_payload(payload)

    def _generate_text_sync(self, prompt: str) -> str:
        payload = self._build_text_payload(prompt=prompt)
        return self._post_json_payload(payload)

    def _post_json_payload(self, payload: dict[str, Any]) -> str:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiClientError("GEMINI_API_KEY is not configured.")

        url = f"{self.api_base_url}/{self.model}:generateContent?key={api_key}"
        attempts = self.max_retries + 1
        last_error: GeminiClientError | None = None

        for attempt in range(1, attempts + 1):
            http_request = request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = self._is_retryable_http_error(exc.code)
                last_error = GeminiClientError(f"Gemini HTTP error {exc.code}: {detail}")
                if not retryable or attempt >= attempts:
                    raise last_error from exc
                self._log_retry(attempt, attempts, f"http {exc.code}")
            except error.URLError as exc:
                retryable = self._is_retryable_url_error(exc.reason)
                last_error = GeminiClientError(f"Gemini request failed: {exc.reason}")
                if not retryable or attempt >= attempts:
                    raise last_error from exc
                self._log_retry(attempt, attempts, f"network error: {exc.reason}")
            except TimeoutError as exc:
                last_error = GeminiClientError(f"Gemini request timed out after {self.timeout_seconds} seconds.")
                if attempt >= attempts:
                    raise last_error from exc
                self._log_retry(attempt, attempts, f"timeout after {self.timeout_seconds} seconds")

            self._sleep_before_retry(attempt)
        else:
            if last_error is not None:
                raise last_error
            raise GeminiClientError("Gemini request failed before receiving a response.")

        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise GeminiClientError("Gemini response was not valid JSON.") from exc
        return self.extract_text(response_payload)

    @staticmethod
    def _is_retryable_http_error(status_code: int) -> bool:
        return status_code == 408 or status_code == 429 or 500 <= status_code < 600

    @staticmethod
    def _is_retryable_url_error(reason: object) -> bool:
        return isinstance(reason, (TimeoutError, socket.timeout))

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(self.retry_backoff_seconds * attempt)

    def _log_retry(self, attempt: int, attempts: int, reason: str) -> None:
        logger.warning(
            "Retrying Gemini request after %s (%s/%s).",
            reason,
            attempt,
            attempts,
        )

    @staticmethod
    def _build_perception_payload(prompt: str, image_bytes: bytes) -> dict[str, Any]:
        """Build the minimal Gemini payload for screenshot perception."""
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
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


class PlaceholderGeminiClient(GeminiClient):
    """Placeholder Gemini client kept for non-perception tests and boundaries."""

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        """Placeholder model interface; external API calls are intentionally disabled."""
        raise NotImplementedError("Gemini perception integration is not configured for this client.")

    async def generate_policy(self, prompt: str) -> str:
        """Placeholder model interface; external API calls are intentionally disabled."""
        raise NotImplementedError("Gemini policy integration is not configured for this client.")
