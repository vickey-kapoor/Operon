from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from src.models.usage import ModelUsage, estimate_usage_cost

logger = logging.getLogger(__name__)


class GeminiComputerUseError(RuntimeError):
    """Raised when the Computer Use client cannot complete a request."""


class GeminiComputerUseClient(ABC):
    @abstractmethod
    async def generate_turn(self, *, contents: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate one Computer Use turn from fully formed content history."""

    @staticmethod
    @abstractmethod
    def build_function_response_content(
        *,
        function_name: str,
        screenshot_path: str,
        current_url: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build a function_response content block for the next user turn."""

    def latest_usage(self) -> ModelUsage | None:
        """Return provider-reported usage for the most recent turn, when available."""
        return None


class GeminiComputerUseHttpClient(GeminiComputerUseClient):
    """HTTP client for Gemini Computer Use via generateContent."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        environment: str = "ENVIRONMENT_BROWSER",
        excluded_predefined_functions: list[str] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url or os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.environment = environment
        self.excluded_predefined_functions = excluded_predefined_functions or []
        self._client: httpx.AsyncClient | None = None
        self._last_usage: ModelUsage | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                http2=True,
            )
        return self._client

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict[str, Any]:
        image_path = Path(screenshot_path)
        contents = [self._build_user_content(prompt=prompt, image_path=image_path)]
        return await self.generate_turn(contents=contents)

    async def generate_turn(self, *, contents: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self._build_payload(contents=contents)
        response_payload = await self._post_payload(payload)
        self._last_usage = _extract_usage(payload=response_payload, model=self.model)
        return self._normalize_response(response_payload)

    def latest_usage(self) -> ModelUsage | None:
        return self._last_usage

    def _build_payload(self, *, contents: list[dict[str, Any]]) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "computerUse": {
                "environment": self.environment,
            }
        }
        if self.excluded_predefined_functions:
            tool["computerUse"]["excludedPredefinedFunctions"] = self.excluded_predefined_functions
        return {
            "contents": contents,
            "tools": [tool],
            "generationConfig": {
                "temperature": 0,
            },
        }

    @staticmethod
    def _build_user_content(*, prompt: str, image_path: Path) -> dict[str, Any]:
        image_bytes = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        return {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ],
        }

    @staticmethod
    def build_function_response_content(
        *,
        function_name: str,
        screenshot_path: str,
        current_url: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        image_path = Path(screenshot_path)
        image_bytes = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        response_payload: dict[str, Any] = {}
        if current_url:
            response_payload["url"] = current_url
        if error:
            response_payload["error"] = error
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": function_name,
                        "response": response_payload,
                    }
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ],
        }

    async def _post_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiComputerUseError("GEMINI_API_KEY is not configured.")

        url = f"{self.api_base_url}/{self.model}:generateContent?key={api_key}"
        client = await self._get_client()
        attempts = self.max_retries + 1
        payload_bytes = json.dumps(payload).encode("utf-8")
        last_error: GeminiComputerUseError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(
                    url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    detail = response.text
                    retryable = response.status_code == 408 or response.status_code == 429 or 500 <= response.status_code < 600
                    last_error = GeminiComputerUseError(
                        f"Gemini Computer Use HTTP error {response.status_code}: {detail}"
                    )
                    if not retryable or attempt >= attempts:
                        raise last_error
                    logger.warning(
                        "Retrying Gemini Computer Use request after http %s (%s/%s).",
                        response.status_code,
                        attempt,
                        attempts,
                    )
                else:
                    return response.json()
            except httpx.TimeoutException as exc:
                last_error = GeminiComputerUseError(
                    f"Gemini Computer Use request timed out after {self.timeout_seconds} seconds."
                )
                if attempt >= attempts:
                    raise last_error from exc
                logger.warning(
                    "Retrying Gemini Computer Use request after timeout (%s/%s).",
                    attempt,
                    attempts,
                )
            except httpx.ConnectError as exc:
                last_error = GeminiComputerUseError(f"Gemini Computer Use connection failed: {exc}")
                if attempt >= attempts:
                    raise last_error from exc
                logger.warning(
                    "Retrying Gemini Computer Use request after connection error (%s/%s).",
                    attempt,
                    attempts,
                )
            await asyncio.sleep(self.retry_backoff_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise GeminiComputerUseError("Gemini Computer Use request failed before receiving a response.")

    @staticmethod
    def _normalize_response(payload: dict[str, Any]) -> dict[str, Any]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise GeminiComputerUseError("Gemini Computer Use response did not include any candidates.")
        first_candidate = candidates[0]
        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise GeminiComputerUseError("Gemini Computer Use candidate content was missing.")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise GeminiComputerUseError("Gemini Computer Use candidate parts were missing.")

        texts = [
            part.get("text", "").strip()
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        function_calls = [
            part.get("function_call") or part.get("functionCall")
            for part in parts
            if isinstance(part, dict)
            and (
                isinstance(part.get("function_call"), dict)
                or isinstance(part.get("functionCall"), dict)
            )
        ]
        summary = " ".join(text for text in texts if text).strip() or "Computer Use step evaluated."
        if not function_calls:
            return {
                "perception": {
                    "summary": summary,
                    "page_hint": "unknown",
                    "visible_elements": [],
                    "confidence": 0.5,
                },
                "action": {"action_type": "wait", "wait_ms": 1000},
                "rationale": summary,
                "confidence": 0.4,
                "active_subgoal": "gather more browser context",
                "model_content": content,
                "raw_candidate": {
                    "finish_reason": first_candidate.get("finishReason"),
                    "text_parts": texts,
                    "had_function_call": False,
                },
            }
        function_call = function_calls[0]
        return {
            "perception": {
                "summary": f"{summary} Tool call: {function_call.get('name', 'unknown')}.",
                "page_hint": "unknown",
                "visible_elements": [],
                "confidence": 0.5,
            },
            "function_call": function_call,
            "function_calls": function_calls,
            "rationale": summary,
            "confidence": 0.7,
            "active_subgoal": function_call.get("name", "browser step"),
            "model_content": content,
            "raw_candidate": {
                "finish_reason": first_candidate.get("finishReason"),
                "text_parts": texts,
                "had_function_call": True,
                "function_name": function_call.get("name"),
                "function_names": [call.get("name") for call in function_calls],
            },
        }


def _extract_usage(*, payload: dict[str, Any], model: str) -> ModelUsage | None:
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
        request_kind="computer_use",
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
