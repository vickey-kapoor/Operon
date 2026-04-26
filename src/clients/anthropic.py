"""Anthropic client for planner-only text generation."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from src.models.usage import ModelUsage, estimate_usage_cost


class AnthropicClientError(RuntimeError):
    """Raised when Anthropic requests or responses are invalid."""


class AnthropicHttpClient:
    """Async HTTP client for Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        api_base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url or os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com/v1/messages")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
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

    async def generate_policy(self, prompt: str) -> str:
        return await self._post_message(
            payload={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            request_kind="text",
        )

    async def generate_verification(self, prompt: str, screenshot_path: str) -> str:
        image_path = Path(screenshot_path)
        image_bytes = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        media_type = "image/png" if mime_type == "image/png" else "image/jpeg"
        return await self._post_message(
            payload={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                                },
                            },
                        ],
                    }
                ],
            },
            request_kind="image",
        )

    def latest_usage(self) -> ModelUsage | None:
        return self._last_usage

    async def _post_message(self, payload: dict[str, Any], *, request_kind: str) -> str:
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise AnthropicClientError("ANTHROPIC_API_KEY is not configured.")

        payload_bytes = json.dumps(payload).encode("utf-8")
        client = await self._get_client()
        attempts = self.max_retries + 1
        last_error: AnthropicClientError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(
                    self.api_base_url,
                    content=payload_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                if response.status_code != 200:
                    detail = response.text
                    retryable = response.status_code in {408, 429} or 500 <= response.status_code < 600
                    last_error = AnthropicClientError(f"Anthropic HTTP error {response.status_code}: {detail}")
                    if not retryable or attempt >= attempts:
                        raise last_error
                else:
                    response_payload = response.json()
                    self._last_usage = _extract_usage(payload=response_payload, model=self.model, request_kind=request_kind)
                    return _extract_text(response_payload)
            except httpx.TimeoutException as exc:
                last_error = AnthropicClientError(
                    f"Anthropic request timed out after {self.timeout_seconds} seconds."
                )
                if attempt >= attempts:
                    raise last_error from exc
            except httpx.ConnectError as exc:
                last_error = AnthropicClientError(f"Anthropic connection failed: {exc}")
                if attempt >= attempts:
                    raise last_error from exc

            await asyncio.sleep(self.retry_backoff_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise AnthropicClientError("Anthropic request failed before receiving a response.")


def _extract_text(response_payload: dict[str, Any]) -> str:
    content = response_payload.get("content")
    if not isinstance(content, list):
        raise AnthropicClientError("Anthropic response missing content list.")

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    combined = "".join(text_parts).strip()
    if not combined:
        raise AnthropicClientError("Anthropic response did not contain text content.")
    return combined


def _extract_usage(*, payload: dict[str, Any], model: str, request_kind: str) -> ModelUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    cache_creation_input_tokens = _as_int(usage.get("cache_creation_input_tokens"))
    cache_read_input_tokens = _as_int(usage.get("cache_read_input_tokens"))
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    input_cost, output_cost, total_cost = estimate_usage_cost(
        provider="anthropic",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return ModelUsage(
        provider="anthropic",
        model=model,
        request_kind=request_kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        estimated_cost_usd=total_cost,
    )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
