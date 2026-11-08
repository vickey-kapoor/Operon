"""Client for OpenAI's Responses API with the ``computer`` tool."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports.openai_compat import (
    ModelTransportError,
)
from operon.models.usage import ModelUsage

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_DEFAULT_URL = "https://api.openai.com/v1/responses"


class OpenAIResponsesClient:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.profile = profile
        self.url = (os.getenv("OPENAI_API_BASE_URL") or _DEFAULT_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._last_usage: ModelUsage | None = None

    def latest_usage(self) -> ModelUsage | None:
        return self._last_usage

    async def create(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        self._last_usage = None
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ModelTransportError("OPENAI_API_KEY is not set")
        payload: dict[str, Any] = {"model": self.profile.model, "tools": tools, "input": input_items}
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if instructions:
            payload["instructions"] = instructions
        if self.profile.effort:
            payload["reasoning"] = {"effort": self.profile.effort}
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0), transport=self._transport
            )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        attempt = 0
        while True:
            try:
                response = await self._client.post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise ModelTransportError(f"OpenAI request failed: {exc}") from exc
            if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                attempt += 1
                await asyncio.sleep(self.retry_backoff_seconds * attempt)
                continue
            if response.status_code >= 400:
                raise ModelTransportError(f"OpenAI HTTP {response.status_code}: {response.text[:300]}")
            break
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelTransportError(f"OpenAI reply is not JSON: {response.text[:300]}") from exc
        if not isinstance(body, dict) or not isinstance(body.get("output"), list) or not body.get("id"):
            raise ModelTransportError(f"OpenAI reply has no output: {str(body)[:300]}")
        self._last_usage = self._usage(body.get("usage") or {})
        return body

    def _usage(self, usage: dict[str, Any]) -> ModelUsage:
        def count(value: Any) -> int | None:
            return value if isinstance(value, int) and value >= 0 else None

        input_tokens, output_tokens = count(usage.get("input_tokens")), count(usage.get("output_tokens"))
        cached = count((usage.get("input_tokens_details") or {}).get("cached_tokens"))
        reasoning = count((usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
        pricing = self.profile.pricing
        input_cost = output_cost = total = None
        if pricing.input_per_mtok is not None and input_tokens is not None:
            input_cost = round(input_tokens * pricing.input_per_mtok / 1_000_000, 8)
        if pricing.output_per_mtok is not None and output_tokens is not None:
            # Reasoning tokens are part of output_tokens and bill as output.
            output_cost = round(output_tokens * pricing.output_per_mtok / 1_000_000, 8)
        if input_cost is not None or output_cost is not None:
            total = round((input_cost or 0.0) + (output_cost or 0.0), 8)
        # Operon counts thoughts_tokens on top of output_tokens; OpenAI counts reasoning
        # inside output_tokens, so the visible part is reported as output.
        visible_output = output_tokens
        if output_tokens is not None and reasoning is not None:
            visible_output = max(0, output_tokens - reasoning)
        return ModelUsage(
            provider="openai",
            model=self.profile.model,
            request_kind="image",
            input_tokens=input_tokens,
            output_tokens=visible_output,
            total_tokens=count(usage.get("total_tokens")),
            cache_read_input_tokens=cached,
            thoughts_tokens=reasoning,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            estimated_cost_usd=total,
        )
