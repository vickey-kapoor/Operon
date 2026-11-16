"""Client for any OpenAI-compatible ``/v1/chat/completions`` endpoint (vLLM, SGLang,
hosted providers)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.models.usage import ModelUsage

logger = logging.getLogger(__name__)

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class ModelTransportError(RuntimeError):
    """The request failed, or the reply has no usable message."""


class OpenAICompatClient:
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
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._last_usage: ModelUsage | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0), transport=self._transport
            )
        return self._client

    def latest_usage(self) -> ModelUsage | None:
        return self._last_usage

    async def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """One chat completion; returns the full response JSON."""
        self._last_usage = None
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": self.profile.max_tokens,
        }
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if api_key := self.profile.api_key():
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.profile.resolved_base_url()}/chat/completions"
        client = await self._get_client()
        attempt = 0
        while True:
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise ModelTransportError(f"{self.profile.name}: request failed: {exc}") from exc
            if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                attempt += 1
                await asyncio.sleep(self.retry_backoff_seconds * attempt)
                continue
            if response.status_code >= 400:
                raise ModelTransportError(
                    f"{self.profile.name}: HTTP {response.status_code}: {response.text[:300]}"
                )
            break
        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelTransportError(f"{self.profile.name}: reply has no message: {response.text[:300]}") from exc
        if not isinstance(message, dict):
            raise ModelTransportError(f"{self.profile.name}: reply message is not an object")
        self._last_usage = self._usage(body.get("usage") or {})
        return body

    def _usage(self, usage: dict[str, Any]) -> ModelUsage:
        input_tokens = _int(usage.get("prompt_tokens"))
        output_tokens = _int(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else None
        pricing = self.profile.pricing
        input_cost = output_cost = total = None
        if pricing.input_per_mtok is not None and input_tokens is not None:
            input_cost = round(input_tokens * pricing.input_per_mtok / 1_000_000, 8)
        if pricing.output_per_mtok is not None and output_tokens is not None:
            output_cost = round(output_tokens * pricing.output_per_mtok / 1_000_000, 8)
        if input_cost is not None or output_cost is not None:
            total = round((input_cost or 0.0) + (output_cost or 0.0), 8)
        return ModelUsage(
            provider="openai_compat",
            model=self.profile.model,
            request_kind="image",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_int(usage.get("total_tokens")),
            cache_read_input_tokens=cached,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            # Self-hosted models cost GPU time, not tokens; the eval report adds that.
            estimated_cost_usd=total,
        )


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
