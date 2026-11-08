"""OpenAI-compatible transport: request shape, auth, retries, usage and cost."""

from __future__ import annotations

import json

import httpx
import pytest

from operon.agent.backends.model_adapter.registry import ModelProfile, Pricing
from operon.agent.backends.model_adapter.transports import (
    ModelTransportError,
    OpenAICompatClient,
)

from .conftest import sample


def _profile(**overrides) -> ModelProfile:
    base = {"name": "qwen", "adapter": "openai_compat", "model": "Qwen/Qwen3-VL-32B-Instruct", "base_url": "http://gpu:8000/v1"}
    return ModelProfile(**{**base, **overrides})


async def test_request_shape_and_bearer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=sample("qwen3_vl_click"))

    monkeypatch.setenv("OPENWEIGHT_API_KEY", "vllm-key")
    client = OpenAICompatClient(_profile(), transport=httpx.MockTransport(handler))
    body = await client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "click"}}])

    request = seen[0]
    assert str(request.url) == "http://gpu:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer vllm-key"
    payload = json.loads(request.content)
    assert payload["model"] == "Qwen/Qwen3-VL-32B-Instruct" and payload["temperature"] == 0.0
    assert payload["tool_choice"] == "auto" and payload["max_tokens"] == 1024
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "click"
    usage = client.latest_usage()
    assert usage.provider == "openai_compat" and usage.input_tokens == 1834 and usage.estimated_cost_usd is None


async def test_token_pricing_gives_a_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample("qwen3_vl_click"))

    profile = _profile(pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0))
    client = OpenAICompatClient(profile, transport=httpx.MockTransport(handler))
    await client.chat([], tools=None)
    assert client.latest_usage().estimated_cost_usd == pytest.approx((1834 * 1.0 + 37 * 2.0) / 1e6)


async def test_retries_a_busy_server_then_gives_up() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="loading model")

    client = OpenAICompatClient(_profile(), transport=httpx.MockTransport(handler), retry_backoff_seconds=0)
    with pytest.raises(ModelTransportError, match="HTTP 503"):
        await client.chat([], tools=None)
    assert calls == 3


async def test_a_reply_without_a_message_is_an_error() -> None:
    client = OpenAICompatClient(
        _profile(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))
    )
    with pytest.raises(ModelTransportError, match="no message"):
        await client.chat([], tools=None)
