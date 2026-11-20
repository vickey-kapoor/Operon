"""Thinking settings the plain Anthropic client sends per model (the full harness's
critic and the planner use it), and its retry and error paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from operon.clients.anthropic import (
    AnthropicClientError,
    AnthropicHttpClient,
    _extract_text,
    _extract_usage,
    thinking_off_fields,
)


def test_sonnet_5_5_turns_thinking_off_with_between_tools() -> None:
    # Sonnet 5.5 answers {"type": "disabled"} with a 400.
    assert thinking_off_fields("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-2", "claude-mythos-5-2"])
def test_models_that_always_think_get_low_effort_and_no_thinking_field(model: str) -> None:
    assert thinking_off_fields(model) == {"output_config": {"effort": "low"}}


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-5-5"])
def test_other_models_keep_disabled_thinking(model: str) -> None:
    assert thinking_off_fields(model) == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_the_policy_request_for_sonnet_5_5_never_sends_disabled_thinking() -> None:
    client = AnthropicHttpClient(api_key="test", model="claude-sonnet-5-5")
    client._post_json = AsyncMock(return_value={"content": [{"type": "text", "text": "{}"}]})

    await client.generate_policy("Plan the next step.")

    for call in client._post_json.await_args_list:
        payload = call.args[0]
        assert payload["max_tokens"] == 4096
        assert payload["thinking"] == {"type": "between_tools"}
        assert "temperature" not in payload


# --- retry and error paths of _post_json (fake HTTP client, no network, no real sleeping) ---

OK_BODY = {
    "content": [{"type": "text", "text": " hello "}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _client_with(monkeypatch, responses, **kwargs):
    """Client whose HTTP layer replays ``responses`` (a Response or an exception per call)."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("operon.clients.anthropic.asyncio.sleep", fake_sleep)
    client = AnthropicHttpClient(api_key="test-key", model="claude-sonnet-5", **kwargs)
    calls: list[dict] = []
    queue = list(responses)

    async def fake_post(url, *, content=None, headers=None, **_):
        calls.append({"url": url, "content": content, "headers": headers})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    fake = AsyncMock()
    fake.post = fake_post
    fake.is_closed = False
    client._client = fake
    return client, calls, sleeps


@pytest.mark.asyncio
async def test_success_returns_text_sends_headers_and_records_usage(monkeypatch) -> None:
    client, calls, sleeps = _client_with(monkeypatch, [httpx.Response(200, json=OK_BODY)])

    assert await client.generate_policy("plan") == "hello"

    assert len(calls) == 1 and sleeps == []
    assert calls[0]["headers"]["x-api-key"] == "test-key"
    assert calls[0]["headers"]["anthropic-version"] == "2023-06-01"
    usage = client.latest_usage()
    assert usage is not None and usage.input_tokens == 10 and usage.total_tokens == 15


@pytest.mark.asyncio
async def test_429_then_success_retries_with_linear_backoff(monkeypatch) -> None:
    client, calls, sleeps = _client_with(
        monkeypatch,
        [httpx.Response(429, text="slow"), httpx.Response(429, text="slow"), httpx.Response(200, json=OK_BODY)],
        max_retries=2,
        retry_backoff_seconds=0.5,
    )

    assert await client.generate_policy("plan") == "hello"

    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503, 529, 599])
async def test_5xx_retries_up_to_the_limit_then_raises_typed_error(monkeypatch, status) -> None:
    client, calls, sleeps = _client_with(
        monkeypatch, [httpx.Response(status, text="boom")] * 3, max_retries=2, retry_backoff_seconds=0.1
    )

    with pytest.raises(AnthropicClientError, match=f"HTTP error {status}: boom"):
        await client.generate_policy("plan")

    assert len(calls) == 3
    assert sleeps == pytest.approx([0.1, 0.2])  # no sleep after the final attempt
    assert client.latest_usage() is None


@pytest.mark.asyncio
async def test_408_is_retried(monkeypatch) -> None:
    client, calls, _ = _client_with(monkeypatch, [httpx.Response(408, text="t"), httpx.Response(200, json=OK_BODY)])

    assert await client.generate_policy("plan") == "hello"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
async def test_non_retryable_4xx_raises_without_retry(monkeypatch, status) -> None:
    client, calls, sleeps = _client_with(monkeypatch, [httpx.Response(status, text="nope")], max_retries=3)

    with pytest.raises(AnthropicClientError, match=f"HTTP error {status}: nope"):
        await client.generate_policy("plan")

    assert len(calls) == 1 and sleeps == []


@pytest.mark.asyncio
async def test_timeout_is_retried_then_succeeds(monkeypatch) -> None:
    client, calls, sleeps = _client_with(
        monkeypatch, [httpx.ReadTimeout("t"), httpx.Response(200, json=OK_BODY)], retry_backoff_seconds=0.25
    )

    assert await client.generate_policy("plan") == "hello"
    assert len(calls) == 2 and sleeps == [0.25]


@pytest.mark.asyncio
async def test_timeout_every_attempt_raises_typed_error_chained_to_cause(monkeypatch) -> None:
    client, calls, _ = _client_with(monkeypatch, [httpx.ReadTimeout("t")] * 2, max_retries=1, timeout_seconds=7.0)

    with pytest.raises(AnthropicClientError, match="timed out after 7.0 seconds") as info:
        await client.generate_policy("plan")

    assert len(calls) == 2
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


@pytest.mark.asyncio
async def test_connect_error_retries_then_raises_typed_error(monkeypatch) -> None:
    client, calls, sleeps = _client_with(
        monkeypatch, [httpx.ConnectError("refused")] * 3, max_retries=2, retry_backoff_seconds=0.0
    )

    with pytest.raises(AnthropicClientError, match="connection failed: refused"):
        await client.generate_policy("plan")

    assert len(calls) == 3 and len(sleeps) == 2


@pytest.mark.asyncio
async def test_connect_error_then_success(monkeypatch) -> None:
    client, calls, _ = _client_with(monkeypatch, [httpx.ConnectError("x"), httpx.Response(200, json=OK_BODY)])

    assert await client.generate_policy("plan") == "hello"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_zero_retries_means_one_attempt(monkeypatch) -> None:
    client, calls, sleeps = _client_with(monkeypatch, [httpx.Response(503, text="down")], max_retries=0)

    with pytest.raises(AnthropicClientError, match="HTTP error 503"):
        await client.generate_policy("plan")

    assert len(calls) == 1 and sleeps == []


def test_negative_retry_settings_are_clamped() -> None:
    client = AnthropicHttpClient(api_key="k", max_retries=-3, retry_backoff_seconds=-1.0)
    assert client.max_retries == 0 and client.retry_backoff_seconds == 0.0


@pytest.mark.asyncio
async def test_missing_api_key_raises_before_any_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "operon.clients.anthropic.get_settings",
        lambda: SimpleNamespace(anthropic_api_key=None, anthropic_api_base_url="http://x"),
    )
    client = AnthropicHttpClient(api_key=None)
    client._get_client = AsyncMock()  # must never be reached

    with pytest.raises(AnthropicClientError, match="ANTHROPIC_API_KEY is not configured"):
        await client.generate_policy("plan")

    client._get_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_key_falls_back_to_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "operon.clients.anthropic.get_settings",
        lambda: SimpleNamespace(anthropic_api_key="from-settings", anthropic_api_base_url="http://x"),
    )
    client, calls, _ = _client_with(monkeypatch, [httpx.Response(200, json=OK_BODY)])
    client.api_key = None

    await client.generate_policy("plan")

    assert calls[0]["headers"]["x-api-key"] == "from-settings"


@pytest.mark.asyncio
async def test_failed_call_clears_previous_usage(monkeypatch) -> None:
    client, _, _ = _client_with(monkeypatch, [httpx.Response(200, json=OK_BODY), httpx.Response(400, text="bad")])

    await client.generate_policy("plan")
    assert client.latest_usage() is not None
    with pytest.raises(AnthropicClientError):
        await client.generate_policy("plan")

    assert client.latest_usage() is None


@pytest.mark.asyncio
async def test_extra_headers_are_merged(monkeypatch) -> None:
    client, calls, _ = _client_with(monkeypatch, [httpx.Response(200, json=OK_BODY)])

    await client._post_json({"m": 1}, request_kind="text", extra_headers={"anthropic-beta": "x"})

    assert calls[0]["headers"]["anthropic-beta"] == "x"
    assert calls[0]["headers"]["x-api-key"] == "test-key"


@pytest.mark.parametrize("body", [{}, {"content": "str"}, {"content": []}, {"content": [{"type": "tool_use"}]}])
def test_extract_text_rejects_bad_content(body) -> None:
    with pytest.raises(AnthropicClientError):
        _extract_text(body)


def test_extract_text_joins_text_blocks_and_skips_others() -> None:
    body = {"content": ["junk", {"type": "thinking"}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _extract_text(body) == "ab"


def test_extract_usage_ignores_missing_or_invalid_fields() -> None:
    assert _extract_usage(payload={}, model="claude-sonnet-5", request_kind="text") is None
    usage = _extract_usage(
        payload={"usage": {"input_tokens": -1, "output_tokens": 4}}, model="claude-sonnet-5", request_kind="text"
    )
    assert usage is not None and usage.input_tokens is None and usage.total_tokens == 4
