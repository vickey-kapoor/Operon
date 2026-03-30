"""Focused tests for the Gemini HTTP client boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.clients.gemini import GeminiClientError, GeminiHttpClient


def test_extract_text_returns_first_text_part() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"summary":"Inbox visible","page_hint":"gmail_inbox","focused_element_id":null,"confidence":0.9,"visible_elements":[]}'},
                    ]
                }
            }
        ]
    }

    text = GeminiHttpClient.extract_text(payload)

    assert '"summary":"Inbox visible"' in text


def test_extract_text_rejects_missing_candidates() -> None:
    with pytest.raises(GeminiClientError, match="candidates"):
        GeminiHttpClient.extract_text({})


def test_build_perception_payload_requests_json_output() -> None:
    payload = GeminiHttpClient._build_perception_payload(prompt="perceive", image_bytes=b"image-bytes")

    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["contents"][0]["parts"][0]["text"] == "perceive"
    assert payload["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"


def test_build_text_payload_requests_json_output() -> None:
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")

    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["contents"][0]["parts"] == [{"text": "choose one action"}]


@pytest.mark.asyncio
async def test_post_json_payload_retries_transient_timeout(caplog) -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        timeout_seconds=1.5,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    ok_response = httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
    )

    async def fake_post(url, *, content=None, json=None, headers=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.TimeoutException("timed out")
        return ok_response

    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.is_closed = False
    client._client = mock_client

    with caplog.at_level("WARNING"):
        text = await client._post_json_payload(payload)

    assert text == "{}"
    assert attempts["count"] == 3
    assert "Retrying Gemini request" in caplog.text


@pytest.mark.asyncio
async def test_post_json_payload_fails_cleanly_after_retry_limit(caplog) -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        timeout_seconds=2.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    async def fake_post(url, *, content=None, json=None, headers=None, **kwargs):
        attempts["count"] += 1
        raise httpx.TimeoutException("timed out")

    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.is_closed = False
    client._client = mock_client

    with caplog.at_level("WARNING"):
        with pytest.raises(GeminiClientError, match="timed out"):
            await client._post_json_payload(payload)

    assert attempts["count"] == 3
    assert caplog.text.count("Retrying Gemini request") == 2


@pytest.mark.asyncio
async def test_post_json_payload_does_not_retry_non_retryable_http_error() -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    async def fake_post(url, *, content=None, json=None, headers=None, **kwargs):
        attempts["count"] += 1
        return httpx.Response(400, text='{"error":"bad request"}')

    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.is_closed = False
    client._client = mock_client

    with pytest.raises(GeminiClientError, match="Gemini HTTP error 400"):
        await client._post_json_payload(payload)

    assert attempts["count"] == 1
