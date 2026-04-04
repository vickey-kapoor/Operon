"""Focused tests for the Gemini Computer Use HTTP client."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from src.clients.gemini_computer_use import (
    GeminiComputerUseError,
    GeminiComputerUseHttpClient,
)


def test_build_payload_includes_computer_use_tool() -> None:
    client = GeminiComputerUseHttpClient(api_key="test-key", model="gemini-3-flash-preview")
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Search for pricing"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(b"img").decode("utf-8"),
                    }
                },
            ],
        }
    ]

    payload = client._build_payload(contents=contents)

    assert payload["tools"][0]["computerUse"]["environment"] == "ENVIRONMENT_BROWSER"
    assert payload["contents"][0]["parts"][0]["text"] == "Search for pricing"
    assert payload["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"


def test_normalize_response_extracts_function_call() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Click the pricing link."},
                        {"function_call": {"name": "click_at", "args": {"x": 500, "y": 250}}},
                    ]
                }
            }
        ]
    }

    normalized = GeminiComputerUseHttpClient._normalize_response(payload)

    assert normalized["function_call"]["name"] == "click_at"
    assert normalized["function_calls"][0]["name"] == "click_at"
    assert normalized["rationale"] == "Click the pricing link."
    assert normalized["raw_candidate"]["had_function_call"] is True
    assert normalized["raw_candidate"]["function_name"] == "click_at"


def test_normalize_response_extracts_camel_case_function_call() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Click the pricing link."},
                        {"functionCall": {"name": "click_at", "args": {"x": 500, "y": 250}}},
                    ]
                }
            }
        ]
    }

    normalized = GeminiComputerUseHttpClient._normalize_response(payload)

    assert normalized["function_call"]["name"] == "click_at"
    assert normalized["raw_candidate"]["had_function_call"] is True


def test_normalize_response_preserves_multiple_function_calls() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Open the page and wait briefly."},
                        {"function_call": {"name": "navigate", "args": {"url": "https://example.com"}}},
                        {"function_call": {"name": "wait_5_seconds", "args": {}}},
                    ]
                }
            }
        ]
    }

    normalized = GeminiComputerUseHttpClient._normalize_response(payload)

    assert [call["name"] for call in normalized["function_calls"]] == ["navigate", "wait_5_seconds"]
    assert normalized["raw_candidate"]["function_names"] == ["navigate", "wait_5_seconds"]


def test_normalize_response_without_function_call_defaults_to_wait() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "The page is visible but I am not confident about the next action."},
                    ]
                }
            }
        ]
    }

    normalized = GeminiComputerUseHttpClient._normalize_response(payload)

    assert normalized["action"]["action_type"] == "wait"
    assert normalized["action"]["wait_ms"] == 1000
    assert normalized["active_subgoal"] == "gather more browser context"
    assert normalized["raw_candidate"]["had_function_call"] is False


@pytest.mark.asyncio
async def test_post_payload_retries_timeout() -> None:
    client = GeminiComputerUseHttpClient(
        api_key="test-key",
        model="gemini-3-flash-preview",
        timeout_seconds=1.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = {"contents": []}
    attempts = {"count": 0}

    ok_response = httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "done"}]}}]})

    async def fake_post(url, *, content=None, headers=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.TimeoutException("timed out")
        return ok_response

    mock_client = AsyncMock()
    mock_client.post = fake_post
    mock_client.is_closed = False
    client._client = mock_client

    response = await client._post_payload(payload)

    assert response["candidates"][0]["content"]["parts"][0]["text"] == "done"
    assert attempts["count"] == 3


def test_normalize_response_rejects_missing_candidates() -> None:
    with pytest.raises(GeminiComputerUseError, match="candidates"):
        GeminiComputerUseHttpClient._normalize_response({})


def test_build_function_response_content_includes_url_and_screenshot(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xa5\x1d\xb8"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    content = GeminiComputerUseHttpClient.build_function_response_content(
        function_name="click_at",
        screenshot_path=str(screenshot),
        current_url="https://example.com",
    )

    assert content["role"] == "user"
    assert content["parts"][0]["function_response"]["name"] == "click_at"
    assert content["parts"][0]["function_response"]["response"]["url"] == "https://example.com"
    assert content["parts"][1]["inline_data"]["mime_type"] == "image/png"
