"""Focused tests for the Gemini HTTP client boundary."""

from __future__ import annotations

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
    assert payload["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"


def test_build_text_payload_requests_json_output() -> None:
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")

    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["contents"][0]["parts"] == [{"text": "choose one action"}]
