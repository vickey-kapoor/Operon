"""Focused tests for the Gemini HTTP client boundary."""

from __future__ import annotations

from urllib import error

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


def test_post_json_payload_retries_transient_url_timeout(monkeypatch, caplog) -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        timeout_seconds=1.5,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"candidates":[{"content":{"parts":[{"text":"{}"}]}}]}'

    def fake_urlopen(http_request, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise error.URLError(TimeoutError("timed out"))
        return _Response()

    monkeypatch.setattr("src.clients.gemini.request.urlopen", fake_urlopen)

    with caplog.at_level("WARNING"):
        text = client._post_json_payload(payload)

    assert text == "{}"
    assert attempts["count"] == 3
    assert "Retrying Gemini request" in caplog.text


def test_post_json_payload_fails_cleanly_after_retry_limit(monkeypatch, caplog) -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        timeout_seconds=2.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    def fake_urlopen(http_request, timeout):
        attempts["count"] += 1
        raise error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("src.clients.gemini.request.urlopen", fake_urlopen)

    with caplog.at_level("WARNING"):
        with pytest.raises(GeminiClientError, match="Gemini request failed: timed out"):
            client._post_json_payload(payload)

    assert attempts["count"] == 3
    assert caplog.text.count("Retrying Gemini request") == 2


def test_post_json_payload_does_not_retry_non_retryable_http_error(monkeypatch) -> None:
    client = GeminiHttpClient(
        api_key="test-key",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    payload = GeminiHttpClient._build_text_payload(prompt="choose one action")
    attempts = {"count": 0}

    class FakeHttpError(error.HTTPError):
        def __init__(self):
            super().__init__(
                url="https://example.test",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=None,
            )

        def read(self) -> bytes:
            return b'{"error":"bad request"}'

    def fake_urlopen(http_request, timeout):
        attempts["count"] += 1
        raise FakeHttpError()

    monkeypatch.setattr("src.clients.gemini.request.urlopen", fake_urlopen)

    with pytest.raises(GeminiClientError, match="Gemini HTTP error 400"):
        client._post_json_payload(payload)

    assert attempts["count"] == 1
