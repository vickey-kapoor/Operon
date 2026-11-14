"""Tests for Human-in-the-Loop subsystem (hitl.py).

Covers: generate_hitl_message, post_hitl_webhook, start_escalation_timer.

Tier progression: SIMPLE → MODERATE → COMPLEX
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from operon.agent.hitl import (
    generate_hitl_message,
    post_hitl_webhook,
    start_escalation_timer,
)

# ===========================================================================
# SIMPLE: generate_hitl_message
# ===========================================================================

@pytest.mark.asyncio
async def test_generate_hitl_message_no_client_returns_fallback() -> None:
    """When no Gemini client is provided the fallback message is returned immediately."""
    msg = await generate_hitl_message(
        intent="Fill form",
        page_hint="captcha",
        url="https://example.com",
        visible_element_names=["CAPTCHA widget"],
        gemini_client=None,
    )
    assert "captcha" in msg.lower()
    assert "Resume" in msg or "resume" in msg.lower()


@pytest.mark.asyncio
async def test_generate_hitl_message_llm_failure_returns_fallback() -> None:
    """When the LLM call raises an exception the fallback message is returned."""
    client = MagicMock()
    client.generate_policy = AsyncMock(side_effect=RuntimeError("quota exceeded"))

    msg = await generate_hitl_message(
        intent="Submit report",
        page_hint="login",
        url=None,
        visible_element_names=[],
        gemini_client=client,  # type: ignore[arg-type]
    )
    assert "login" in msg.lower() or "Resume" in msg


@pytest.mark.asyncio
async def test_generate_hitl_message_empty_llm_response_returns_fallback() -> None:
    """When the LLM returns an empty string the fallback message is used."""
    client = MagicMock()
    client.generate_policy = AsyncMock(return_value="   ")

    msg = await generate_hitl_message(
        intent="Buy ticket",
        page_hint="checkout",
        url="https://shop.example.com",
        visible_element_names=["Credit card form", "Pay button"],
        gemini_client=client,  # type: ignore[arg-type]
    )
    assert msg  # never empty
    assert "checkout" in msg.lower() or "Resume" in msg


@pytest.mark.asyncio
async def test_generate_hitl_message_uses_llm_response_when_valid() -> None:
    """When the LLM returns text it is passed through as the message."""
    expected = "The agent hit a CAPTCHA. Please complete it in the browser, then click Resume."
    client = MagicMock()
    client.generate_policy = AsyncMock(return_value=expected)

    msg = await generate_hitl_message(
        intent="Submit form",
        page_hint="captcha",
        url="https://example.com",
        visible_element_names=["CAPTCHA"],
        gemini_client=client,  # type: ignore[arg-type]
    )
    assert msg == expected


# ===========================================================================
# MODERATE: post_hitl_webhook
# ===========================================================================

@pytest.mark.asyncio
async def test_post_hitl_webhook_skips_when_no_url_configured() -> None:
    """When HITL_WEBHOOK_URL is unset the function returns silently (no network call)."""
    with patch.dict("os.environ", {"HITL_WEBHOOK_URL": ""}):
        # Would fail if it attempted an actual HTTP call
        await post_hitl_webhook(
            run_id="run-1",
            intent="Fill form",
            page_hint="captcha",
            message="Please solve the CAPTCHA.",
            url="https://example.com",
        )
    # No assertion needed — the test passes if no exception is raised


@pytest.mark.asyncio
async def test_post_hitl_webhook_posts_correct_payload(tmp_path) -> None:
    """Webhook posts all required fields as JSON."""
    received_payload: dict = {}


    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 200

    async def _mock_post(url, *, content, headers):
        nonlocal received_payload
        received_payload = json.loads(content)
        return mock_response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = _mock_post

    with patch.dict("os.environ", {"HITL_WEBHOOK_URL": "https://hooks.example.com/notify"}):
        with patch("operon.agent.hitl.httpx.AsyncClient", return_value=mock_client):
            await post_hitl_webhook(
                run_id="run-abc",
                intent="Submit report",
                page_hint="captcha",
                message="CAPTCHA detected. Please solve it.",
                url="https://target.example.com",
            )

    assert received_payload["event"] == "hitl_triggered"
    assert received_payload["run_id"] == "run-abc"
    assert received_payload["page_hint"] == "captcha"
    assert received_payload["intent"] == "Submit report"
    assert "message" in received_payload


@pytest.mark.asyncio
async def test_post_hitl_webhook_does_not_raise_on_http_error() -> None:
    """A failed webhook response should log a warning but not raise."""
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.dict("os.environ", {"HITL_WEBHOOK_URL": "https://hooks.example.com/notify"}):
        with patch("operon.agent.hitl.httpx.AsyncClient", return_value=mock_client):
            await post_hitl_webhook(
                run_id="run-1",
                intent="Fill form",
                page_hint="captcha",
                message="Help needed.",
                url=None,
            )
    # Test passes if no exception propagates


@pytest.mark.asyncio
async def test_post_hitl_webhook_does_not_raise_on_network_error() -> None:
    """A network error during webhook delivery should not raise."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=ConnectionError("network unreachable"))

    with patch.dict("os.environ", {"HITL_WEBHOOK_URL": "https://hooks.example.com/notify"}):
        with patch("operon.agent.hitl.httpx.AsyncClient", return_value=mock_client):
            await post_hitl_webhook(
                run_id="run-1",
                intent="Buy ticket",
                page_hint="blocked",
                message="Blocked.",
                url="https://target.example.com",
            )


# ===========================================================================
# MODERATE: start_escalation_timer
# ===========================================================================

@pytest.mark.asyncio
async def test_escalation_timer_exits_early_when_status_changes() -> None:
    """Timer should stop re-notifying once the run leaves WAITING_FOR_USER."""
    notify_calls: list[str] = []

    async def _get_status():
        return "running"  # already resumed

    def _notify():
        notify_calls.append("notified")

    await start_escalation_timer(
        "run-1",
        get_status_fn=_get_status,
        notify_fn=_notify,
        intervals_seconds=(0, 0, 0),  # immediate for test speed
    )

    assert len(notify_calls) == 0


@pytest.mark.asyncio
async def test_escalation_timer_notifies_when_still_waiting() -> None:
    """Timer should notify once for each interval where the run is still waiting."""
    notify_calls: list[str] = []
    call_count = 0

    async def _get_status():
        nonlocal call_count
        call_count += 1
        # Still waiting on first check, resolved on second
        return "waiting_for_user" if call_count <= 1 else "running"

    def _notify():
        notify_calls.append("notified")

    await start_escalation_timer(
        "run-1",
        get_status_fn=_get_status,
        notify_fn=_notify,
        intervals_seconds=(0, 0),  # immediate for speed
    )

    assert len(notify_calls) == 1  # notified once, then exited


@pytest.mark.asyncio
async def test_escalation_timer_handles_get_status_exception() -> None:
    """An exception from get_status_fn should silently abort the timer."""
    async def _get_status():
        raise RuntimeError("store gone")

    notify_calls: list[str] = []

    await start_escalation_timer(
        "run-1",
        get_status_fn=_get_status,
        notify_fn=lambda: notify_calls.append("x"),
        intervals_seconds=(0,),
    )
    assert len(notify_calls) == 0


@pytest.mark.asyncio
async def test_escalation_timer_handles_notify_exception() -> None:
    """An exception in notify_fn should not propagate — timer continues."""
    notified = []

    async def _always_waiting():
        return "waiting_for_user"

    def _bad_notify():
        if len(notified) == 0:
            notified.append("fail")
            raise OSError("notification daemon unavailable")
        notified.append("ok")

    # Should not raise even though first notify throws
    await start_escalation_timer(
        "run-1",
        get_status_fn=_always_waiting,
        notify_fn=_bad_notify,
        intervals_seconds=(0, 0),
    )
