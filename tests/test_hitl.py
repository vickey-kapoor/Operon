"""Tests for Human-in-the-Loop subsystem (hitl.py).

Covers: generate_hitl_message, post_hitl_webhook, start_escalation_timer,
and the HITL debounce inside PolicyRuleEngine._human_intervention_rule.

Tier progression: SIMPLE → MODERATE → COMPLEX
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.hitl import (
    HITL_PAGE_HINT_KEYWORDS,
    generate_hitl_message,
    post_hitl_webhook,
    start_escalation_timer,
)
from src.agent.policy_rules import PolicyRuleEngine
from src.models.common import RunStatus
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.state import AgentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(run_id: str = "run-1") -> AgentState:
    return AgentState(run_id=run_id, intent="Fill out contact form", status=RunStatus.RUNNING)


def _perception(page_hint: str = "captcha") -> ScreenPerception:
    return ScreenPerception(
        summary="A CAPTCHA is visible",
        page_hint=page_hint,
        capture_artifact_path="runs/test/step_1/before.png",
        visible_elements=[
            UIElement(
                element_id="captcha-widget",
                element_type=UIElementType.UNKNOWN,
                label="I am not a robot",
                x=300,
                y=400,
                width=200,
                height=60,
                is_interactable=True,
                confidence=0.9,
            )
        ],
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
# SIMPLE: HITL_PAGE_HINT_KEYWORDS coverage
# ===========================================================================

@pytest.mark.parametrize("keyword", [
    "captcha", "recaptcha", "robot",
    "login", "sign_in", "signin",
    "cookie_consent", "gdpr",
    "two_factor", "2fa", "mfa", "otp",
    "payment", "checkout", "billing",
    "blocked", "access_denied", "bot_detection",
])
def test_hitl_keywords_in_frozenset(keyword: str) -> None:
    """Every documented HITL keyword must be in the frozenset."""
    assert keyword in HITL_PAGE_HINT_KEYWORDS, f"{keyword!r} not in HITL_PAGE_HINT_KEYWORDS"


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
        with patch("src.agent.hitl.httpx.AsyncClient", return_value=mock_client):
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
        with patch("src.agent.hitl.httpx.AsyncClient", return_value=mock_client):
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
        with patch("src.agent.hitl.httpx.AsyncClient", return_value=mock_client):
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


# ===========================================================================
# COMPLEX: HITL debounce inside PolicyRuleEngine
# ===========================================================================

def _engine() -> PolicyRuleEngine:
    return PolicyRuleEngine()


def _captcha_perception() -> ScreenPerception:
    return _perception("captcha_page")


def _normal_perception() -> ScreenPerception:
    return _perception("generic_page")


def test_hitl_debounce_first_hit_returns_none() -> None:
    """First HITL keyword match must NOT fire — debounce threshold is 2."""
    engine = _engine()
    state = _state("run-abc")
    perception = _captcha_perception()

    decision = engine.choose_action(state, perception, [])
    assert decision is None, "Should NOT fire HITL on the first consecutive match"


def test_hitl_debounce_second_consecutive_hit_fires() -> None:
    """Second consecutive HITL keyword match MUST fire WAIT_FOR_USER."""
    from src.models.policy import ActionType
    engine = _engine()
    state = _state("run-abc")
    perception = _captcha_perception()

    engine.choose_action(state, perception, [])  # first hit — no fire
    decision = engine.choose_action(state, perception, [])  # second hit — fire

    assert decision is not None, "Should fire HITL on second consecutive match"
    assert decision.action.action_type == ActionType.WAIT_FOR_USER


def test_hitl_debounce_resets_after_normal_page() -> None:
    """Counter resets when a non-HITL page appears between matches."""
    engine = _engine()
    state = _state("run-reset")
    captcha = _captcha_perception()
    normal = _normal_perception()

    engine.choose_action(state, captcha, [])  # hit 1
    engine.choose_action(state, normal, [])   # reset
    decision = engine.choose_action(state, captcha, [])  # hit 1 again

    assert decision is None, "Counter should reset; one match is not enough"


def test_hitl_debounce_tracked_per_run_id() -> None:
    """Debounce counters for different run IDs must not interfere."""
    from src.models.policy import ActionType
    engine = _engine()
    state_a = _state("run-aaa")
    state_b = _state("run-bbb")
    perception = _captcha_perception()

    # run-aaa: hit 1, then hit 2 → should fire
    engine.choose_action(state_a, perception, [])
    decision_a = engine.choose_action(state_a, perception, [])

    # run-bbb: only hit 1 → should NOT fire
    decision_b = engine.choose_action(state_b, perception, [])

    assert decision_a is not None and decision_a.action.action_type == ActionType.WAIT_FOR_USER
    assert decision_b is None


def test_hitl_fires_for_multiple_keyword_variants() -> None:
    """Each HITL keyword variant triggers the debounce path."""
    from src.models.policy import ActionType
    for keyword in ["login_wall", "recaptcha_v3", "2fa_code", "gdpr_banner"]:
        engine = _engine()
        state = _state(f"run-{keyword}")
        perc = ScreenPerception(
            summary=f"Page with {keyword}",
            page_hint=keyword,
            capture_artifact_path="runs/test/step_1/before.png",
            visible_elements=[],
        )
        engine.choose_action(state, perc, [])          # hit 1
        decision = engine.choose_action(state, perc, [])  # hit 2 → fire
        assert decision is not None, f"HITL should fire for keyword variant {keyword!r}"
        assert decision.action.action_type == ActionType.WAIT_FOR_USER
