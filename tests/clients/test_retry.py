"""Shared retry schedule: Retry-After parsing, Google retryDelay, backoff rules (offline, no sleeping)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from operon.clients import retry
from operon.clients.retry import (
    RATE_LIMIT_RETRIES,
    RETRY_AFTER_CAP_SECONDS,
    RetrySchedule,
    google_retry_delay,
    parse_retry_after,
    pause,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _sched(**kwargs) -> RetrySchedule:
    return RetrySchedule(rng=lambda: 0.0, **kwargs)


def test_parse_retry_after_seconds() -> None:
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 1.5 ") == 1.5


def test_parse_retry_after_http_date() -> None:
    assert parse_retry_after("Thu, 01 Jan 2026 12:00:30 GMT", now=NOW) == pytest.approx(30.0)
    assert parse_retry_after("Thu, 01 Jan 2026 11:00:00 GMT", now=NOW) == 0.0


@pytest.mark.parametrize("value", [None, "", "   ", "soon", "-3", "nan"])
def test_parse_retry_after_garbage(value) -> None:
    assert parse_retry_after(value) is None


def test_google_retry_delay() -> None:
    def body(delay):
        return json.dumps({"error": {"details": [{"@type": "x"}, {"retryDelay": delay}]}})

    assert google_retry_delay(body("17s")) == 17.0
    assert google_retry_delay(body("1.5s")) == 1.5
    assert google_retry_delay(body("soon")) is None
    assert google_retry_delay("not json") is None
    assert google_retry_delay(json.dumps({"error": {}})) is None
    assert google_retry_delay(json.dumps([1])) is None


def test_5xx_sequence_is_exponential_then_gives_up() -> None:
    s = _sched(max_retries=3, backoff_seconds=1.0)
    assert [s.next_delay(status=503) for _ in range(4)] == [1.0, 2.0, 4.0, None]


def test_timeouts_use_the_non_rate_limit_counter() -> None:
    s = _sched(max_retries=1, backoff_seconds=1.0)
    assert s.next_delay() == 1.0
    assert s.next_delay() is None


def test_429_sequence_is_longer_and_has_its_own_counter() -> None:
    s = _sched(max_retries=3, backoff_seconds=1.0)
    delays = [s.next_delay(status=429) for _ in range(RATE_LIMIT_RETRIES + 1)]
    assert delays == [4.0, 8.0, 16.0, 30.0, 30.0, None]  # capped at MAX_BACKOFF_SECONDS
    s2 = _sched(max_retries=2, backoff_seconds=1.0)
    s2.next_delay(status=429)
    assert s2.next_delay(status=500) == 1.0  # the 5xx counter is untouched by the 429


def test_jitter_scales_schedule_delays_but_not_hints() -> None:
    s = RetrySchedule(max_retries=2, backoff_seconds=1.0, rng=lambda: 1.0)
    assert s.next_delay(status=503) == pytest.approx(1.25)
    assert s.next_delay(status=503, retry_after="3") == 3.0


def test_retry_after_is_capped_and_preferred_over_server_delay() -> None:
    s = _sched(max_retries=2, backoff_seconds=1.0)
    assert s.next_delay(status=429, retry_after="500") == RETRY_AFTER_CAP_SECONDS
    assert s.next_delay(status=429, retry_after="2", server_delay=9.0) == 2.0
    assert s.next_delay(status=429, retry_after="junk", server_delay=9.0) == 9.0


def test_zero_max_retries_never_retries_even_429() -> None:
    s = _sched(max_retries=0, backoff_seconds=1.0)
    assert s.next_delay(status=429, retry_after="1") is None
    assert s.next_delay(status=500) is None
    assert s.next_delay() is None


def test_budget_exhaustion_gives_up() -> None:
    s = _sched(max_retries=5, backoff_seconds=1.0)
    assert s.next_delay(status=429, retry_after="60") == 60.0
    assert s.next_delay(status=429, retry_after="60") == 60.0
    assert s.next_delay(status=429, retry_after="1") is None  # 121s > 120s budget


def test_zero_base_stays_zero() -> None:
    s = _sched(max_retries=2, backoff_seconds=0.0)
    assert s.next_delay(status=503) == 0.0
    assert s.next_delay(status=429) == 0.0


@pytest.mark.asyncio
async def test_pause_skips_non_positive(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", fake_sleep)
    await pause(0.0)
    await pause(2.0)
    assert slept == [2.0]
