"""Shared retry schedule for model HTTP calls (KAN-173): honors Retry-After, backs off longer on 429."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_FACTOR = 4.0
MAX_BACKOFF_SECONDS = 30.0
RETRY_AFTER_CAP_SECONDS = 60.0
RETRY_BUDGET_SECONDS = 120.0
JITTER = 0.25


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds, or None."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return max(0.0, (when - current).total_seconds())
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return None
    return seconds


def google_retry_delay(body_text: str) -> float | None:
    """Read error.details[*].retryDelay (like "17s") from a Gemini error body, or None."""
    try:
        details = json.loads(body_text)["error"]["details"]
        for item in details:
            raw = item.get("retryDelay") if isinstance(item, dict) else None
            if isinstance(raw, str) and raw.endswith("s"):
                seconds = float(raw[:-1])
                if seconds >= 0 and seconds != float("inf"):
                    return seconds
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return None


class RetrySchedule:
    """Per-request retry state: separate 429 and other counters, a total sleep budget."""

    def __init__(
        self,
        *,
        max_retries: int,
        backoff_seconds: float,
        rate_limit_retries: int | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        if rate_limit_retries is None:
            rate_limit_retries = 0 if max_retries == 0 else max(max_retries, RATE_LIMIT_RETRIES)
        self.rate_limit_retries = rate_limit_retries
        self._rng: Callable[[], float] = rng or (lambda: random.random())
        self._rate_limit_used = 0
        self._other_used = 0
        self._waited = 0.0

    def next_delay(
        self,
        *,
        status: int | None = None,
        retry_after: str | None = None,
        server_delay: float | None = None,
    ) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        is_rate_limit = status == 429
        if is_rate_limit:
            if self._rate_limit_used >= self.rate_limit_retries:
                return None
            n = self._rate_limit_used + 1
        else:
            if self._other_used >= self.max_retries:
                return None
            n = self._other_used + 1
        hint = parse_retry_after(retry_after)
        if hint is None:
            hint = server_delay
        if hint is not None:
            delay = min(hint, RETRY_AFTER_CAP_SECONDS)
        else:
            factor = RATE_LIMIT_BACKOFF_FACTOR if is_rate_limit else 1.0
            delay = min(self.backoff_seconds * factor * 2 ** (n - 1), MAX_BACKOFF_SECONDS)
            delay *= 1 + JITTER * self._rng()
        if self._waited + delay > RETRY_BUDGET_SECONDS:
            return None
        if is_rate_limit:
            self._rate_limit_used += 1
        else:
            self._other_used += 1
        self._waited += delay
        return delay


async def pause(seconds: float) -> None:
    """Sleep for `seconds`, skipping the sleep when it is not positive."""
    if seconds > 0:
        await asyncio.sleep(seconds)
