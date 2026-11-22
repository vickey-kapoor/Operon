"""Settle detection: wait for the screen to stop changing, not for a fixed time (KAN-110).

After an action the screen is often still moving: a page loading, a menu sliding open, a
spinner. A fixed sleep either waits too long when the screen settled at once or captures
a half-loaded frame when it did not. This module watches a stream of small frames after
the action and returns when the screen has been quiet for a few frames in a row, or when
a cap runs out.

The frames are tiny grayscale signatures (like ``screen_diff``), so this is local and
cheap: no model call, no GPU, fine for a weak PC.

The loop is a small *frame stream*. Anything that wants the frames between steps can
subscribe as a :class:`FrameObserver`: the between-step change note (KAN-111) and a
later UI detector (KAN-118) plug in here without changing the executors.

Off by default: with ``OPERON_SETTLE_DETECT`` (and ``OPERON_CHANGE_NOTE``, which needs the
same frames) unset, :func:`settle_after_action` is exactly the old ``asyncio.sleep``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from operon.core.settings import get_settings

logger = logging.getLogger(__name__)

# Frame signatures are this size in grayscale: about 5 KB, enough to see a banner or a
# spinner, small enough to compare in well under a millisecond.
SIGNATURE_SIZE = (96, 54)

# A pixel counts as changed when its grayscale value moves by more than this.
PIXEL_DIFF = 24

# Fraction of changed pixels at or below which two frames count as the same. Matches
# ``screen_diff.SCREEN_CHANGE_THRESHOLD``: a blinking cursor stays under it.
QUIET_THRESHOLD = 0.002

# Frames to look at one after another, and how many quiet ones in a row mean "settled".
SAMPLE_INTERVAL_S = 0.1
QUIET_FRAMES = 2

# Never return sooner than this: an action's effect can start a moment after it ran, so
# the very first frames can look still when nothing has happened yet.
MIN_WAIT_S = 0.15

FrameGrab = Callable[[], Awaitable["bytes | None"]]


class FrameObserver(Protocol):
    """Receives each sampled frame signature while a settle wait runs."""

    def observe(self, index: int, signature: bytes, change: float | None) -> None:
        """``change`` is the changed-pixel fraction since the previous frame (None for the first)."""


@dataclass(frozen=True)
class SettleResult:
    settled: bool  # False: the cap ran out, or no frame could be grabbed
    waited_s: float
    frames: int
    max_change: float  # largest frame-to-frame change seen


def frame_signature(image_bytes: bytes) -> bytes | None:
    """Reduce an encoded image (PNG/JPEG bytes) to a small grayscale signature."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("L").resize(SIGNATURE_SIZE, Image.Resampling.BILINEAR).tobytes()
    except Exception:  # noqa: BLE001 - a bad frame must never break an action
        logger.debug("settle: could not read frame", exc_info=True)
        return None


def signature_from_rgb(rgb: bytes, size: tuple[int, int]) -> bytes | None:
    """Same as :func:`frame_signature` for raw RGB pixels (what ``mss`` returns)."""
    try:
        with Image.frombytes("RGB", size, rgb) as img:
            return img.convert("L").resize(SIGNATURE_SIZE, Image.Resampling.BILINEAR).tobytes()
    except Exception:  # noqa: BLE001
        logger.debug("settle: could not read raw frame", exc_info=True)
        return None


def change_ratio(a: bytes, b: bytes) -> float:
    """Fraction (0..1) of pixels that differ by more than ``PIXEL_DIFF``."""
    if not a or len(a) != len(b):
        return 1.0
    changed = sum(1 for x, y in zip(a, b) if abs(x - y) > PIXEL_DIFF)
    return changed / len(a)


async def _sleep(seconds: float) -> None:
    """The default wait (a hook: the free replay swaps it for a fake clock, KAN-178)."""
    await asyncio.sleep(seconds)


def _clock() -> float:
    """The default clock (a hook, like :func:`_sleep`)."""
    return time.monotonic()


async def wait_until_settled(
    grab: FrameGrab,
    *,
    min_s: float = MIN_WAIT_S,
    max_s: float = 3.0,
    fail_wait_s: float | None = None,
    interval_s: float = SAMPLE_INTERVAL_S,
    quiet_frames: int = QUIET_FRAMES,
    threshold: float = QUIET_THRESHOLD,
    observers: Sequence[FrameObserver] = (),
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> SettleResult:
    """Sample frames until the screen is quiet for ``quiet_frames`` in a row.

    Returns after at least ``min_s`` and at most ``max_s``. If a frame cannot be grabbed
    (it fails, or takes longer than the time left) the wait falls back to a plain sleep
    of ``fail_wait_s`` in total (default ``min_s``): with no evidence the screen settled,
    the caller's old fixed delay is the safe wait. A broken grab never fails an action,
    and no grab can hold one past ``max_s``. ``sleep`` and ``clock`` default to this
    module's ``_sleep`` and ``_clock``, looked up at call time so a replay can patch them.
    """
    sleep = sleep or _sleep
    clock = clock or _clock
    start = clock()
    fail_wait = min_s if fail_wait_s is None else fail_wait_s

    async def fall_back(frames: int, max_change: float) -> SettleResult:
        remaining = fail_wait - (clock() - start)
        if remaining > 0:
            await sleep(remaining)
        return SettleResult(False, clock() - start, frames, max_change)

    previous = await _bounded_grab(grab, max_s)
    if previous is None:
        return await fall_back(0, 0.0)

    frames = 1
    quiet = 0
    max_change = 0.0
    for observer in observers:
        _notify(observer, 0, previous, None)

    while True:
        elapsed = clock() - start
        if elapsed >= max_s:
            return SettleResult(False, elapsed, frames, max_change)
        await sleep(min(interval_s, max(0.0, max_s - elapsed)))
        current = await _bounded_grab(grab, max(0.0, max_s - (clock() - start)))
        if current is None:
            return await fall_back(frames, max_change)
        change = change_ratio(previous, current)
        max_change = max(max_change, change)
        for observer in observers:
            _notify(observer, frames, current, change)
        frames += 1
        quiet = quiet + 1 if change <= threshold else 0
        previous = current
        elapsed = clock() - start
        if quiet >= quiet_frames and elapsed >= min_s:
            return SettleResult(True, elapsed, frames, max_change)


async def _bounded_grab(grab: FrameGrab, timeout_s: float) -> bytes | None:
    """One grab, given up on (None) if it fails or outlasts ``timeout_s``."""
    try:
        return await asyncio.wait_for(grab(), timeout=max(timeout_s, 0.05))
    except Exception:  # noqa: BLE001 - includes TimeoutError: a slow grab counts as a failed one
        logger.debug("settle: frame grab failed or timed out", exc_info=True)
        return None


def _notify(observer: FrameObserver, index: int, signature: bytes, change: float | None) -> None:
    try:
        observer.observe(index, signature, change)
    except Exception:  # noqa: BLE001 - an observer must never stall an action
        logger.debug("settle: observer failed", exc_info=True)


async def settle_after_action(
    grab: FrameGrab,
    fallback_delay: float,
    *,
    observers: Sequence[FrameObserver] = (),
) -> SettleResult | None:
    """The post-action wait: settle detection when it is on, else the old fixed sleep.

    ``fallback_delay`` is what the executor slept before. With the flag off that is all
    this does. With it on, the wait ends as soon as the screen is quiet (but not before
    ``MIN_WAIT_S``, or ``fallback_delay`` if that is shorter) and runs for up to
    ``OPERON_SETTLE_MAX_S``.
    """
    settings = get_settings()
    if not (settings.settle_detect or settings.change_note):
        await asyncio.sleep(fallback_delay)
        return None
    result = await wait_until_settled(
        grab,
        min_s=min(fallback_delay, MIN_WAIT_S),
        max_s=max(settings.settle_max_s, fallback_delay),
        fail_wait_s=fallback_delay,
        observers=observers,
    )
    logger.debug(
        "settle: %s after %.2fs (%d frames, max change %.3f; fixed delay was %.2fs)",
        "settled" if result.settled else "capped",
        result.waited_s,
        result.frames,
        result.max_change,
        fallback_delay,
    )
    return result
