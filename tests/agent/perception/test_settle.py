"""Settle detection (KAN-110): wait for the screen to stop changing, not a fixed time."""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from operon.agent.perception import settle
from operon.agent.perception.settle import (
    SIGNATURE_SIZE,
    change_ratio,
    frame_signature,
    settle_after_action,
    signature_from_rgb,
    wait_until_settled,
)
from operon.core.settings import Settings

_PIXELS = SIGNATURE_SIZE[0] * SIGNATURE_SIZE[1]
_STILL = bytes([100]) * _PIXELS
_BRIGHT = bytes([220]) * _PIXELS


class FakeClock:
    """A clock the fake sleep advances, so the tests take no real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _grabber(frames: list[bytes | None]):
    """A grab() that plays ``frames`` in order, then repeats the last one."""
    state = {"i": 0}

    async def grab() -> bytes | None:
        frame = frames[min(state["i"], len(frames) - 1)]
        state["i"] += 1
        return frame

    return grab


async def _settle(frames, clock: FakeClock, **kwargs):
    return await wait_until_settled(
        _grabber(frames), sleep=clock.sleep, clock=clock.time, **kwargs
    )


def test_change_ratio() -> None:
    assert change_ratio(_STILL, _STILL) == 0.0
    assert change_ratio(_STILL, _BRIGHT) == 1.0
    half = _STILL[: _PIXELS // 2] + _BRIGHT[_PIXELS // 2 :]
    assert change_ratio(_STILL, half) == pytest.approx(0.5, abs=0.01)
    assert change_ratio(_STILL, b"short") == 1.0  # different sizes: treat as changed


def test_signatures_from_png_and_raw_rgb() -> None:
    img = Image.new("RGB", (320, 200), (10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    from_png = frame_signature(buf.getvalue())
    from_raw = signature_from_rgb(img.tobytes(), img.size)
    assert from_png is not None and len(from_png) == _PIXELS
    assert from_raw == from_png
    assert frame_signature(b"not an image") is None


async def test_a_still_screen_settles_as_soon_as_the_floor_passes() -> None:
    clock = FakeClock()
    result = await _settle([_STILL], clock, max_s=3.0)
    assert result.settled
    assert result.waited_s == pytest.approx(0.2)  # two quiet frames at 0.1 s
    assert result.waited_s < 3.0


async def test_the_floor_is_kept_even_when_the_screen_is_quiet() -> None:
    clock = FakeClock()
    result = await _settle([_STILL], clock, min_s=0.5, max_s=3.0)
    assert result.settled
    assert result.waited_s >= 0.5


async def test_a_moving_screen_is_waited_for_until_it_stops() -> None:
    clock = FakeClock()
    frames = [_STILL, _BRIGHT, _STILL, _BRIGHT, _STILL]  # five changes, then quiet
    result = await _settle(frames, clock, max_s=3.0)
    assert result.settled
    assert result.max_change == 1.0
    assert result.waited_s > 0.4  # it did not stop at the first still-looking frame


async def test_a_screen_that_never_stops_hits_the_cap() -> None:
    clock = FakeClock()
    frames = [_STILL, _BRIGHT] * 100
    result = await _settle(frames, clock, max_s=1.0)
    assert not result.settled
    assert 1.0 <= result.waited_s <= 1.1


async def test_a_failed_first_grab_falls_back_to_the_plain_floor() -> None:
    clock = FakeClock()
    result = await _settle([None], clock, min_s=0.15)
    assert not result.settled and result.frames == 0
    assert clock.sleeps == [0.15]


async def test_a_failed_grab_waits_out_the_callers_old_delay() -> None:
    """No evidence the screen settled: the old fixed delay is the safe wait (not just the floor)."""
    clock = FakeClock()
    result = await _settle([None], clock, min_s=0.15, fail_wait_s=1.0)
    assert not result.settled
    assert result.waited_s == pytest.approx(1.0)

    clock = FakeClock()
    result = await _settle([_STILL, None], clock, min_s=0.15, fail_wait_s=1.0)
    assert not result.settled
    assert result.waited_s == pytest.approx(1.0)  # 0.1 s sampling, then the rest of the 1.0 s


async def test_a_stalled_grab_cannot_hold_the_action_past_the_cap() -> None:
    clock = FakeClock()

    async def stalled() -> bytes | None:
        await asyncio.sleep(30)
        return _STILL

    import time as _time

    started = _time.monotonic()
    result = await wait_until_settled(stalled, max_s=0.2, sleep=clock.sleep, clock=clock.time)
    assert not result.settled and result.frames == 0
    assert _time.monotonic() - started < 2.0  # gave up after the cap, not after 30 s


async def test_a_grab_that_fails_midway_still_waits_out_the_floor() -> None:
    clock = FakeClock()
    result = await _settle([_STILL, None], clock, min_s=0.5)
    assert not result.settled
    assert result.waited_s >= 0.5


async def test_observers_see_every_frame_and_cannot_break_the_wait() -> None:
    clock = FakeClock()
    seen: list[tuple[int, float | None]] = []

    class Recorder:
        def observe(self, index: int, signature: bytes, change: float | None) -> None:
            seen.append((index, change))

    class Broken:
        def observe(self, index: int, signature: bytes, change: float | None) -> None:
            raise RuntimeError("observer bug")

    result = await _settle([_STILL, _BRIGHT, _STILL], clock, observers=[Broken(), Recorder()])
    assert result.settled
    assert seen[0] == (0, None)  # the first frame has nothing to compare with
    assert [i for i, _ in seen] == list(range(len(seen)))
    assert seen[1][1] == 1.0


async def test_flag_off_is_exactly_the_old_sleep(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def grab() -> bytes | None:
        raise AssertionError("no frame may be grabbed with the flag off")

    monkeypatch.delenv("OPERON_SETTLE_DETECT", raising=False)
    monkeypatch.setattr(settle.asyncio, "sleep", fake_sleep)
    assert await settle_after_action(grab, 0.3) is None
    assert slept == [0.3]


async def test_flag_on_returns_early_on_a_still_screen(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SETTLE_DETECT", "on")
    monkeypatch.setenv("OPERON_SETTLE_MAX_S", "3")
    monkeypatch.setattr(settle, "SAMPLE_INTERVAL_S", 0.01)

    async def grab() -> bytes | None:
        return _STILL

    real_wait = settle.wait_until_settled

    async def quick_wait(grab, **kwargs):
        return await real_wait(grab, interval_s=0.01, **kwargs)

    monkeypatch.setattr(settle, "wait_until_settled", quick_wait)
    result = await asyncio.wait_for(settle_after_action(grab, 1.0), timeout=2.0)
    assert result is not None and result.settled
    assert result.waited_s < 1.0  # the old fixed delay was 1.0 s


def test_settings_defaults_and_clamps() -> None:
    off = Settings.from_env({})
    assert off.settle_detect is False and off.settle_max_s == 3.0
    on = Settings.from_env({"OPERON_SETTLE_DETECT": "on", "OPERON_SETTLE_MAX_S": "99"})
    assert on.settle_detect is True and on.settle_max_s == 15.0
    assert Settings.from_env({"OPERON_SETTLE_MAX_S": "0"}).settle_max_s == 0.5
