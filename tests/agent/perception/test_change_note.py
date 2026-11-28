"""Between-step change note (KAN-111): synthetic frames, no screen."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from operon.agent.perception.change_note import ChangeNoteBuffer, ChangeWatcher
from operon.agent.perception.settle import SIGNATURE_SIZE

W, H = SIGNATURE_SIZE


def frame(box=None, value=255, base=20) -> bytes:
    """A dark frame, with an optional bright box (x0, y0, x1, y1) in signature pixels."""
    px = bytearray([base]) * (W * H)
    if box:
        x0, y0, x1, y1 = box
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[y * W + x] = value
    return bytes(px)


TOP_RIGHT = (72, 0, 96, 18)


def run(frames: list[bytes]) -> str | None:
    watcher = ChangeWatcher()
    for i, f in enumerate(frames):
        watcher.observe(i, f, None)
    return watcher.note()


def test_still_screen_is_quiet():
    assert run([frame()] * 5) is None


def test_change_that_stays_is_not_reported():
    assert run([frame(), frame(), frame(TOP_RIGHT), frame(TOP_RIGHT), frame(TOP_RIGHT)]) is None


def test_toast_that_appears_and_goes_is_reported_with_position():
    note = run([frame(), frame(TOP_RIGHT), frame(TOP_RIGHT), frame(), frame(), frame()])
    assert note is not None
    assert "top right" in note
    assert "gone again" in note


def test_region_still_changing_at_the_end_is_reported():
    on, off = frame(TOP_RIGHT), frame()
    note = run([off, on, off, on, off, on])
    assert note is not None
    assert "still changing" in note


def test_whole_screen_activity_is_one_phrase():
    note = run([frame(base=0), frame(base=200), frame(base=0), frame(base=200)])
    assert note is not None and "most of the screen" in note


def test_note_is_capped():
    note = run([frame(), frame(TOP_RIGHT), frame((0, 36, 24, 54)), frame(), frame()])
    assert note is not None and len(note) <= 300


def test_single_frame_gives_nothing():
    assert run([frame()]) is None


def _settings(change_note: bool, settle: bool = False):
    s = MagicMock()
    s.change_note = change_note
    s.settle_detect = settle
    s.settle_max_s = 3.0
    return s


def test_buffer_off_collects_nothing(monkeypatch):
    monkeypatch.setattr("operon.agent.perception.change_note.get_settings", lambda: _settings(False))
    monkeypatch.setattr("operon.agent.perception.settle.get_settings", lambda: _settings(False))
    calls = []

    async def grab():
        calls.append(1)
        return frame()

    buf = ChangeNoteBuffer()
    asyncio.run(buf.settle(grab, 0.0))
    assert not calls and buf.pop() is None


def test_buffer_on_collects_and_pops_once(monkeypatch):
    monkeypatch.setattr("operon.agent.perception.change_note.get_settings", lambda: _settings(True))
    monkeypatch.setattr("operon.agent.perception.settle.get_settings", lambda: _settings(True))
    seq = iter([frame(), frame(TOP_RIGHT), frame(), frame(), frame(), frame(), frame()])

    async def grab():
        return next(seq, frame())

    buf = ChangeNoteBuffer()
    asyncio.run(buf.settle(grab, 0.0))
    note = buf.pop()
    assert note and "top right" in note
    assert buf.pop() is None


def test_buffer_keeps_only_newest_notes(monkeypatch):
    monkeypatch.setattr("operon.agent.perception.change_note.get_settings", lambda: _settings(True))
    monkeypatch.setattr("operon.agent.perception.settle.get_settings", lambda: _settings(True))

    async def many():
        buf = ChangeNoteBuffer()
        for _ in range(6):
            seq = iter([frame(), frame(TOP_RIGHT), frame(), frame(), frame(), frame()])

            async def grab():
                return next(seq, frame())

            await buf.settle(grab, 0.0)
        return buf.pop()

    note = asyncio.run(many())
    assert note is not None and note.count("Screen activity") == 3


def test_settle_forwards_min_s_to_settle_after_action(monkeypatch) -> None:
    """KAN-264: min_s reaches settle_after_action with change_note on and off."""
    seen: list[dict] = []

    async def fake(grab, fallback_delay, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr("operon.agent.perception.change_note.settle_after_action", fake)
    for on in (False, True):
        monkeypatch.setattr("operon.agent.perception.change_note.get_settings", lambda on=on: _settings(on))
        asyncio.run(ChangeNoteBuffer().settle(MagicMock(), 2.5, min_s=1.0))
    assert [k["min_s"] for k in seen] == [1.0, 1.0]
