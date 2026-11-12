"""The task page's live view of a desktop (KAN-108): /ws/stream sends the desktop
executor's frames for the run on it, and the run status says the page can drive it."""

from __future__ import annotations

import asyncio

import pytest

from operon.api import frontdoor, ws_stream


class _Desktop:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def live_frame_png(self, run_id: str) -> bytes | None:
        self.asked.append(run_id)
        return b"\xff\xd8jpeg" if run_id == "desk-1" else None


async def _frames_for(monkeypatch: pytest.MonkeyPatch, run_id: str) -> list[bytes]:
    desktop = _Desktop()
    monkeypatch.setattr(ws_stream, "_desktop_executor", desktop)
    monkeypatch.setattr(ws_stream, "_active_run_id", run_id)
    monkeypatch.setattr(ws_stream, "_browser_manager", object())  # a browser screencast is also live
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    task = asyncio.create_task(ws_stream._frame_pusher(q))
    await asyncio.sleep(0.35)
    task.cancel()
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    return frames


async def test_the_desktop_run_streams_its_screen(monkeypatch) -> None:
    frames = await _frames_for(monkeypatch, "desk-1")
    assert frames and all(frame == b"\xff\xd8jpeg" for frame in frames)


async def test_other_runs_get_no_desktop_frames(monkeypatch) -> None:
    assert await _frames_for(monkeypatch, "browser-run") == []


def test_desktop_executor_is_set_by_its_setter(monkeypatch) -> None:
    monkeypatch.setattr(ws_stream, "_desktop_executor", None)
    marker = object()
    ws_stream.set_desktop_executor(marker)
    assert ws_stream._desktop_executor is marker


async def test_status_marks_a_streamable_desktop_as_live(monkeypatch) -> None:
    from types import SimpleNamespace

    from operon.models.common import RunStatus

    record = SimpleNamespace(
        run_id="desk-1", intent="x", status=RunStatus.RUNNING, step_count=0, max_steps=5,
        permission_mode=None, in_takeover=False,
    )

    async def require(run_id):
        return SimpleNamespace(executor=_Desktop()), record

    monkeypatch.setattr(frontdoor, "_require_run", require)
    monkeypatch.setattr(frontdoor, "_steps", lambda loop, run_id: [])
    monkeypatch.setattr(frontdoor, "_waiting", lambda loop, record: None)
    monkeypatch.setattr(frontdoor, "_environment_of", lambda run_id: "desktop")
    assert (await frontdoor.run_status("desk-1"))["live_view"] is True

    async def require_windows(run_id):
        return SimpleNamespace(executor=object()), record

    monkeypatch.setattr(frontdoor, "_require_run", require_windows)
    assert (await frontdoor.run_status("desk-1"))["live_view"] is False
