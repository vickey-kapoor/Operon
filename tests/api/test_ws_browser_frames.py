"""/ws/stream browser frames: a parallel run streams its own executor, not the shared one."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from operon.api import ws_stream
from operon.core.runtime import current_runtime


class _Exec:
    def __init__(self, frame: bytes | None) -> None:
        self.frame = frame
        self.calls = 0

    async def live_frame_png(self, run_id: str) -> bytes | None:
        self.calls += 1
        return self.frame


async def _frames(monkeypatch: pytest.MonkeyPatch, *, shared, loop, desktop=None) -> list[bytes]:
    stream = current_runtime().stream
    monkeypatch.setattr(stream, "browser_executor", shared)
    monkeypatch.setattr(stream, "desktop_executor", desktop)
    monkeypatch.setattr(stream, "active_loop", loop)
    monkeypatch.setattr(stream, "active_run_id", "run-1")
    monkeypatch.setattr(stream, "browser_manager", None)
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    task = asyncio.create_task(ws_stream._frame_pusher(q))
    await asyncio.sleep(0.35)
    task.cancel()
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def test_parallel_run_streams_its_own_executor(monkeypatch) -> None:
    own, shared = _Exec(b"own"), _Exec(b"shared")
    frames = await _frames(monkeypatch, shared=shared, loop=SimpleNamespace(executor=own))
    assert frames and set(frames) == {b"own"}
    assert shared.calls == 0


async def test_no_active_loop_uses_shared_executor(monkeypatch) -> None:
    frames = await _frames(monkeypatch, shared=_Exec(b"shared"), loop=None)
    assert frames and set(frames) == {b"shared"}


async def test_desktop_active_loop_is_not_polled_as_browser(monkeypatch) -> None:
    desktop = _Exec(None)
    shared = _Exec(b"shared")
    frames = await _frames(monkeypatch, shared=shared, loop=SimpleNamespace(executor=desktop), desktop=desktop)
    assert set(frames) == {b"shared"}
    # the desktop executor is polled once per tick, by the desktop branch only
    assert desktop.calls == shared.calls
