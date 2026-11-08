"""Opening a run's browser waits for the previous run's browser to finish closing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from operon.executor.browser_native import NativeBrowserExecutor


async def test_a_new_session_waits_for_the_previous_one_to_close(tmp_path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True, record_video=False)
    events: list[str] = []

    async def slow_close(session, *, observable=False):
        events.append("close:start")
        await asyncio.sleep(0.05)
        events.append("close:end")

    async def fake_open(run_id, *, foreground=True):
        events.append(f"open:{run_id}")
        session = SimpleNamespace(page=object())
        executor._sessions[run_id] = session
        return session

    executor._close_session = slow_close
    executor._ensure_session_batch = fake_open
    executor._sessions["A"] = SimpleNamespace(page=object())

    closing = asyncio.create_task(executor.aclose_run("A"))
    await asyncio.sleep(0)
    await executor._ensure_session("B")
    await closing

    assert events == ["close:start", "close:end", "open:B"]
