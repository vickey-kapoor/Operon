"""Lifespan shutdown wiring: the persistent observable browser is torn down."""

import asyncio

from operon.api import ws_stream
from operon.api.server import _lifespan, app


def _drive_lifespan() -> None:
    async def _run() -> None:
        async with _lifespan(app):
            pass

    asyncio.run(_run())


def test_lifespan_closes_persistent_browser_on_shutdown(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")  # skip retention sweep
    calls: list[str] = []

    class _FakeBrowserExecutor:
        async def close_persistent_browser(self) -> None:
            calls.append("closed")

    prev = ws_stream.get_executor()
    ws_stream.set_executor(_FakeBrowserExecutor())
    try:
        _drive_lifespan()
    finally:
        ws_stream.set_executor(prev)

    assert calls == ["closed"]


def test_lifespan_skips_executor_without_close(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")

    class _DesktopLikeExecutor:
        """No close_persistent_browser — shutdown must skip it without raising."""

    prev = ws_stream.get_executor()
    ws_stream.set_executor(_DesktopLikeExecutor())
    try:
        _drive_lifespan()  # must not raise
    finally:
        ws_stream.set_executor(prev)
