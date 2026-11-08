"""Lifespan shutdown wiring: built executors get a uniform aclose() teardown."""

import asyncio
from types import SimpleNamespace

from operon.api.runtime import loops
from operon.api.server import _lifespan, app


def _drive_lifespan() -> None:
    async def _run() -> None:
        async with _lifespan(app):
            pass

    asyncio.run(_run())


def test_lifespan_acloses_built_executors_on_shutdown(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")  # skip retention sweep
    calls: list[str] = []

    class _FakeExecutor:
        def __init__(self, tag: str) -> None:
            self._tag = tag

        async def aclose(self) -> None:
            calls.append(self._tag)

    # Stand in two "built" agent loops (browser + desktop), each exposing .executor.
    monkeypatch.setattr(loops, "_agent_loop", SimpleNamespace(executor=_FakeExecutor("browser")))
    monkeypatch.setattr(loops, "_desktop_agent_loop", SimpleNamespace(executor=_FakeExecutor("desktop")))

    _drive_lifespan()

    assert calls == ["browser", "desktop"]


def test_lifespan_skips_unbuilt_loops_and_missing_aclose(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")
    calls: list[str] = []

    class _LegacyExecutor:
        """No aclose() — shutdown must skip it without raising."""

    # Browser loop never built (None); desktop executor lacks aclose().
    monkeypatch.setattr(loops, "_agent_loop", None)
    monkeypatch.setattr(loops, "_desktop_agent_loop", SimpleNamespace(executor=_LegacyExecutor()))

    _drive_lifespan()  # must not raise

    assert calls == []


def test_lifespan_swallows_aclose_errors(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")
    reached: list[str] = []

    class _BoomExecutor:
        async def aclose(self) -> None:
            raise RuntimeError("teardown blew up")

    class _OkExecutor:
        async def aclose(self) -> None:
            reached.append("ok")

    # A failing teardown must not prevent the next executor from closing.
    monkeypatch.setattr(loops, "_agent_loop", SimpleNamespace(executor=_BoomExecutor()))
    monkeypatch.setattr(loops, "_desktop_agent_loop", SimpleNamespace(executor=_OkExecutor()))

    _drive_lifespan()  # must not raise

    assert reached == ["ok"]
