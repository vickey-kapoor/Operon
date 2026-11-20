"""Lazy AgentLoop construction for API routes."""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

from operon.agent.backends.model_adapter.backend import GAVE_UP_PREFIX
from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
from operon.agent.tools import enabled_tools
from operon.api import ws_stream as _ws_stream
from operon.api.runtime.services import build_browser_services, build_desktop_services
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.core.contracts.perception import Environment as UnifiedEnvironment
from operon.core.runtime import current_runtime
from operon.core.settings import get_settings
from operon.environments import EnvironmentProvider, get_provider
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

# The loops and the desktop provider live on the process runtime (KAN-156).
NativeBrowserExecutor = None
logger = logging.getLogger(__name__)


def iter_built_executors():
    """Yield the executor of each agent loop that has actually been built.

    Reads the runtime's fields directly so it never triggers lazy construction
    (which would launch a browser). Used by the server lifespan to tear down
    executor-held resources at shutdown.
    """
    runtime = current_runtime()
    for loop in (runtime.browser_loop, runtime.desktop_loop, *runtime.run_loops.values()):
        if loop is not None:
            yield getattr(loop, "raw_executor", loop.executor)


async def close_built_runtime() -> None:
    """Release what the built loops hold: each executor's ``aclose()``, then the
    desktop environment provider's ``stop()`` (KAN-80). Never raises."""
    for executor in iter_built_executors():
        aclose = getattr(executor, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown cleanup must not raise
            logger.warning("executor shutdown cleanup skipped: %s", exc)
    provider = current_runtime().desktop_provider
    if provider is not None:
        try:
            await asyncio.to_thread(provider.stop)
        except Exception as exc:  # noqa: BLE001 - shutdown cleanup must not raise
            logger.warning("environment %r stop skipped: %s", provider.name, exc)


def desktop_provider() -> EnvironmentProvider:
    """The desktop run environment (``OPERON_ENVIRONMENT``), built and started once."""
    runtime = current_runtime()
    if runtime.desktop_provider is None:
        provider = get_provider()
        check = provider.check()
        if not check.ok:
            raise RuntimeError(
                f"Environment {provider.name!r} cannot run here: " + " ".join(check.problems)
            )
        provider.start()
        runtime.desktop_provider = provider
    return runtime.desktop_provider


def _build_browser_executor():
    global NativeBrowserExecutor
    if NativeBrowserExecutor is None:
        NativeBrowserExecutor = importlib.import_module("operon.executor.browser_native").NativeBrowserExecutor
    return NativeBrowserExecutor()


def _with_tools(executor, config):
    """The executor plus Operon's tools when ``OPERON_TOOLS`` turns them on (RFC 0002).
    Only the Gemini computer-use backend declares tools, so only its runs get them."""
    tools = enabled_tools()
    if not tools or config.backend != "computer_use":
        return executor
    from operon.executor.tools import ToolExecutor

    return ToolExecutor(executor, tools)


def _loop_layers(*, services) -> dict:
    """Policy, verifier and recovery: the model's action is never replaced, retried or reset."""
    return {
        "policy_service": ThinPolicy(services.policy_delegate),
        "verifier_service": OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX),
        "recovery_manager": ThinRecovery(),
    }


def _new_browser_loop(*, run_store=None, memory_store=None, loop_cls=AgentLoop):
    """One browser AgentLoop with its own executor, backends and policy; returns
    ``(loop, raw executor)``. Stores are given when several loops share them."""
    browser_config = browser_mode_config()
    executor = _build_browser_executor()
    services = build_browser_services(executor)
    run_store = run_store or FileBackedRunStore(environment="browser")
    memory_store = memory_store or FileBackedMemoryStore()
    loop = loop_cls(
        capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
        perception_service=services.perception_service,
        run_store=run_store,
        executor=_with_tools(executor, browser_config),
        memory_store=memory_store,
        environment=UnifiedEnvironment.BROWSER,
        **_loop_layers(services=services),
    )
    return loop, executor


def get_agent_loop() -> AgentLoop:
    """Build the browser runtime lazily so env loading can happen first."""
    runtime = current_runtime()
    if runtime.browser_loop is None:
        loop, executor = _new_browser_loop()
        runtime.browser_loop = loop
        _ws_stream.set_executor(executor)
        _ws_stream.set_agent_loop(loop)
    return runtime.browser_loop


# --- Parallel batch runs (KAN-81, OPERON_PARALLEL_RUNS) -------------------------------


class RunSlotsFull(Exception):
    """Every parallel run slot is taken; the caller answers 429 and the client retries."""


def parallel_runs() -> int:
    return max(1, get_settings().parallel_runs)


def runs_in_parallel(mode: str | None) -> bool:
    """Whether a browser run of ``mode`` gets its own loop. Only batch mode does:
    observable mode owns the one CDP port (9222) and the live view."""
    return parallel_runs() > 1 and (mode or "batch") == "batch"


def loop_for_run(run_id: str):
    """The loop of a parallel run, or None when ``run_id`` runs on the shared loop."""
    return current_runtime().run_loops.get(run_id)


class ParallelRunLoop(AgentLoop):
    """A loop that serves one parallel batch run: it owns the run's executor and a
    slot, and gives both back when the run ends (terminal status, cancel or crash)."""

    raw_executor: Any = None
    _slot_held = False
    _released = False

    async def release_parallel_run(self, run_id: str | None = None) -> None:
        """Close this loop's executor and free its slot. Idempotent."""
        if self._released:
            return
        self._released = True
        runtime = current_runtime()
        for key in [k for k, v in runtime.run_loops.items() if v is self]:
            runtime.run_loops.pop(key, None)
        aclose = getattr(self.raw_executor, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as exc:  # noqa: BLE001 - cleanup must not raise
                logger.warning("parallel run executor cleanup skipped: %s", exc)
        if self._slot_held:
            self._slot_held = False
            if runtime.run_slots is not None:
                runtime.run_slots.release()

    async def _cleanup_completed_run(self, run_id: str) -> None:
        try:
            await super()._cleanup_completed_run(run_id)
        finally:
            await self.release_parallel_run(run_id)


async def release_parallel_run(loop, run_id: str) -> None:
    """Free a parallel run's executor and slot after a crash path that skips the
    loop's own cleanup. A no-op for the shared loop."""
    if isinstance(loop, ParallelRunLoop):
        await loop.release_parallel_run(run_id)


async def start_parallel_browser_run(request):
    """Start a batch run on its own loop. Returns ``(loop, response)``; raises
    ``RunSlotsFull`` when ``OPERON_PARALLEL_RUNS`` runs already hold a slot."""
    runtime = current_runtime()
    if runtime.run_slots is None:
        runtime.run_slots = asyncio.Semaphore(parallel_runs())
    slots = runtime.run_slots
    if slots.locked():
        raise RunSlotsFull(f"{parallel_runs()} parallel runs are already active; retry when one finishes")
    await slots.acquire()  # not locked, so this does not wait
    base = get_agent_loop()
    loop = None
    try:
        loop, executor = _new_browser_loop(
            run_store=base.run_store, memory_store=base.memory_store, loop_cls=ParallelRunLoop
        )
        loop.raw_executor = executor
        loop._slot_held = True
        response = await loop.start_run(request)
    except BaseException:
        if loop is not None:
            await loop.release_parallel_run()
        else:
            slots.release()
        raise
    runtime.run_loops[response.run_id] = loop
    return loop, response


def desktop_loop_holds_run(run_id: str) -> bool:
    """Whether the desktop loop (if it was ever built) owns ``run_id``. Never builds it."""
    loop = current_runtime().desktop_loop
    holds = getattr(getattr(loop, "run_store", None), "holds_run", None)
    return bool(holds is not None and holds(run_id))


def get_desktop_agent_loop() -> AgentLoop:
    """Build the desktop runtime lazily so env loading can happen first."""
    runtime = current_runtime()
    if runtime.desktop_loop is None:
        desktop_config = desktop_mode_config()
        services = build_desktop_services()
        # KAN-80: the environment provider owns the machine and builds its executor.
        provider = desktop_provider()
        executor = provider.build_executor()
        run_store = FileBackedRunStore(environment="desktop")
        memory_store = FileBackedMemoryStore()
        runtime.desktop_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
            perception_service=services.perception_service,
            run_store=run_store,
            # A machine reached through the guest agent runs the tools itself (guest.py),
            # so they act inside it, never on this host.
            executor=_with_tools(executor, desktop_config) if provider.runs_on_host else executor,
            memory_store=memory_store,
            environment=UnifiedEnvironment.DESKTOP,
            **_loop_layers(services=services),
        )
        # The API streams the desktop when its executor can (the Linux cloud computer).
        _ws_stream.set_desktop_executor(executor)
    return runtime.desktop_loop

