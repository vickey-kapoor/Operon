"""Lazy AgentLoop construction for API routes."""

from __future__ import annotations

import asyncio
import importlib
import logging

from operon.agent.backends.model_adapter.backend import GAVE_UP_PREFIX
from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.coordinator import PolicyCoordinator
from operon.agent.policy.recovery import RuleBasedRecoveryManager
from operon.agent.policy.rules import PolicyRuleEngine
from operon.agent.policy.thin import (
    THIN,
    OwnDoneVerifier,
    ThinPolicy,
    ThinRecovery,
    harness_profile,
)
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.agent.tools import enabled_tools
from operon.api import ws_stream as _ws_stream
from operon.api.runtime.services import (
    build_browser_services,
    build_critic_client,
    build_desktop_services,
)
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.core.contracts.perception import Environment as UnifiedEnvironment
from operon.core.runtime import current_runtime
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
    for loop in (runtime.browser_loop, runtime.desktop_loop):
        if loop is not None:
            yield loop.executor


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
    try:
        from operon.browser.builtin import close_builtin_browser

        await close_builtin_browser()
    except Exception as exc:  # noqa: BLE001 - shutdown cleanup must not raise
        logger.warning("built-in browser shutdown cleanup skipped: %s", exc)
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


def _loop_layers(*, config, services, memory_store, desktop: bool = False) -> dict:
    """Policy, verifier and recovery for the configured harness profile."""
    profile = harness_profile(config.backend)
    if profile == THIN:
        return {
            "policy_service": ThinPolicy(services.policy_delegate),
            "verifier_service": OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX),
            "recovery_manager": ThinRecovery(),
            "harness_profile": profile,
        }
    return {
        "policy_service": PolicyCoordinator(
            delegate=services.policy_delegate,
            memory_store=memory_store,
            rule_engine=PolicyRuleEngine(launch_apps=desktop),
        ),
        "verifier_service": DeterministicVerifierService(gemini_client=build_critic_client(config=config)),
        "recovery_manager": RuleBasedRecoveryManager(),
        "harness_profile": profile,
    }


def get_agent_loop() -> AgentLoop:
    """Build the browser runtime lazily so env loading can happen first."""
    runtime = current_runtime()
    if runtime.browser_loop is None:
        browser_config = browser_mode_config()
        executor = _build_browser_executor()
        services = build_browser_services(executor)
        run_store = FileBackedRunStore(environment="browser")
        memory_store = FileBackedMemoryStore()
        loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
            perception_service=services.perception_service,
            run_store=run_store,
            executor=_with_tools(executor, browser_config),
            memory_store=memory_store,
            environment=UnifiedEnvironment.BROWSER,
            **_loop_layers(config=browser_config, services=services, memory_store=memory_store, desktop=False),
        )
        runtime.browser_loop = loop
        _ws_stream.set_executor(executor)
        _ws_stream.set_agent_loop(loop)
    return runtime.browser_loop


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
            **_loop_layers(config=desktop_config, services=services, memory_store=memory_store, desktop=True),
        )
        # The task page streams the desktop when its executor can (the Linux cloud computer).
        _ws_stream.set_desktop_executor(executor)
    return runtime.desktop_loop

