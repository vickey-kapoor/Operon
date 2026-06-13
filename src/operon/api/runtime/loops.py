"""Lazy AgentLoop construction for API routes."""

from __future__ import annotations

import importlib

from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.coordinator import PolicyCoordinator
from operon.agent.policy.recovery import RuleBasedRecoveryManager
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.api import ws_stream as _ws_stream
from operon.api.runtime.services import (
    build_browser_services,
    build_desktop_services,
    build_verifier_client,
)
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.core.contracts.perception import Environment as UnifiedEnvironment
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

_agent_loop: AgentLoop | None = None
_desktop_agent_loop: AgentLoop | None = None
DesktopExecutor = None
NativeBrowserExecutor = None


def iter_built_executors():
    """Yield the executor of each agent loop that has actually been built.

    Reads the module globals directly so it never triggers lazy construction
    (which would launch a browser). Used by the server lifespan to tear down
    executor-held resources at shutdown.
    """
    for loop in (_agent_loop, _desktop_agent_loop):
        if loop is not None:
            yield loop.executor


def _build_browser_executor():
    global NativeBrowserExecutor
    if NativeBrowserExecutor is None:
        NativeBrowserExecutor = importlib.import_module("operon.executor.browser_native").NativeBrowserExecutor
    return NativeBrowserExecutor()


def get_agent_loop() -> AgentLoop:
    """Build the browser runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        browser_config = browser_mode_config()
        verifier_client = build_verifier_client(config=browser_config)
        executor = _build_browser_executor()
        services = build_browser_services(executor)
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
                element_buffer=getattr(services.perception_service, "element_buffer", None),
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=verifier_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            environment=UnifiedEnvironment.BROWSER,
        )
        _ws_stream.set_executor(executor)
        _ws_stream.set_agent_loop(_agent_loop)
    return _agent_loop


def get_desktop_agent_loop() -> AgentLoop:
    """Build the desktop runtime lazily so env loading can happen first."""
    global _desktop_agent_loop, DesktopExecutor
    if _desktop_agent_loop is None:
        if DesktopExecutor is None:
            from operon.executor.desktop import DesktopExecutor as _DesktopExecutor

            DesktopExecutor = _DesktopExecutor

        desktop_config = desktop_mode_config()
        services = build_desktop_services()
        verifier_client = build_verifier_client(config=desktop_config)
        executor = DesktopExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _desktop_agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor),
            perception_service=services.perception_service,
            run_store=run_store,
            policy_service=PolicyCoordinator(
                delegate=services.policy_delegate,
                memory_store=memory_store,
                element_buffer=getattr(services.perception_service, "element_buffer", None),
            ),
            executor=executor,
            verifier_service=DeterministicVerifierService(gemini_client=verifier_client),
            recovery_manager=RuleBasedRecoveryManager(),
            memory_store=memory_store,
            environment=UnifiedEnvironment.DESKTOP,
        )
    return _desktop_agent_loop

