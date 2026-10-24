"""Lazy AgentLoop construction for API routes."""

from __future__ import annotations

import importlib

from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.coordinator import PolicyCoordinator
from operon.agent.policy.recovery import RuleBasedRecoveryManager
from operon.agent.policy.thin import (
    THIN,
    FinalCheckVerifier,
    ThinPolicy,
    ThinRecovery,
    final_check_model,
    harness_profile,
)
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.api import ws_stream as _ws_stream
from operon.api.runtime.services import (
    build_browser_services,
    build_desktop_services,
    build_verifier_client,
)
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.clients.anthropic import AnthropicHttpClient
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


def _loop_layers(*, config, services, memory_store) -> dict:
    """Policy, verifier and recovery for the configured harness profile."""
    profile = harness_profile(config.backend)
    if profile == THIN:
        delegate = services.policy_delegate
        return {
            "policy_service": ThinPolicy(delegate),
            "verifier_service": FinalCheckVerifier(
                AnthropicHttpClient(model=final_check_model(config.claude_model), timeout_seconds=120.0),
                on_rejected=lambda run_id, hint: delegate.add_advisory_hints([hint], source="final_check", run_id=run_id),
                effort=config.claude_effort,
            ),
            "recovery_manager": ThinRecovery(),
            "harness_profile": profile,
        }
    return {
        "policy_service": PolicyCoordinator(
            delegate=services.policy_delegate,
            memory_store=memory_store,
            element_buffer=getattr(services.perception_service, "element_buffer", None),
        ),
        "verifier_service": DeterministicVerifierService(gemini_client=build_verifier_client(config=config)),
        "recovery_manager": RuleBasedRecoveryManager(),
        "harness_profile": profile,
    }


def get_agent_loop() -> AgentLoop:
    """Build the browser runtime lazily so env loading can happen first."""
    global _agent_loop
    if _agent_loop is None:
        browser_config = browser_mode_config()
        executor = _build_browser_executor()
        services = build_browser_services(executor)
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
            perception_service=services.perception_service,
            run_store=run_store,
            executor=executor,
            memory_store=memory_store,
            environment=UnifiedEnvironment.BROWSER,
            **_loop_layers(config=browser_config, services=services, memory_store=memory_store),
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
        executor = DesktopExecutor()
        run_store = FileBackedRunStore()
        memory_store = FileBackedMemoryStore()
        _desktop_agent_loop = AgentLoop(
            capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
            perception_service=services.perception_service,
            run_store=run_store,
            executor=executor,
            memory_store=memory_store,
            environment=UnifiedEnvironment.DESKTOP,
            **_loop_layers(config=desktop_config, services=services, memory_store=memory_store),
        )
    return _desktop_agent_loop

