"""Runtime service construction for browser and desktop modes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from operon.agent.app_catalog import LINUX, desktop_platform
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.browser_json import BrowserJsonBackend
from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.backends.desktop_computer_use import (
    DESKTOP_ENVIRONMENT,
    DesktopComputerUseBackend,
    launch_app_function,
)
from operon.agent.backends.fallback import FallbackBackend
from operon.agent.perception import PerceptionService
from operon.agent.policy import PolicyService
from operon.agent.policy.thin import THIN, harness_profile
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.clients.gemini import GeminiHttpClient
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.paths import prompts_dir

_PROMPTS_DIR = prompts_dir()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeServices:
    perception_service: PerceptionService
    policy_delegate: PolicyService


def build_json_backend(*, prompt_name: str, model: str, timeout_seconds: float = 120.0) -> AgentBackend:
    return CombinedPerceptionPolicyService(
        gemini_client=GeminiHttpClient(model=model, timeout_seconds=timeout_seconds),
        prompt_path=_PROMPTS_DIR / prompt_name,
    )


def build_verifier_client(*, config):
    verifier_model = config.verifier_model or config.fallback_model or config.primary_model
    return GeminiHttpClient(
        model=verifier_model,
        timeout_seconds=120.0,
        thinking=config.verifier_thinking,
    )


SAME_CRITIC = "same"


def critic_setting() -> str:
    """``OPERON_CRITIC``: ``same`` runs the full harness's step critic on the acting
    model; anything else (the default) keeps the ``OPERON_*_VERIFIER_*`` critic."""
    return (os.getenv("OPERON_CRITIC") or "").strip().lower()


def build_critic_client(*, config):
    """The full harness's step critic. With ``OPERON_CRITIC=same`` it is the run's own
    acting model (KAN-68); otherwise the configured verifier (``build_verifier_client``)."""
    if critic_setting() != SAME_CRITIC:
        return build_verifier_client(config=config)
    return GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0, thinking=config.thinking)


def build_browser_services(executor) -> RuntimeServices:
    config = browser_mode_config()
    if config.backend == "json":
        backend = BrowserJsonBackend(
            gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend == "computer_use":
        primary = BrowserComputerUseBackend(
            client=GeminiComputerUseHttpClient(
                model=config.primary_model,
                thinking_level=config.thinking,
            ),
            prompt_path=_PROMPTS_DIR / "browser_computer_use_prompt.txt",
            browser_runtime=executor,
        )
        if harness_profile(config.backend) == THIN:
            # The loop's policy, verifier and recovery are fixed for the profile when it
            # is built. A JSON fallback would run its decisions with no rules, guards or
            # critic, so under the thin harness a model failure ends the run instead.
            if config.fallback_backend:
                logger.debug("OPERON_BROWSER_FALLBACK_BACKEND is ignored under the thin harness")
            return RuntimeServices(perception_service=primary, policy_delegate=primary)
        if config.fallback_backend == "json" and config.fallback_model:
            secondary = BrowserJsonBackend(
                gemini_client=GeminiHttpClient(model=config.fallback_model, timeout_seconds=120.0),
                prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
            )
            backend = FallbackBackend(primary=primary, secondary=secondary)
            return RuntimeServices(perception_service=backend, policy_delegate=backend)
        return RuntimeServices(perception_service=primary, policy_delegate=primary)
    raise ValueError(f"Unsupported browser backend {config.backend!r}")


def desktop_prompt_name() -> str:
    """The desktop computer-use prompt for the configured platform (Windows by default)."""
    if desktop_platform() == LINUX:
        return "desktop_computer_use_prompt_linux.txt"
    return "desktop_computer_use_prompt.txt"


def build_desktop_services() -> RuntimeServices:
    config = desktop_mode_config()
    if config.backend == "computer_use":
        backend = DesktopComputerUseBackend(
            client=GeminiComputerUseHttpClient(
                model=config.primary_model,
                thinking_level=config.thinking,
                environment=DESKTOP_ENVIRONMENT,
                function_declarations=[launch_app_function()],
            ),
            prompt_path=_PROMPTS_DIR / desktop_prompt_name(),
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Use 'computer_use' or 'json'."
        )
    backend = build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )
    return RuntimeServices(perception_service=backend, policy_delegate=backend)

