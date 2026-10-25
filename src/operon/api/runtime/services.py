"""Runtime service construction for browser and desktop modes."""

from __future__ import annotations

from dataclasses import dataclass

from operon.agent.backends.base import AgentBackend
from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.browser_json import BrowserJsonBackend
from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.backends.desktop_claude_computer_use import (
    DesktopClaudeComputerUseBackend,
)
from operon.agent.backends.fallback import FallbackBackend
from operon.agent.perception import PerceptionService
from operon.agent.policy import PolicyService
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.clients.anthropic import AnthropicHttpClient
from operon.clients.anthropic_computer_use import (
    LAUNCH_APP_TOOL,
    AnthropicComputerUseHttpClient,
)
from operon.clients.gemini import GeminiHttpClient
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.paths import prompts_dir

_PROMPTS_DIR = prompts_dir()


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
    verifier_provider = config.verifier_provider.lower()
    verifier_model = config.verifier_model or config.fallback_model or config.primary_model
    if verifier_provider == "anthropic":
        verifier_model = verifier_model or "claude-sonnet-5"
        return AnthropicHttpClient(model=verifier_model, timeout_seconds=120.0)
    return GeminiHttpClient(
        model=verifier_model,
        timeout_seconds=120.0,
        thinking=config.verifier_thinking,
    )


def build_browser_services(executor) -> RuntimeServices:
    config = browser_mode_config()
    if config.backend == "json":
        backend = BrowserJsonBackend(
            gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend in {"computer_use", "claude_computer_use"}:
        if config.backend == "claude_computer_use":
            primary = BrowserClaudeComputerUseBackend(
                client=AnthropicComputerUseHttpClient(
                    model=config.claude_model or "claude-opus-5-5",
                    effort=config.claude_effort,
                ),
                prompt_path=_PROMPTS_DIR / "browser_claude_computer_use_prompt.txt",
                browser_runtime=executor,
            )
        else:
            primary = BrowserComputerUseBackend(
                client=GeminiComputerUseHttpClient(
                    model=config.primary_model,
                    thinking_level=config.thinking,
                ),
                prompt_path=_PROMPTS_DIR / "browser_computer_use_prompt.txt",
                browser_runtime=executor,
            )
        if config.fallback_backend == "json" and config.fallback_model:
            secondary = BrowserJsonBackend(
                gemini_client=GeminiHttpClient(model=config.fallback_model, timeout_seconds=120.0),
                prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
            )
            backend = FallbackBackend(primary=primary, secondary=secondary)
            return RuntimeServices(perception_service=backend, policy_delegate=backend)
        return RuntimeServices(perception_service=primary, policy_delegate=primary)
    raise ValueError(f"Unsupported browser backend {config.backend!r}")


def build_desktop_services() -> RuntimeServices:
    config = desktop_mode_config()
    if config.backend == "claude_computer_use":
        backend = DesktopClaudeComputerUseBackend(
            client=AnthropicComputerUseHttpClient(
                model=config.claude_model or "claude-opus-5-5",
                effort=config.claude_effort,
                custom_tools=(LAUNCH_APP_TOOL,),
            ),
            prompt_path=_PROMPTS_DIR / "desktop_claude_computer_use_prompt.txt",
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Use 'json' or 'claude_computer_use'."
        )
    backend = build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )
    return RuntimeServices(perception_service=backend, policy_delegate=backend)

