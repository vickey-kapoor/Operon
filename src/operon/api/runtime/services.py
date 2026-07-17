"""Runtime service construction for browser and desktop modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from operon.agent.backends.anthropic_policy import AnthropicPolicyService
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.browser_json import BrowserJsonBackend
from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.backends.fallback import FallbackBackend
from operon.agent.perception import GeminiPerceptionService, PerceptionService
from operon.agent.policy import GeminiPolicyService, PolicyService
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.clients.anthropic import AnthropicHttpClient
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


def read_static_prefix(prompt_path: Path) -> str | None:
    """Read the static prefix before the first template placeholder."""
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except OSError:
        return None
    idx = text.find("{")
    if idx <= 0:
        return None
    prefix = text[:idx].rstrip()
    return prefix or None


def build_policy_delegate(*, config, prompt_name: str) -> PolicyService:
    planner_provider = config.planner_provider.lower()
    if planner_provider == "anthropic":
        planner_model = config.planner_model or "claude-sonnet-5"
        return AnthropicPolicyService(
            anthropic_client=AnthropicHttpClient(model=planner_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / prompt_name,
        )
    prompt_path = _PROMPTS_DIR / prompt_name
    return GeminiPolicyService(
        gemini_client=GeminiHttpClient(
            model=config.primary_model,
            timeout_seconds=120.0,
            cacheable_system_prompt=read_static_prefix(prompt_path),
        ),
        prompt_path=prompt_path,
    )


def build_verifier_client(*, config):
    verifier_provider = config.verifier_provider.lower()
    verifier_model = config.verifier_model or config.fallback_model or config.primary_model
    if verifier_provider == "anthropic":
        verifier_model = verifier_model or "claude-sonnet-5"
        return AnthropicHttpClient(model=verifier_model, timeout_seconds=120.0)
    return GeminiHttpClient(model=verifier_model, timeout_seconds=120.0)


def build_browser_services(executor) -> RuntimeServices:
    config = browser_mode_config()
    if config.backend == "json":
        if config.planner_provider.lower() == "anthropic":
            return RuntimeServices(
                perception_service=GeminiPerceptionService(
                    gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
                    prompt_path=_PROMPTS_DIR / "perception_prompt.txt",
                ),
                policy_delegate=build_policy_delegate(config=config, prompt_name="policy_prompt.txt"),
            )
        backend = BrowserJsonBackend(
            gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
            prompt_path=_PROMPTS_DIR / "browser_combined_prompt.txt",
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend == "computer_use":
        primary = BrowserComputerUseBackend(
            client=GeminiComputerUseHttpClient(model=config.primary_model),
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
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Only 'json' is implemented in this slice."
        )
    if config.planner_provider.lower() == "anthropic":
        return RuntimeServices(
            perception_service=GeminiPerceptionService(
                gemini_client=GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0),
                prompt_path=_PROMPTS_DIR / "desktop_perception_prompt.txt",
            ),
            policy_delegate=build_policy_delegate(config=config, prompt_name="desktop_policy_prompt.txt"),
        )
    backend = build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )
    return RuntimeServices(perception_service=backend, policy_delegate=backend)

