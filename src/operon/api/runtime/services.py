"""Runtime service construction for browser and desktop modes."""

from __future__ import annotations

import logging
import os
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
from operon.agent.backends.desktop_computer_use import (
    DESKTOP_ENVIRONMENT,
    GEMINI_LAUNCH_APP_FUNCTION,
    DesktopComputerUseBackend,
)
from operon.agent.backends.fallback import FallbackBackend
from operon.agent.backends.model_adapter.backend import ModelAdapterBackend
from operon.agent.backends.model_adapter.critic import ProfileCriticClient
from operon.agent.backends.model_adapter.openai_native import OpenAIComputerUseBackend
from operon.agent.backends.model_adapter.registry import (
    ModelProfile,
    ModelRegistryError,
    resolve_profile,
)
from operon.agent.backends.model_adapter.transports import (
    OpenAICompatClient,
    OpenAIResponsesClient,
)
from operon.agent.perception import PerceptionService
from operon.agent.policy import PolicyService
from operon.agent.policy.thin import THIN, harness_profile
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
    verifier_provider = config.verifier_provider.lower()
    verifier_model = config.verifier_model or config.fallback_model or config.primary_model
    if verifier_provider == "anthropic":
        verifier_model = verifier_model or "claude-sonnet-5-5"
        return AnthropicHttpClient(model=verifier_model, timeout_seconds=120.0)
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
    model (KAN-68), so "Claude full" or "GPT full" is not judged by a Gemini critic;
    otherwise the configured verifier (``build_verifier_client``)."""
    if critic_setting() != SAME_CRITIC:
        return build_verifier_client(config=config)
    if config.backend == "claude_computer_use":
        return AnthropicHttpClient(model=config.claude_model or "claude-opus-5-5", timeout_seconds=120.0)
    if config.backend == "model":
        profile = resolve_profile()
        if profile.adapter == "claude_native":
            return AnthropicHttpClient(model=profile.model, timeout_seconds=120.0)
        if profile.adapter == "gemini_native":
            return GeminiHttpClient(model=profile.model, timeout_seconds=120.0, thinking=profile.effort)
        return ProfileCriticClient(profile)
    # json and Gemini computer use act with a Gemini model: judge with that model.
    return GeminiHttpClient(model=config.primary_model, timeout_seconds=120.0, thinking=config.thinking)


def build_model_backend(
    environment: str, *, executor=None, profile: ModelProfile | None = None, client=None
) -> AgentBackend:
    """The backend for ``OPERON_{BROWSER,DESKTOP}_BACKEND=model``: the registry profile
    named by ``OPERON_MODEL_PROFILE``. Native profiles reuse Operon's own backends with
    the profile's model; the rest go through the model adapter. ``client`` reuses a
    transport client built for an earlier backend of the same profile and environment."""
    profile = profile or resolve_profile()
    if environment not in profile.environments:
        raise ModelRegistryError(f"profile {profile.name} does not support {environment}")
    if profile.adapter == "claude_native":
        backend_cls = BrowserClaudeComputerUseBackend if environment == "browser" else DesktopClaudeComputerUseBackend
        return backend_cls(
            client=client
            or AnthropicComputerUseHttpClient(
                model=profile.model,
                effort=profile.effort,
                **({"custom_tools": (LAUNCH_APP_TOOL,)} if environment == "desktop" else {}),
            ),
            prompt_path=_PROMPTS_DIR / f"{environment}_claude_computer_use_prompt.txt",
            **({"browser_runtime": executor} if environment == "browser" else {}),
        )
    if profile.adapter == "gemini_native":
        if environment == "desktop":
            return DesktopComputerUseBackend(
                client=client
                or GeminiComputerUseHttpClient(
                    model=profile.model,
                    thinking_level=profile.effort,
                    environment=DESKTOP_ENVIRONMENT,
                    function_declarations=[GEMINI_LAUNCH_APP_FUNCTION],
                ),
                prompt_path=_PROMPTS_DIR / "desktop_computer_use_prompt.txt",
            )
        return BrowserComputerUseBackend(
            client=client or GeminiComputerUseHttpClient(model=profile.model, thinking_level=profile.effort),
            prompt_path=_PROMPTS_DIR / "browser_computer_use_prompt.txt",
            browser_runtime=executor,
        )
    if profile.adapter == "openai_compat":
        return ModelAdapterBackend(
            profile=profile,
            client=client or OpenAICompatClient(profile),
            environment=environment,
            task_prompt_path=_PROMPTS_DIR / "model_adapter_task_prompt.txt",
            browser_runtime=executor,
        )
    if profile.adapter == "openai_native":
        if profile.pricing.input_per_mtok is None or profile.pricing.output_per_mtok is None:
            # Without prices every step costs $0 and no run cost cap can stop a paid run.
            raise ModelRegistryError(f"profile {profile.name} needs pricing (input_per_mtok and output_per_mtok)")
        return OpenAIComputerUseBackend(
            profile=profile,
            client=client or OpenAIResponsesClient(profile),
            environment=environment,
            task_prompt_path=_PROMPTS_DIR / "model_adapter_task_prompt.txt",
            browser_runtime=executor,
        )
    raise ModelRegistryError(f"profile {profile.name}: adapter {profile.adapter} is not available")


def build_browser_services(executor) -> RuntimeServices:
    config = browser_mode_config()
    if config.backend == "model":
        backend = build_model_backend("browser", executor=executor)
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
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


def build_desktop_services() -> RuntimeServices:
    config = desktop_mode_config()
    if config.backend == "model":
        backend = build_model_backend("desktop")
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
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
    if config.backend == "computer_use":
        backend = DesktopComputerUseBackend(
            client=GeminiComputerUseHttpClient(
                model=config.primary_model,
                thinking_level=config.thinking,
                environment=DESKTOP_ENVIRONMENT,
                function_declarations=[GEMINI_LAUNCH_APP_FUNCTION],
            ),
            prompt_path=_PROMPTS_DIR / "desktop_computer_use_prompt.txt",
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend != "json":
        raise ValueError(
            f"Unsupported desktop backend {config.backend!r}. "
            "Use 'json', 'computer_use', 'claude_computer_use' or 'model'."
        )
    backend = build_json_backend(
        prompt_name="desktop_combined_prompt.txt",
        model=config.primary_model,
    )
    return RuntimeServices(perception_service=backend, policy_delegate=backend)

