"""Runtime service construction for browser and desktop modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from operon.agent.app_catalog import LINUX, desktop_platform
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.desktop_claude_computer_use import (
    DesktopClaudeComputerUseBackend,
)
from operon.agent.backends.desktop_computer_use import (
    DESKTOP_ENVIRONMENT,
    DesktopComputerUseBackend,
    launch_app_function,
)
from operon.agent.backends.model_adapter.backend import ModelAdapterBackend
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
from operon.agent.tools import enabled_tools, function_declarations
from operon.api.runtime_config import browser_mode_config, desktop_mode_config
from operon.clients.anthropic_computer_use import (
    AnthropicComputerUseHttpClient,
    launch_app_tool,
)
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.paths import prompts_dir

_PROMPTS_DIR = prompts_dir()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeServices:
    perception_service: PerceptionService
    policy_delegate: PolicyService


def _claude_prompt_name(environment: str) -> str:
    if environment == "desktop" and desktop_platform() == LINUX:
        return "desktop_claude_computer_use_prompt_linux.txt"
    return f"{environment}_claude_computer_use_prompt.txt"


def build_model_backend(
    environment: str, *, executor=None, profile: ModelProfile | None = None, client=None
) -> AgentBackend:
    """The backend for ``OPERON_{BROWSER,DESKTOP}_BACKEND=model``: the registry profile
    named by ``OPERON_MODEL_PROFILE`` (Claude, GPT, ...). ``client`` reuses a transport
    client built for an earlier backend of the same profile and environment."""
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
                **({"custom_tools": (launch_app_tool(),)} if environment == "desktop" else {}),
            ),
            prompt_path=_PROMPTS_DIR / _claude_prompt_name(environment),
            **({"browser_runtime": executor} if environment == "browser" else {}),
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
    raise ModelRegistryError(f"profile {profile.name}: adapter {profile.adapter} is not available here")


def build_browser_services(executor) -> RuntimeServices:
    config = browser_mode_config()
    if config.backend == "model":
        backend = build_model_backend("browser", executor=executor)
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend == "computer_use":
        tools = enabled_tools()
        primary = BrowserComputerUseBackend(
            client=GeminiComputerUseHttpClient(
                model=config.primary_model,
                thinking_level=config.thinking,
                function_declarations=function_declarations(tools),
            ),
            prompt_path=_PROMPTS_DIR / "browser_computer_use_prompt.txt",
            browser_runtime=executor,
            tools=tools,
        )
        return RuntimeServices(perception_service=primary, policy_delegate=primary)
    raise ValueError(f"Unsupported browser backend {config.backend!r}")


def desktop_prompt_name() -> str:
    """The desktop computer-use prompt for the configured platform (Windows by default)."""
    if desktop_platform() == LINUX:
        return "desktop_computer_use_prompt_linux.txt"
    return "desktop_computer_use_prompt.txt"


def build_desktop_services() -> RuntimeServices:
    config = desktop_mode_config()
    if config.backend == "model":
        backend = build_model_backend("desktop")
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    if config.backend == "computer_use":
        tools = enabled_tools()
        backend = DesktopComputerUseBackend(
            client=GeminiComputerUseHttpClient(
                model=config.primary_model,
                thinking_level=config.thinking,
                environment=DESKTOP_ENVIRONMENT,
                function_declarations=[launch_app_function(), *function_declarations(tools)],
            ),
            prompt_path=_PROMPTS_DIR / desktop_prompt_name(),
            tools=tools,
        )
        return RuntimeServices(perception_service=backend, policy_delegate=backend)
    raise ValueError(f"Unsupported desktop backend {config.backend!r}. Use 'computer_use' or 'model'.")
