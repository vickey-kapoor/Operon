"""Provider lookup by name (``OPERON_ENVIRONMENT``)."""

from __future__ import annotations

import os
from collections.abc import Callable

from operon.environments.base import EnvironmentProvider

DEFAULT_ENVIRONMENT = "live"


def _live() -> EnvironmentProvider:
    from operon.environments.live import LiveDesktopProvider

    return LiveDesktopProvider()


def _windows_sandbox() -> EnvironmentProvider:
    from operon.environments.windows_sandbox import WindowsSandboxProvider

    return WindowsSandboxProvider()


def _linux_container() -> EnvironmentProvider:
    from operon.environments.linux_container import LinuxContainerProvider

    return LinuxContainerProvider()


# Name -> factory. Factories import lazily so an unused provider's dependencies
# (pyautogui, docker, az) are never loaded.
_PROVIDERS: dict[str, Callable[[], EnvironmentProvider]] = {
    "live": _live,
    "windows-sandbox": _windows_sandbox,
    "linux-container": _linux_container,
}


def provider_names() -> list[str]:
    return sorted(_PROVIDERS)


def environment_name() -> str:
    """``OPERON_ENVIRONMENT``, lower-cased; ``live`` when unset or blank."""
    return (os.getenv("OPERON_ENVIRONMENT") or "").strip().lower() or DEFAULT_ENVIRONMENT


def get_provider(name: str | None = None) -> EnvironmentProvider:
    """The provider called ``name`` (default: ``OPERON_ENVIRONMENT``)."""
    key = (name or environment_name()).strip().lower()
    factory = _PROVIDERS.get(key)
    if factory is None:
        raise ValueError(
            f"OPERON_ENVIRONMENT={key!r} is not a known environment; use one of {provider_names()}"
        )
    return factory()
