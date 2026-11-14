"""The environment-provider interface (KAN-80)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderCheck:
    """Whether a provider can run here, and if not, what is missing (plain sentences)."""

    ok: bool
    problems: tuple[str, ...] = field(default_factory=tuple)


class EnvironmentProvider(ABC):
    """One place a desktop run can happen.

    Lifecycle, as the API drives it: ``check()`` (cheap, no side effects), then
    ``start()`` once before the desktop loop is built, then ``build_executor()``,
    and ``stop()`` once at shutdown. ``start`` and ``stop`` must be safe to call
    twice; ``stop`` must not raise for a provider that never started.
    """

    #: Short name used in ``OPERON_ENVIRONMENT`` and in run reports.
    name: str = ""
    #: One plain sentence for listings (``operon envs``).
    summary: str = ""

    @abstractmethod
    def platform(self) -> str:
        """``windows`` or ``linux``: picks the app catalog, prompt and executor."""

    def check(self) -> ProviderCheck:
        """Whether this provider can run on this machine right now."""
        return ProviderCheck(ok=True)

    def start(self) -> None:
        """Get the machine ready (boot a sandbox, start a container). No-op by default."""

    @abstractmethod
    def build_executor(self):
        """The desktop executor that drives this machine."""

    def stop(self) -> None:
        """Release the machine (close the sandbox, stop the container). No-op by default."""
