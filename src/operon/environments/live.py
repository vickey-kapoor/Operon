"""The live desktop provider: the machine Operon itself runs on (today's behavior)."""

from __future__ import annotations

from operon.agent.app_catalog import LINUX, desktop_platform
from operon.environments.base import EnvironmentProvider


class LiveDesktopProvider(EnvironmentProvider):
    """Drives the real screen, mouse and keyboard of this machine.

    Nothing to start or stop. ``OPERON_DESKTOP_PLATFORM`` still picks Windows (the
    default) or an X11 Linux desktop such as the Azure Linux computer (KAN-108).
    """

    name = "live"
    summary = "This machine's own screen, mouse and keyboard (the default)."

    def platform(self) -> str:
        return desktop_platform()

    def build_executor(self):
        if self.platform() == LINUX:
            from operon.executor.linux_desktop import LinuxDesktopExecutor

            return LinuxDesktopExecutor()
        from operon.executor.desktop import DesktopExecutor

        return DesktopExecutor()
