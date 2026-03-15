"""Abstract executor protocol for UI navigation backends."""

from typing import Protocol, runtime_checkable

from PIL import Image

from .actions import Action, ActionResult


@runtime_checkable
class AbstractExecutor(Protocol):
    """Protocol that all executor backends must satisfy.

    Both PlaywrightBrowserExecutor (browser) and DesktopExecutor (desktop)
    implement this interface so the vision pipeline can remain agnostic
    about the input/output surface.
    """

    async def start(self) -> None:
        """Initialize the executor and acquire any required resources."""
        ...

    async def stop(self) -> None:
        """Release all resources held by the executor."""
        ...

    async def screenshot(self) -> Image.Image:
        """Capture the current screen/viewport and return as a PIL Image."""
        ...

    async def screenshot_base64(self) -> str:
        """Capture the current screen/viewport and return as a base64 PNG string."""
        ...

    async def execute(self, action: Action) -> ActionResult:
        """Execute a single action and return the result."""
        ...
