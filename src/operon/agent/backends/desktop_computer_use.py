"""Desktop backend for Gemini's computer use tool (``environment: desktop``)."""

from __future__ import annotations

from typing import Any

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend

# Gemini's desktop environment has OS-level cursor and key actions but no navigate,
# so apps are opened with this custom function, as the Claude and OpenAI desktop
# backends do with their own launch_app tools.
GEMINI_LAUNCH_APP_FUNCTION: dict[str, Any] = {
    "name": "launch_app",
    "description": (
        "Open a desktop application by name. Use this instead of clicking the Start menu, "
        "taskbar or desktop icons. Opening an app that is already running opens another window."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app": {"type": "string", "enum": sorted(APP_LAUNCH_TARGETS), "description": "Application to open."},
        },
        "required": ["app"],
    },
}


class DesktopComputerUseBackend(BrowserComputerUseBackend):
    """Drives the Windows desktop with Gemini computer use.

    Same conversation, safety-confirmation and completion handling as the browser
    backend. The client must ask for ``ENVIRONMENT_DESKTOP`` and declare
    ``launch_app``; the desktop executor runs the actions. There is no page URL
    to report back, so no browser runtime is passed.
    """

    def __init__(self, *, client, prompt_path) -> None:
        super().__init__(client=client, prompt_path=prompt_path, browser_runtime=None)
        self._retry_suffix = (
            "\n\nRetry guidance:\n"
            "- A text-only response is not enough while the task is not done on screen.\n"
            "- Take one concrete action: launch_app to open the app the task needs, or click in its window.\n"
        )
