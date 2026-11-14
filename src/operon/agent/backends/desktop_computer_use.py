"""Desktop backend for Gemini's computer use tool (``environment: desktop``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operon.agent.app_catalog import APP_LAUNCH_TARGETS, launch_targets
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.clients.gemini_computer_use import GeminiComputerUseClient

DESKTOP_ENVIRONMENT = "ENVIRONMENT_DESKTOP"


# Gemini's desktop environment has OS-level cursor and key actions but no navigate,
# so apps are opened with this custom function. Only catalog apps can be launched;
# the executor refuses anything else.
def launch_app_function(targets: dict[str, str] | None = None) -> dict[str, Any]:
    """The launch_app declaration, its enum drawn from ``targets`` (default: the
    configured platform's catalog, see ``app_catalog.launch_targets``)."""
    apps = sorted(launch_targets() if targets is None else targets)
    return {
        "name": "launch_app",
        "description": (
            "Open a desktop application by name. Use this instead of clicking the Start menu, "
            "taskbar or desktop icons. Opening an app that is already running opens another window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "app": {"type": "string", "enum": apps, "description": "Application to open."},
            },
            "required": ["app"],
        },
    }


# The Windows declaration (the default platform).
GEMINI_LAUNCH_APP_FUNCTION: dict[str, Any] = launch_app_function(APP_LAUNCH_TARGETS)


class DesktopComputerUseBackend(BrowserComputerUseBackend):
    """Drives the Windows desktop with Gemini computer use.

    Same conversation, safety-confirmation and completion handling as the browser
    backend; the desktop executor runs the actions. There is no page URL to report
    back, so no browser runtime is passed. A client with a ``environment`` setting
    (the HTTP client) must ask for the desktop environment and declare launch_app.
    """

    _retry_suffix = (
        "\n\nRetry guidance:\n"
        "- A text-only response is not enough while the task is not done on screen.\n"
        "- Take one concrete action: launch_app to open the app the task needs, or click in its window.\n"
    )

    def __init__(self, *, client: GeminiComputerUseClient, prompt_path: Path, tools: tuple[str, ...] = ()) -> None:
        environment = getattr(client, "environment", DESKTOP_ENVIRONMENT)
        declared = {tool.get("name") for tool in getattr(client, "function_declarations", [GEMINI_LAUNCH_APP_FUNCTION])}
        if environment != DESKTOP_ENVIRONMENT or "launch_app" not in declared:
            raise ValueError(
                "DesktopComputerUseBackend needs a client with environment=ENVIRONMENT_DESKTOP "
                "and the launch_app function declared"
            )
        super().__init__(client=client, prompt_path=prompt_path, browser_runtime=None, tools=tools)
