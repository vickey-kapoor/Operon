"""Desktop backend for Gemini's computer use tool (``environment: desktop``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.clients.anthropic_computer_use import LAUNCH_APP_TOOL
from operon.clients.gemini_computer_use import GeminiComputerUseClient

DESKTOP_ENVIRONMENT = "ENVIRONMENT_DESKTOP"

# Gemini's desktop environment has OS-level cursor and key actions but no navigate,
# so apps are opened with this custom function: the same name, description and app
# list as Claude's launch_app tool, in Gemini's function-declaration shape.
GEMINI_LAUNCH_APP_FUNCTION: dict[str, Any] = {
    "name": LAUNCH_APP_TOOL["name"],
    "description": LAUNCH_APP_TOOL["description"],
    "parameters": {
        key: value for key, value in LAUNCH_APP_TOOL["input_schema"].items() if key != "additionalProperties"
    },
}


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

    def __init__(self, *, client: GeminiComputerUseClient, prompt_path: Path) -> None:
        environment = getattr(client, "environment", DESKTOP_ENVIRONMENT)
        declared = {tool.get("name") for tool in getattr(client, "function_declarations", [GEMINI_LAUNCH_APP_FUNCTION])}
        if environment != DESKTOP_ENVIRONMENT or "launch_app" not in declared:
            raise ValueError(
                "DesktopComputerUseBackend needs a client with environment=ENVIRONMENT_DESKTOP "
                "and the launch_app function declared"
            )
        super().__init__(client=client, prompt_path=prompt_path, browser_runtime=None)
