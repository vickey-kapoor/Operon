"""Desktop backend for Claude's computer use toolset (``OPERON_DESKTOP_BACKEND=claude_computer_use``)."""

from __future__ import annotations

from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)


class DesktopClaudeComputerUseBackend(BrowserClaudeComputerUseBackend):
    """Drives the Windows desktop with Claude's computer use toolset.

    Same conversation, screenshot downscaling and coordinate mapping as the
    browser backend; the desktop executor runs the actions. Apps are opened with
    the ``launch_app`` custom tool instead of ``navigate``. There is no fallback
    backend, so API failures surface to the loop as step errors.
    """

    system_prompt = (
        "You are a desktop automation agent on Windows. "
        "You see the whole screen through screenshots and act with the tools."
    )
    custom_tool_names = frozenset({"launch_app"})
    screen_name = "Desktop"
