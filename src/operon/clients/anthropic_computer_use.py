"""Anthropic client for Claude's computer use toolset (``computer_toolset_20260801``)."""

from __future__ import annotations

from typing import Any

from operon.clients.anthropic import AnthropicClientError, AnthropicHttpClient

TOOLSET_NAME = "computer"

# Toolset members the browser executor has no primitive for. Turning them off keeps
# Claude from planning around actions that would fail.
_DISABLED_MEMBERS = (
    "zoom",
    "triple_click",
    "left_mouse_down",
    "left_mouse_up",
    "cursor_position",
    "hold_key",
)

COMPUTER_TOOLSET: dict[str, Any] = {
    "type": "computer_toolset_20260801",
    "configs": {member: {"enabled": False} for member in _DISABLED_MEMBERS},
}

# The screenshot shows only the page, with no address bar, so opening a URL needs a tool.
NAVIGATE_TOOL: dict[str, Any] = {
    "name": "navigate",
    "description": (
        "Open a URL in the current browser tab. The screenshots show only the web page, "
        "with no address bar, so use this instead of typing into one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute URL to open."}},
        "required": ["url"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Server-side refusal fallback: a declined turn is re-run on a fallback model in the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicComputerUseHttpClient(AnthropicHttpClient):
    """Messages API client that drives one computer use conversation turn at a time."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        effort: str | None = "low",
        timeout_seconds: float = 120.0,
        max_tokens: int = 16000,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, timeout_seconds=timeout_seconds, **kwargs)
        self.effort = effort
        self.max_tokens = max_tokens

    def build_payload(self, *, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": [COMPUTER_TOOLSET, NAVIGATE_TOOL],
            "messages": messages,
            # Each turn resends the whole conversation; cache everything up to the newest turn.
            "cache_control": {"type": "ephemeral"},
            "fallbacks": "default",
        }
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        return payload

    async def generate_turn(self, *, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send the conversation and return the raw response (``content``, ``stop_reason``, ...)."""
        response = await self._post_json(
            self.build_payload(system=system, messages=messages),
            request_kind="computer_use",
            extra_headers={"anthropic-beta": _FALLBACK_BETA},
        )
        if not isinstance(response.get("content"), list):
            raise AnthropicClientError("Anthropic response missing content list.")
        return response
