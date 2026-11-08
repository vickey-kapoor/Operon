"""Action formats: the prompt, tools and reply parser for one family of models."""

from __future__ import annotations

from operon.agent.backends.model_adapter.formats.base import (
    ActionFormat,
    ParsedReply,
    PromptContext,
    Turn,
)
from operon.agent.backends.model_adapter.formats.neutral_tools import NeutralToolsFormat

FORMATS: dict[str, type[ActionFormat]] = {
    "neutral_tools": NeutralToolsFormat,
}


def get_format(name: str) -> ActionFormat:
    try:
        return FORMATS[name]()
    except KeyError as exc:
        raise ValueError(f"no action format named {name!r}; known: {', '.join(sorted(FORMATS))}") from exc


__all__ = ["ActionFormat", "FORMATS", "ParsedReply", "PromptContext", "Turn", "get_format"]
