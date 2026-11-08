"""What every action format provides, and the records the backend keeps per turn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.backends.model_adapter.actions import CoordinateMapper, NeutralAction
from operon.agent.backends.model_adapter.images import SentImage

_BROWSER_RULES = (
    "You are driving a web browser. Each screenshot shows only the current web page: there is no "
    "address bar, tab strip or browser menu, so open a URL with the navigate action and go back with "
    "the key alt+Left."
)
_DESKTOP_RULES = (
    "You are driving a Windows desktop. Each screenshot shows the whole screen. Open an app with the "
    "launch_app action instead of the Start menu. Apps you can launch: {apps}."
)


@dataclass(frozen=True, slots=True)
class PromptContext:
    intent: str
    environment: Literal["browser", "desktop"]
    mapper: CoordinateMapper
    task_prompt: str  # the shared task prompt, already filled in

    @staticmethod
    def environment_rules(environment: str) -> str:
        if environment == "desktop":
            return _DESKTOP_RULES.format(apps=", ".join(sorted(APP_LAUNCH_TARGETS)))
        return _BROWSER_RULES


@dataclass(slots=True)
class ParsedReply:
    """One reply, parsed. ``invalid`` is set when nothing in it can run."""

    actions: list[NeutralAction] = field(default_factory=list)
    terminal: NeutralAction | None = None
    thought: str = ""
    invalid: str | None = None
    # Tool-call ids in reply order, for formats whose history answers each call.
    call_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Turn:
    """One model turn as the history replays it."""

    image: SentImage
    assistant: dict[str, Any]  # the assistant message exactly as the model sent it
    parsed: ParsedReply
    # What happened when the turn's actions ran, filled in on the next step.
    outcome: str | None = None
    # Extra text sent with this turn's screenshot (advisory hints, a URL).
    note: str = ""


class ActionFormat(ABC):
    """Prompt, tools and parser for one family of models."""

    name: str = ""

    @abstractmethod
    def system_prompt(self, ctx: PromptContext) -> str: ...

    def tools(self, ctx: PromptContext) -> list[dict[str, Any]] | None:
        """OpenAI-style function tools to send, or None for a text-only format."""
        return None

    @abstractmethod
    def parse(self, message: dict[str, Any], ctx: PromptContext) -> ParsedReply:
        """Parse one assistant message (``content`` and, for tool formats, ``tool_calls``)."""

    def replay(self, turn: Turn) -> tuple[list[dict[str, Any]], str]:
        """Messages that replay ``turn`` after its screenshot, and text to carry into the
        next user message. Text formats carry the outcome forward; tool formats answer
        each call with a tool message instead."""
        carry = f"Result of your last action: {turn.outcome}" if turn.outcome else ""
        return [{"role": "assistant", "content": _text_of(turn.assistant) or "(no reply)"}], carry


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""
