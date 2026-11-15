"""Neutral tool calling: OpenAI-style function tools any tool-calling model can use.

For general vision-language models served behind an OpenAI-compatible endpoint
(Qwen3-VL on vLLM, hosted chat models). The model calls one or more tools per turn
and must call ``done`` (or ``fail``) itself; a reply with no tool call is invalid.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.backends.model_adapter.actions import TERMINAL_ACTIONS, NeutralAction
from operon.agent.backends.model_adapter.formats.base import (
    ActionFormat,
    ParsedReply,
    PromptContext,
    Turn,
    _text_of,
)

_NUMBER = {"type": "number"}


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _point(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"x": _NUMBER, "y": _NUMBER, **(extra or {})}


def neutral_tools(environment: str) -> list[dict[str, Any]]:
    tools = [
        _fn("click", "Left-click at a point.", _point(), ["x", "y"]),
        _fn("double_click", "Double-click at a point.", _point(), ["x", "y"]),
        _fn("triple_click", "Triple-click at a point to select a line or paragraph.", _point(), ["x", "y"]),
        _fn("right_click", "Right-click at a point.", _point(), ["x", "y"]),
        _fn("hover", "Move the mouse to a point without clicking.", _point(), ["x", "y"]),
        _fn(
            "drag",
            "Press at the start point, move to the end point and release.",
            {"start_x": _NUMBER, "start_y": _NUMBER, "end_x": _NUMBER, "end_y": _NUMBER},
            ["start_x", "start_y", "end_x", "end_y"],
        ),
        _fn(
            "type",
            "Type text. With x and y, click there first. clear empties the field first; "
            "press_enter presses Enter after typing.",
            _point({"text": {"type": "string"}, "clear": {"type": "boolean"}, "press_enter": {"type": "boolean"}}),
            ["text"],
        ),
        _fn(
            "key",
            "Press a key or a chord, such as 'Enter', 'Escape', 'Tab', 'ctrl+a' or 'alt+Left'.",
            {"keys": {"type": "string"}},
            ["keys"],
        ),
        _fn(
            "scroll",
            "Scroll at a point (the screen centre if omitted) by a number of wheel notches.",
            _point({
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "integer", "minimum": 1, "maximum": 50},
            }),
            ["direction"],
        ),
        _fn("wait", "Wait for the screen to settle.", {"seconds": _NUMBER}, ["seconds"]),
    ]
    if environment == "desktop":
        tools.append(
            _fn(
                "launch_app",
                "Open a desktop application by name.",
                {"app": {"type": "string", "enum": sorted(APP_LAUNCH_TARGETS)}},
                ["app"],
            )
        )
    else:
        tools.append(_fn("navigate", "Open a URL in the current tab.", {"url": {"type": "string"}}, ["url"]))
    tools += [
        _fn(
            "done",
            "Finish: the task is fully done. Put any answer the task asks for in answer.",
            {"answer": {"type": "string"}},
            [],
        ),
        _fn("fail", "Give up: the task cannot be done. Say why.", {"reason": {"type": "string"}}, ["reason"]),
    ]
    return tools


_TOOL_NAMES_CACHE: dict[str, frozenset[str]] = {}


def _tool_names(environment: str) -> frozenset[str]:
    if environment not in _TOOL_NAMES_CACHE:
        _TOOL_NAMES_CACHE[environment] = frozenset(t["function"]["name"] for t in neutral_tools(environment))
    return _TOOL_NAMES_CACHE[environment]


# Tool calls some servers leave in the text when their tool parser is off (Hermes style).
_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_FORMAT_RULES = """Act only through the tools, one or more tool calls per reply. Coordinates: {coords}.
When the task is fully done, call done (with the answer if the task asks for one). If it cannot be done, call fail.
A reply without a tool call is not an action."""


class NeutralToolsFormat(ActionFormat):
    name = "neutral_tools"

    def system_prompt(self, ctx: PromptContext) -> str:
        return "You operate a computer to complete a task.\n\n" + _FORMAT_RULES.format(coords=ctx.mapper.describe())

    def tools(self, ctx: PromptContext) -> list[dict[str, Any]] | None:
        return neutral_tools(ctx.environment)

    def parse(self, message: dict[str, Any], ctx: PromptContext) -> ParsedReply:
        thought = _text_of(message).strip()
        calls = [
            (call.get("id") or "", (call.get("function") or {}).get("name"), (call.get("function") or {}).get("arguments"))
            for call in message.get("tool_calls") or []
        ]
        if not calls and thought:
            calls = []
            for match in _TEXT_TOOL_CALL.finditer(thought):
                try:
                    body = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return ParsedReply(thought=thought, invalid="a <tool_call> block is not valid JSON")
                calls.append(("", body.get("name"), body.get("arguments")))
            thought = _TEXT_TOOL_CALL.sub("", thought).strip()
        if not calls:
            return ParsedReply(thought=thought, invalid="the reply has no tool call")

        reply = ParsedReply(thought=thought)
        allowed = _tool_names(ctx.environment)
        for call_id, name, arguments in calls:
            reply.call_ids.append(call_id or f"call_{uuid.uuid4().hex[:12]}")
            if name not in allowed:
                reply.invalid = f"there is no {name!r} tool"
                return reply
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    reply.invalid = f"the arguments of {name} are not valid JSON"
                    return reply
            if not isinstance(arguments, dict):
                reply.invalid = f"the arguments of {name} must be an object"
                return reply
            action = NeutralAction(name, arguments)
            if name in TERMINAL_ACTIONS:
                reply.terminal = action
            else:
                reply.actions.append(action)
        if reply.terminal is not None and reply.actions:
            reply.invalid = f"{reply.terminal.name} must be the only tool call in its reply"
        return reply

    def replay(self, turn: Turn) -> tuple[list[dict[str, Any]], str]:
        ids = [call.get("id") for call in turn.assistant.get("tool_calls") or []]
        if not ids or not all(ids):
            # Free-text reply, or calls written in the text: replay as text.
            return super().replay(turn)
        assistant = {k: v for k, v in turn.assistant.items() if k in {"role", "content", "tool_calls"}}
        assistant["role"] = "assistant"
        outcome = turn.outcome or "OK"
        tool_messages = [
            {"role": "tool", "tool_call_id": call_id, "content": outcome if i == 0 else "OK" if outcome == "OK" else "See above."}
            for i, call_id in enumerate(ids)
        ]
        return [assistant, *tool_messages], ""
