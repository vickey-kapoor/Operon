"""UI-TARS text actions (``Thought: ... Action: click(start_box='(x,y)')``).

The action space is UI-TARS's computer-use prompt. Points arrive as
``(x,y)`` inside box tokens, as ``<point>x y</point>``, or as a two-corner box
(its centre is used). UI-TARS 1.0 uses a 0 to 1000 grid (``coords: relative_1000``);
UI-TARS 1.5 uses pixels of the smart-resized image (``coords: pixels`` with
``image: {resize: smart, factor: 28}``).
"""

from __future__ import annotations

import re
from typing import Any

from operon.agent.backends.model_adapter.actions import NeutralAction
from operon.agent.backends.model_adapter.formats._calls import (
    CallSyntaxError,
    parse_calls,
    point_of,
)
from operon.agent.backends.model_adapter.formats.base import (
    ActionFormat,
    ParsedReply,
    PromptContext,
    _text_of,
)

_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\\\', \\\\", and \\\\n in content part to ensure we can parse the content in normal python string format.
{extra_actions}
## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""

_EXTRA = {
    "browser": "navigate(url='') #Open a URL in the current tab.\n",
    "desktop": "open_app(app_name='') #Open a desktop application by name.\n",
}

_THOUGHT = re.compile(r"(?:Thought|Reflection)\s*:\s*(.*?)(?=\n\s*Action\s*:|\Z)", re.DOTALL | re.IGNORECASE)
_ACTION = re.compile(r"Action\s*:\s*(.*)\Z", re.DOTALL | re.IGNORECASE)


def _point(kwargs: dict[str, Any], *names: str) -> dict[str, float]:
    for name in names:
        if name in kwargs:
            x, y = point_of(kwargs[name])
            return {"x": x, "y": y}
    raise CallSyntaxError(f"missing {names[0]}")


def _keys(value: Any) -> str:
    return "+".join(part for part in str(value).replace("+", " ").split() if part)


def to_neutral(name: str, kwargs: dict[str, Any], args: list[Any]) -> NeutralAction:
    if args and not kwargs:
        # Positional forms some checkpoints emit, e.g. click('(1,2)') or hotkey('ctrl c').
        kwargs = {"start_box": args[0]} if name not in {"hotkey", "press", "type", "finished", "navigate"} else {"_": args[0]}
    first = next(iter(kwargs.values()), None)
    if name in {"click", "left_single"}:
        return NeutralAction("click", _point(kwargs, "start_box", "point"))
    if name == "left_double":
        return NeutralAction("double_click", _point(kwargs, "start_box", "point"))
    if name == "right_single":
        return NeutralAction("right_click", _point(kwargs, "start_box", "point"))
    if name in {"hover", "mouse_move"}:
        return NeutralAction("hover", _point(kwargs, "start_box", "point"))
    if name in {"drag", "select"}:
        start = _point(kwargs, "start_box", "start_point")
        end = _point(kwargs, "end_box", "end_point")
        return NeutralAction("drag", {"start_x": start["x"], "start_y": start["y"], "end_x": end["x"], "end_y": end["y"]})
    if name in {"hotkey", "press", "keydown"}:
        return NeutralAction("key", {"keys": _keys(kwargs.get("key", kwargs.get("hotkey", first)) or "")})
    if name == "type":
        content = str(kwargs.get("content", first) or "")
        press_enter = content.endswith("\n")
        return NeutralAction("type", {"text": content.rstrip("\n"), "press_enter": press_enter})
    if name == "scroll":
        direction = str(kwargs.get("direction") or "down").strip().lower()
        payload: dict[str, Any] = {"direction": direction, "amount": 5}
        if "start_box" in kwargs or "point" in kwargs:
            payload.update(_point(kwargs, "start_box", "point"))
        return NeutralAction("scroll", payload)
    if name == "wait":
        return NeutralAction("wait", {"seconds": 5})
    if name == "navigate":
        return NeutralAction("navigate", {"url": str(kwargs.get("url", first) or "")})
    if name == "open_app":
        return NeutralAction("launch_app", {"app": str(kwargs.get("app_name", first) or "").lower()})
    if name == "finished":
        return NeutralAction("done", {"answer": str(kwargs.get("content", first) or "")})
    if name == "call_user":
        return NeutralAction("fail", {"reason": "The model asked for the user's help (call_user)."})
    raise CallSyntaxError(f"unknown action {name!r}")


class UITarsFormat(ActionFormat):
    name = "ui_tars"
    system_as_user = True

    def system_prompt(self, ctx: PromptContext) -> str:
        return _PROMPT.format(extra_actions=_EXTRA[ctx.environment], instruction=ctx.task_prompt)

    def parse(self, message: dict[str, Any], ctx: PromptContext) -> ParsedReply:
        text = _text_of(message).strip()
        thought_match = _THOUGHT.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""
        action_match = _ACTION.search(text)
        if not action_match:
            return ParsedReply(thought=thought or text, invalid="the reply has no 'Action:' line")
        reply = ParsedReply(thought=thought)
        # Several actions may follow, one per line (or separated by blank lines).
        code = "\n".join(line.strip() for line in action_match.group(1).strip().splitlines() if line.strip())
        try:
            calls = parse_calls(code)
        except CallSyntaxError as exc:
            reply.invalid = str(exc)
            return reply
        if not calls:
            reply.invalid = "the 'Action:' line is empty"
            return reply
        for call in calls:
            try:
                action = to_neutral(call.name, call.kwargs, call.args)
            except CallSyntaxError as exc:
                reply.invalid = str(exc)
                return reply
            if action.name in {"done", "fail"}:
                reply.terminal = action
            else:
                reply.actions.append(action)
        if reply.terminal is not None and reply.actions:
            reply.invalid = f"{reply.terminal.name} must be the only action in its reply"
        return reply
