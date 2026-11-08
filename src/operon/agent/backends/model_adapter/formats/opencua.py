"""OpenCUA pyautogui actions (``## Thought`` / ``## Action`` / ``## Code`` replies).

The model writes pyautogui calls plus ``computer.wait()`` and
``computer.terminate(status=...)``. The calls are parsed with ``ast`` and never run.
OpenCUA-7B and 32B give pixels of the smart-resized image (``coords: pixels`` with
``image: {resize: smart, factor: 28}``); checkpoints that give 0 to 1 fractions use
``coords: normalized``.
"""

from __future__ import annotations

import re
from typing import Any

from operon.agent.backends.model_adapter.actions import NeutralAction
from operon.agent.backends.model_adapter.formats._calls import (
    Call,
    CallSyntaxError,
    parse_calls,
)
from operon.agent.backends.model_adapter.formats.base import (
    ActionFormat,
    ParsedReply,
    PromptContext,
    _text_of,
)

_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:

## Thought
Your reasoning about the screen and what to do next.

## Action
One sentence describing the action.

## Code
```python
pyautogui.click(x=..., y=...)
```

Allowed code: pyautogui.click, doubleClick, tripleClick, rightClick, moveTo, dragTo, write, press, hotkey, scroll; computer.wait(); {extra}computer.terminate(status='success' or 'failure', answer='...').
Coordinates: {coords}. When the task is done, call computer.terminate(status='success')."""

_EXTRA = {
    "browser": "computer.navigate(url='...') to open a URL; ",
    "desktop": "computer.launch_app(app='...') to open an application; ",
}

_SECTION = re.compile(r"##\s*(Thought|Action|Code)\s*\n(.*?)(?=\n##\s*(?:Thought|Action|Code)\b|\Z)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def _xy(call: Call) -> dict[str, float] | None:
    first = call.args[0] if call.args else None
    if isinstance(first, (tuple, list)) and len(first) == 2:
        # pyautogui also takes the point as one (x, y) tuple.
        call = Call(call.name, [*first, *call.args[1:]], call.kwargs)
    x = call.kwargs.get("x", call.args[0] if len(call.args) > 0 and not isinstance(call.args[0], str) else None)
    y = call.kwargs.get("y", call.args[1] if len(call.args) > 1 and not isinstance(call.args[1], str) else None)
    if x is None or y is None:
        return None
    return {"x": float(x), "y": float(y)}


def _need_xy(call: Call, cursor: dict[str, float] | None) -> dict[str, float]:
    point = _xy(call) or cursor
    if point is None:
        raise CallSyntaxError(f"{call.name} needs x and y")
    return point


class OpenCUAFormat(ActionFormat):
    name = "opencua"
    system_as_user = False

    def system_prompt(self, ctx: PromptContext) -> str:
        return _PROMPT.format(extra=_EXTRA[ctx.environment], coords=ctx.mapper.describe())

    def parse(self, message: dict[str, Any], ctx: PromptContext) -> ParsedReply:
        text = _text_of(message).strip()
        sections = {name.lower(): body.strip() for name, body in _SECTION.findall(text)}
        thought = " ".join(part for part in (sections.get("thought"), sections.get("action")) if part)
        code_text = sections.get("code") or text
        fence = _FENCE.search(code_text)
        code = fence.group(1) if fence else (sections.get("code") or "")
        reply = ParsedReply(thought=thought or text[:500])
        if not code.strip():
            reply.invalid = "the reply has no code block"
            return reply
        try:
            calls = parse_calls(code)
            cursor: dict[str, float] | None = None
            for call in calls:
                action = self._convert(call, cursor)
                if action.name in {"hover", "click", "double_click", "triple_click", "right_click"}:
                    cursor = {"x": action.args["x"], "y": action.args["y"]}
                if action.name in {"done", "fail"}:
                    reply.terminal = action
                elif action.name != "_noop":
                    reply.actions.append(action)
        except (CallSyntaxError, TypeError, ValueError) as exc:
            # Wrong literal types (e.g. scroll(clicks='down')) make the reply invalid, never crash the step.
            reply.invalid = str(exc)
            return reply
        if not calls:
            reply.invalid = "the code block has no call"
        elif reply.terminal is not None and reply.actions:
            reply.invalid = "computer.terminate must be the only call in its reply"
        return reply

    @staticmethod
    def _convert(call: Call, cursor: dict[str, float] | None) -> NeutralAction:
        name = call.name.removeprefix("pyautogui.")
        kw, args = call.kwargs, call.args
        if name == "click":
            clicks = int(kw.get("clicks", 1) or 1)
            button = str(kw.get("button") or "left").lower()
            point = _need_xy(call, cursor)
            if button == "right":
                return NeutralAction("right_click", point)
            return NeutralAction({1: "click", 2: "double_click"}.get(clicks, "triple_click"), point)
        if name in {"doubleClick", "tripleClick", "rightClick", "middleClick"}:
            kind = {"doubleClick": "double_click", "tripleClick": "triple_click", "rightClick": "right_click", "middleClick": "middle_click"}[name]
            return NeutralAction(kind, _need_xy(call, cursor))
        if name == "moveTo":
            return NeutralAction("hover", _need_xy(call, None))
        if name == "dragTo":
            if cursor is None:
                raise CallSyntaxError("dragTo needs a moveTo or click before it in the same step")
            end = _need_xy(call, None)
            return NeutralAction("drag", {"start_x": cursor["x"], "start_y": cursor["y"], "end_x": end["x"], "end_y": end["y"]})
        if name in {"write", "typewrite"}:
            text = kw.get("message", args[0] if args else "")
            if isinstance(text, list):
                text = "".join(str(t) for t in text)
            text = str(text)
            return NeutralAction("type", {"text": text.rstrip("\n"), "press_enter": text.endswith("\n")})
        if name == "press":
            keys = kw.get("keys", args[0] if args else "")
            if isinstance(keys, list):
                raise CallSyntaxError("press takes one key; use hotkey for a chord")
            return NeutralAction("key", {"keys": str(keys)})
        if name == "hotkey":
            keys = list(args) or kw.get("keys") or []
            return NeutralAction("key", {"keys": "+".join(str(k) for k in keys)})
        if name in {"scroll", "hscroll"}:
            clicks = kw.get("clicks", args[0] if args else None)
            if clicks is None:
                raise CallSyntaxError(f"{name} needs an amount")
            clicks = float(clicks)
            if name == "scroll":
                direction = "up" if clicks > 0 else "down"
            else:
                direction = "right" if clicks > 0 else "left"
            payload: dict[str, Any] = {"direction": direction, "amount": max(1, min(50, round(abs(clicks))))}
            point = _xy(Call(call.name, list(args[1:3]), {k: v for k, v in kw.items() if k in {"x", "y"}}))
            if point:
                payload.update(point)
            return NeutralAction("scroll", payload)
        if name == "computer.wait":
            return NeutralAction("wait", {"seconds": float(kw.get("seconds", args[0] if args else 3) or 3)})
        if name == "computer.navigate":
            return NeutralAction("navigate", {"url": str(kw.get("url", args[0] if args else ""))})
        if name == "computer.launch_app":
            return NeutralAction("launch_app", {"app": str(kw.get("app", args[0] if args else "")).lower()})
        if name == "computer.terminate":
            status = str(kw.get("status", args[0] if args else "success")).lower()
            answer = str(kw.get("answer") or "")
            if status == "failure":
                return NeutralAction("fail", {"reason": answer or "The model ended the task as a failure."})
            return NeutralAction("done", {"answer": answer})
        raise CallSyntaxError(f"unsupported call {call.name}")
