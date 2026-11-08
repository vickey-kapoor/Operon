"""Neutral actions: the one action vocabulary every parser produces.

A parser turns a model's reply into ``NeutralAction`` items with coordinates still
in the model's own space; ``to_payload`` maps them to the executor's action payload
(the shape ``build_policy_decision`` takes), converting coordinates to screen pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from operon.agent.actions.translation import key_action
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.agent.backends.model_adapter.registry import CoordSpace

# Every neutral action name. ``done`` and ``fail`` end the run.
NEUTRAL_ACTIONS = frozenset({
    "click", "double_click", "triple_click", "right_click", "middle_click", "hover", "drag",
    "type", "key", "scroll", "navigate", "launch_app", "wait", "screenshot", "done", "fail",
})
TERMINAL_ACTIONS = frozenset({"done", "fail"})

# Pixels per scroll notch, as Chromium scrolls for one wheel click.
SCROLL_STEP_PX = 100


# Key names other models use (pyautogui, UI-TARS, browsers) that the executor's
# xdotool-style names spell differently.
_KEY_ALIASES = {
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "pageup": "page_up", "pagedown": "page_down", "pgup": "page_up", "pgdn": "page_down",
    "win": "super", "windows": "super", "command": "cmd", "option": "alt", "del": "delete",
    "ctrlleft": "ctrl", "ctrlright": "ctrl", "shiftleft": "shift", "shiftright": "shift",
    "altleft": "alt", "altright": "alt",
}


class InvalidAction(ValueError):
    """A reply, or one action in it, that cannot be turned into an executor action."""


@dataclass(frozen=True, slots=True)
class NeutralAction:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoordinateMapper:
    """Maps a model's coordinates to screen pixels.

    ``pixels``: pixels of the image the model was sent (``sent_*``).
    ``relative_1000``: 0 to 1000 across each axis (Qwen3-VL, UI-TARS 1.0).
    ``normalized``: 0 to 1 across each axis."""

    space: CoordSpace
    sent_width: int
    sent_height: int
    screen_width: int
    screen_height: int

    def point(self, x: Any, y: Any) -> tuple[int, int]:
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError) as exc:
            raise InvalidAction(f"coordinates must be numbers, got ({x!r}, {y!r})") from exc
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise InvalidAction(f"coordinates must be finite numbers, got ({x!r}, {y!r})")
        if self.space == "relative_1000":
            sx, sy = fx / 1000.0 * self.screen_width, fy / 1000.0 * self.screen_height
        elif self.space == "normalized":
            sx, sy = fx * self.screen_width, fy * self.screen_height
        else:
            sx = fx * self.screen_width / (self.sent_width or self.screen_width)
            sy = fy * self.screen_height / (self.sent_height or self.screen_height)
        return (
            max(0, min(self.screen_width - 1, round(sx))),
            max(0, min(self.screen_height - 1, round(sy))),
        )

    def centre(self) -> tuple[int, int]:
        return self.screen_width // 2, self.screen_height // 2

    def describe(self) -> str:
        """The coordinate rule as the model is told it."""
        if self.space == "relative_1000":
            return "x and y are relative, from 0 to 1000 across the screenshot's width and height"
        if self.space == "normalized":
            return "x and y are fractions from 0 to 1 of the screenshot's width and height"
        return f"x and y are pixels of the {self.sent_width}x{self.sent_height} screenshot, origin top left"


def _xy(args: dict[str, Any], mapper: CoordinateMapper, *, required: bool) -> tuple[int, int] | None:
    x, y = args.get("x"), args.get("y")
    if x is None or y is None:
        if required:
            raise InvalidAction("this action needs x and y")
        return None
    return mapper.point(x, y)


def to_payload(actions: list[NeutralAction], mapper: CoordinateMapper) -> dict[str, Any]:
    """The executor payload for one turn's non-terminal actions (a batch when several)."""
    payloads: list[dict[str, Any]] = []
    for action in actions:
        name, args = action.name, action.args or {}
        if name not in NEUTRAL_ACTIONS:
            raise InvalidAction(f"unknown action {name!r}")
        if name in TERMINAL_ACTIONS:
            raise InvalidAction(f"{name} must be the only action in its turn")
        if name == "screenshot":
            continue  # every turn already ends with a fresh screenshot
        if name in {"click", "double_click", "triple_click", "right_click", "middle_click"}:
            x, y = _xy(args, mapper, required=True)
            button = str(args.get("button") or "left").lower()
            if name == "middle_click" or button in {"middle", "wheel"}:
                # The executors have no middle button; a left click would do something else.
                raise InvalidAction("middle click is not supported; use click")
            kind = name
            if name == "click" and button == "right":
                kind = "right_click"
            payloads.append({"action_type": kind, "x": x, "y": y})
        elif name == "hover":
            x, y = _xy(args, mapper, required=True)
            payloads.append({"action_type": "hover", "x": x, "y": y})
        elif name == "drag":
            try:
                x, y = mapper.point(args["start_x"], args["start_y"])
                x_end, y_end = mapper.point(args["end_x"], args["end_y"])
            except KeyError as exc:
                raise InvalidAction("drag needs start_x, start_y, end_x and end_y") from exc
            payloads.append({"action_type": "drag", "x": x, "y": y, "x_end": x_end, "y_end": y_end})
        elif name == "type":
            text = args.get("text")
            if text == "" and args.get("press_enter"):
                # A bare "\n" (how UI-TARS and OpenCUA submit) is an Enter key press.
                payloads.append(key_action("Enter"))
                continue
            if not isinstance(text, str) or not text:
                raise InvalidAction("type needs non-empty text")
            typed: dict[str, Any] = {
                "action_type": "type",
                "text": text,
                "clear_before_typing": bool(args.get("clear", False)),
                "press_enter": bool(args.get("press_enter", False)),
            }
            point = _xy(args, mapper, required=False)
            previous = payloads[-1] if payloads else None
            clicked_here = (
                previous is not None
                and previous["action_type"] == "click"
                and (point is None or (abs(point[0] - previous["x"]) <= 2 and abs(point[1] - previous["y"]) <= 2))
            )
            if clicked_here:
                # Focus and type in one atomic TYPE, never a CLICK then a TYPE on the same field.
                payloads.pop()
                typed.update(x=previous["x"], y=previous["y"])
            elif point is not None:
                typed.update(x=point[0], y=point[1])
            payloads.append(typed)
        elif name == "key":
            keys = args.get("keys")
            if isinstance(keys, list):
                keys = "+".join(str(k) for k in keys)
            if isinstance(keys, str):
                keys = "+".join(_KEY_ALIASES.get(part.strip().lower(), part.strip()) for part in keys.split("+"))
            if not isinstance(keys, str) or not keys.strip():
                raise InvalidAction("key needs keys, such as 'Enter' or 'ctrl+a'")
            try:
                payloads.append(key_action(keys))
            except BackendCompatibilityError as exc:
                raise InvalidAction(str(exc)) from exc
        elif name == "scroll":
            point = _xy(args, mapper, required=False) or mapper.centre()
            direction = str(args.get("direction") or "down").lower()
            if direction in {"left", "right"}:
                # The executors scroll vertically only; running it would scroll the wrong way.
                raise InvalidAction("horizontal scroll is not supported; scroll up or down, or use the keyboard")
            if direction not in {"up", "down"}:
                raise InvalidAction(f"scroll direction must be up or down, got {direction!r}")
            try:
                notches = max(1, min(50, int(float(args.get("amount") or 3))))
            except (TypeError, ValueError) as exc:
                raise InvalidAction("scroll amount must be a number of wheel notches") from exc
            magnitude = notches * SCROLL_STEP_PX
            # Executors scroll up for a positive amount. 100 per notch suits both: Chromium
            # scrolls ~100 px per notch, and pyautogui on Windows sends WHEEL_DELTA units (120 a notch).
            amount = magnitude if direction == "up" else -magnitude
            payloads.append({"action_type": "scroll", "x": point[0], "y": point[1], "scroll_amount": amount})
        elif name == "navigate":
            url = args.get("url")
            if not isinstance(url, str) or not url.strip():
                raise InvalidAction("navigate needs a url")
            payloads.append({"action_type": "navigate", "url": url.strip()})
        elif name == "launch_app":
            app = args.get("app")
            if not isinstance(app, str) or not app.strip():
                raise InvalidAction("launch_app needs an app name")
            payloads.append({"action_type": "launch_app", "text": app.strip()})
        elif name == "wait":
            try:
                seconds = min(max(float(args.get("seconds") or 1), 0.1), 30.0)
            except (TypeError, ValueError) as exc:
                raise InvalidAction("wait seconds must be a number") from exc
            payloads.append({"action_type": "wait", "wait_ms": int(seconds * 1000)})
    if not payloads:
        return {"action_type": "wait", "wait_ms": 250}
    if len(payloads) == 1:
        return payloads[0]
    return {"action_type": "batch", "actions": payloads}
