from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from operon.agent import tools as _tools
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.models.policy import ActionType, AgentAction, PolicyDecision


def _scale_grid_coord(value: int | float, size: int) -> int:
    grid_value = max(0.0, min(999.0, float(value)))
    if size <= 1:
        return 0
    return max(0, min(size - 1, round((grid_value / 999.0) * (size - 1))))


def _scale_grid_magnitude(value: int | float, size: int) -> int:
    scaled = round((max(0.0, min(999.0, float(value))) / 999.0) * max(size, 1))
    return max(1, scaled)


def _scroll_payload(direction: str, magnitude: int, x: int, y: int) -> dict[str, Any]:
    """A SCROLL payload. Executors scroll up (vertical) or right (horizontal) for a
    positive amount; left and right scroll sideways, never vertically."""
    payload: dict[str, Any] = {"action_type": "scroll", "x": x, "y": y}
    if direction in {"left", "right"}:
        payload["horizontal"] = True
        payload["scroll_amount"] = magnitude if direction == "right" else -magnitude
    else:
        payload["scroll_amount"] = magnitude if direction == "up" else -magnitude
    return payload


# Mouse calls whose meaning changes with a key held (Control-click, Shift-click,
# Control-scroll). The executors have no held-modifier mouse action, so with a key
# held these are skipped and reported back to the model, never run as plain clicks.
_MODIFIED_MOUSE_CALLS = frozenset({
    "click_at", "scroll_at", "drag_and_drop", "click", "middle_click", "double_click",
    "triple_click", "right_click", "scroll",
})


@dataclass
class KeyHold:
    """Keys the model is holding down with Gemini 3.x's ``key_down`` / ``key_up``.

    ``keys`` join the next key press, so ``key_down Control``, ``press_key a``,
    ``key_up Control`` is Control+a. ``skipped`` names mouse calls that were not run
    because a key was held (see ``_MODIFIED_MOUSE_CALLS``)."""

    keys: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def copy(self) -> KeyHold:
        return KeyHold(keys=list(self.keys))


def normalize_computer_use_function_call(
    function_call: dict[str, Any],
    *,
    screen_width: int,
    screen_height: int,
    key_hold: KeyHold | None = None,
) -> dict[str, Any]:
    """One Computer Use call as an Operon action payload.

    ``key_hold`` carries the keys the model holds down (``key_down`` / ``key_up``,
    often across turns) and is updated in place. Without it, a held key lasts only
    for this call."""
    name = function_call.get("name")
    args = function_call.get("args", {})
    hold = key_hold if key_hold is not None else KeyHold()
    held = hold.keys
    safety_decision = args.get("safety_decision")
    if isinstance(safety_decision, dict) and safety_decision.get("decision") == "require_confirmation":
        return {
            "action_type": "wait_for_user",
            "text": safety_decision.get(
                "explanation",
                "User confirmation is required before proceeding.",
            ),
        }

    if held and name in _MODIFIED_MOUSE_CALLS:
        hold.skipped.append(f"{name} with {'+'.join(held)} held")
        return {"action_type": "wait", "wait_ms": 100}
    if name == "open_web_browser":
        return {"action_type": "launch_app", "text": "browser"}
    if name == "wait_5_seconds":
        return {"action_type": "wait", "wait_ms": 5000}
    if name == "go_back":
        return {"action_type": "hotkey", "key": "alt+left"}
    if name == "go_forward":
        return {"action_type": "hotkey", "key": "alt+right"}
    if name == "search":
        return {"action_type": "navigate", "url": "https://www.google.com"}
    if name == "navigate":
        return {"action_type": "navigate", "url": args["url"]}
    if name == "click_at":
        return {
            "action_type": "click",
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
        }
    if name == "hover_at":
        return {
            "action_type": "hover",
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
        }
    if name == "type_text_at":
        return {
            "action_type": "type",
            "text": args["text"],
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
            "clear_before_typing": bool(args.get("clear_before_typing", True)),
            "press_enter": bool(args.get("press_enter", True)),
        }
    if name == "key_combination":
        keys = str(args["keys"]).strip()
        if "+" in keys:
            return {"action_type": "hotkey", "key": keys}
        return {"action_type": "press_key", "key": keys}
    if name == "scroll_document":
        direction = str(args["direction"]).lower()
        return _scroll_payload(direction, 800, max(screen_width // 2, 0), max(screen_height // 2, 0))
    if name == "scroll_at":
        direction = str(args["direction"]).lower()
        axis_size = screen_height if direction in {"up", "down"} else screen_width
        magnitude = _scale_grid_magnitude(int(args.get("magnitude", 800)), axis_size)
        return _scroll_payload(
            direction, magnitude, _scale_grid_coord(args["x"], screen_width), _scale_grid_coord(args["y"], screen_height)
        )
    if name == "drag_and_drop":
        # Gemini 2.5 names the ends x/y and destination_x/y; Gemini 3.x start_* and end_*.
        return {
            "action_type": "drag",
            "x": _scale_grid_coord(args.get("start_x", args.get("x")), screen_width),
            "y": _scale_grid_coord(args.get("start_y", args.get("y")), screen_height),
            "x_end": _scale_grid_coord(args.get("end_x", args.get("destination_x")), screen_width),
            "y_end": _scale_grid_coord(args.get("end_y", args.get("destination_y")), screen_height),
        }
    # Action set of the Gemini 3.x models' built-in computer_use tool. Every call
    # carries an ``intent`` argument, which is the model's reasoning, not a target.
    if name in {"click", "middle_click", "double_click", "triple_click", "right_click", "move"}:
        action_type = {"click": "click", "middle_click": "click", "move": "hover"}.get(name, name)
        return {
            "action_type": action_type,
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
        }
    if name == "type":
        text = args["text"]
        if held and isinstance(text, str) and len(text) == 1:
            # A single character typed with a modifier held is a shortcut (Control + a).
            return {"action_type": "hotkey", "key": "+".join([*held, text])}
        return {
            "action_type": "type",
            "text": args["text"],
            "clear_before_typing": False,
            "press_enter": bool(args.get("press_enter", False)),
        }
    if name in {"key_down", "key_up"}:
        key = str(args["key"])
        if name == "key_down" and key not in held:
            held.append(key)
        elif name == "key_up" and key in held:
            held.remove(key)
        # Nothing to run yet: the held key joins the next key press.
        return {"action_type": "wait", "wait_ms": 100}
    if name == "press_key":
        key = str(args["key"])
        if held:
            return {"action_type": "hotkey", "key": "+".join([*held, key])}
        return {"action_type": "press_key", "key": key}
    if name == "hotkey":
        keys = args["keys"]
        keys = [str(k) for k in keys] if isinstance(keys, list) else str(keys).split("+")
        return {"action_type": "hotkey", "key": "+".join([*[k for k in held if k not in keys], *keys])}
    if name == "wait":
        return {"action_type": "wait", "wait_ms": int(float(args.get("seconds", 1)) * 1000)}
    if name == "launch_app":
        # Operon's custom desktop tool: the desktop environment has no navigate.
        app = args.get("app")
        if not isinstance(app, str) or not app.strip():
            raise BackendCompatibilityError("launch_app needs an app name.")
        return {"action_type": "launch_app", "text": app.strip()}
    if name == "take_screenshot":
        # Nothing to do on screen; the next turn carries a fresh screenshot.
        return {"action_type": "wait", "wait_ms": 250}
    if name == "scroll":
        direction = str(args["direction"]).lower()
        magnitude = int(args.get("magnitude_in_pixels", 300))
        return _scroll_payload(
            direction, magnitude, _scale_grid_coord(args["x"], screen_width), _scale_grid_coord(args["y"], screen_height)
        )
    if name in _tools.TOOL_NAMES:
        # A call to one of Operon's tools (RFC 0002): run by the ToolExecutor, and its
        # output goes back to the model as the answer to this call.
        tool_args = {key: value for key, value in args.items() if key != "safety_decision"}
        return {"action_type": "tool", "tool": name, "tool_args": tool_args}
    raise BackendCompatibilityError(f"Unsupported Computer Use function call: {name}")


def normalize_computer_use_actions(
    function_calls: list[dict[str, Any]],
    *,
    screen_width: int,
    screen_height: int,
    key_hold: KeyHold | None = None,
) -> dict[str, Any]:
    if not function_calls:
        raise BackendCompatibilityError("No Computer Use function calls were provided.")
    for function_call in function_calls:
        safety_decision = (function_call.get("args") or {}).get("safety_decision")
        if isinstance(safety_decision, dict) and safety_decision.get("decision") == "require_confirmation":
            # One call in the turn needs a human: pause before any of them runs. A
            # batch would run the calls before it, and keep going after the pause.
            return normalize_computer_use_function_call(
                function_call,
                screen_width=screen_width,
                screen_height=screen_height,
            )
    # One hold for the whole batch, so a key held early in it reaches later calls.
    hold = key_hold if key_hold is not None else KeyHold()
    normalized: list[dict[str, Any]] = []
    skipped_before = len(hold.skipped)
    for function_call in function_calls:
        action = normalize_computer_use_function_call(
            function_call,
            screen_width=screen_width,
            screen_height=screen_height,
            key_hold=hold,
        )
        if hold.skipped and action["action_type"] == "wait" and len(hold.skipped) > skipped_before:
            # A skipped click leaves the screen as it was, so the calls after it (a
            # Delete for the selection it should have made) must not run either.
            # Nothing physically holds a key, so the hold ends here too.
            hold.skipped[-1] += "; the calls after it were not run and no key is held now"
            hold.keys.clear()
            normalized.append(action)
            break
        previous = normalized[-1] if normalized else None
        if (
            action["action_type"] == "type"
            and "x" not in action
            and previous is not None
            and previous["action_type"] == "click"
        ):
            # Gemini 3.x's ``type`` has no target: focus and type in one atomic TYPE
            # at the click's point rather than a CLICK then a TYPE.
            normalized.pop()
            action = {**action, "x": previous["x"], "y": previous["y"]}
        normalized.append(action)
    if len(normalized) == 1:
        return normalized[0]
    return {"action_type": "batch", "actions": normalized}


def translate_computer_use_action(action_payload: dict[str, Any]) -> AgentAction:
    action_type = action_payload.get("action_type")

    if action_type == "click":
        return AgentAction(
            action_type=ActionType.CLICK,
            x=action_payload["x"],
            y=action_payload["y"],
        )

    if action_type == "right_click":
        return AgentAction(
            action_type=ActionType.RIGHT_CLICK,
            x=action_payload["x"],
            y=action_payload["y"],
        )

    if action_type == "double_click":
        return AgentAction(
            action_type=ActionType.DOUBLE_CLICK,
            x=action_payload["x"],
            y=action_payload["y"],
        )

    if action_type == "triple_click":
        return AgentAction(
            action_type=ActionType.TRIPLE_CLICK,
            x=action_payload["x"],
            y=action_payload["y"],
        )

    if action_type == "type":
        return AgentAction(
            action_type=ActionType.TYPE,
            text=action_payload["text"],
            x=action_payload.get("x"),
            y=action_payload.get("y"),
            clear_before_typing=action_payload.get("clear_before_typing"),
            press_enter=action_payload.get("press_enter"),
            target_element_id=action_payload.get("target_element_id"),
        )

    if action_type == "batch":
        return AgentAction(
            action_type=ActionType.BATCH,
            actions=[translate_computer_use_action(item) for item in action_payload["actions"]],
        )

    if action_type == "keypress":
        key = action_payload["key"]
        if "+" in key:
            return AgentAction(action_type=ActionType.HOTKEY, key=key)
        return AgentAction(action_type=ActionType.PRESS_KEY, key=key)

    if action_type == "hotkey":
        return AgentAction(action_type=ActionType.HOTKEY, key=action_payload["key"])

    if action_type == "press_key":
        return AgentAction(action_type=ActionType.PRESS_KEY, key=action_payload["key"])

    if action_type == "scroll":
        return AgentAction(
            action_type=ActionType.SCROLL,
            x=action_payload["x"],
            y=action_payload["y"],
            scroll_amount=action_payload["scroll_amount"],
        )

    if action_type == "drag":
        return AgentAction(
            action_type=ActionType.DRAG,
            x=action_payload["x"],
            y=action_payload["y"],
            x_end=action_payload["x_end"],
            y_end=action_payload["y_end"],
        )

    if action_type == "wait":
        return AgentAction(
            action_type=ActionType.WAIT,
            wait_ms=action_payload.get("wait_ms", 1000),
        )

    if action_type == "hover":
        return AgentAction(
            action_type=ActionType.HOVER,
            x=action_payload["x"],
            y=action_payload["y"],
        )

    if action_type == "launch_app":
        return AgentAction(
            action_type=ActionType.LAUNCH_APP,
            text=action_payload["text"],
        )

    if action_type == "navigate":
        return AgentAction(
            action_type=ActionType.NAVIGATE,
            url=action_payload["url"],
        )

    if action_type == "tool":
        name = action_payload["tool"]
        tool_args = dict(action_payload.get("tool_args") or {})
        url = tool_args.get("url") if name == _tools.HTTP_GET else None
        return AgentAction(
            action_type=ActionType.TOOL,
            tool=name,
            tool_args=tool_args,
            text=_tools.summary(name, tool_args),
            url=url if isinstance(url, str) and url.strip() else None,
        )

    if action_type == "wait_for_user":
        return AgentAction(
            action_type=ActionType.WAIT_FOR_USER,
            text=action_payload["text"],
        )

    if action_type == "stop":
        return AgentAction(action_type=ActionType.STOP)

    raise ValueError(f"Unsupported computer use action type: {action_type}")


def build_policy_decision(response_payload: dict[str, Any]) -> PolicyDecision:
    from operon.models.policy import ExpectedChange
    action = translate_computer_use_action(response_payload["action"])
    raw_ec = response_payload.get("expected_change", "none")
    try:
        expected_change = ExpectedChange(raw_ec)
    except ValueError:
        expected_change = ExpectedChange.NONE
    return PolicyDecision(
        action=action,
        rationale=response_payload.get("rationale", "computer_use"),
        confidence=float(response_payload.get("confidence", 0.7)),
        active_subgoal=response_payload.get("active_subgoal", "browser step"),
        expected_change=expected_change,
    )
