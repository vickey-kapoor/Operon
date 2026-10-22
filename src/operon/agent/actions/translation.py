from __future__ import annotations

from typing import Any

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


def normalize_computer_use_function_call(
    function_call: dict[str, Any],
    *,
    screen_width: int,
    screen_height: int,
) -> dict[str, Any]:
    name = function_call.get("name")
    args = function_call.get("args", {})
    safety_decision = args.get("safety_decision")
    if isinstance(safety_decision, dict) and safety_decision.get("decision") == "require_confirmation":
        return {
            "action_type": "wait_for_user",
            "text": safety_decision.get(
                "explanation",
                "User confirmation is required before proceeding.",
            ),
        }

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
        # Executors scroll up for a positive amount (as scroll_at below assumes).
        amount = 800 if direction in {"up", "right"} else -800
        return {
            "action_type": "scroll",
            "x": max(screen_width // 2, 0),
            "y": max(screen_height // 2, 0),
            "scroll_amount": amount,
        }
    if name == "scroll_at":
        direction = str(args["direction"]).lower()
        axis_size = screen_height if direction in {"up", "down"} else screen_width
        magnitude = _scale_grid_magnitude(int(args.get("magnitude", 800)), axis_size)
        amount = magnitude if direction in {"up", "right"} else -magnitude
        return {
            "action_type": "scroll",
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
            "scroll_amount": amount,
        }
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
    if name in {"click", "middle_click", "double_click", "right_click", "move"}:
        action_type = {"click": "click", "middle_click": "click", "move": "hover"}.get(name, name)
        return {
            "action_type": action_type,
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
        }
    if name == "type":
        return {
            "action_type": "type",
            "text": args["text"],
            "clear_before_typing": False,
            "press_enter": bool(args.get("press_enter", False)),
        }
    if name == "press_key":
        return {"action_type": "press_key", "key": str(args["key"])}
    if name == "hotkey":
        keys = args["keys"]
        return {"action_type": "hotkey", "key": "+".join(keys) if isinstance(keys, list) else str(keys)}
    if name == "wait":
        return {"action_type": "wait", "wait_ms": int(float(args.get("seconds", 1)) * 1000)}
    if name == "take_screenshot":
        # Nothing to do on screen; the next turn carries a fresh screenshot.
        return {"action_type": "wait", "wait_ms": 250}
    if name == "scroll":
        direction = str(args["direction"]).lower()
        magnitude = int(args.get("magnitude_in_pixels", 300))
        return {
            "action_type": "scroll",
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
            "scroll_amount": magnitude if direction in {"up", "right"} else -magnitude,
        }
    raise BackendCompatibilityError(f"Unsupported Computer Use function call: {name}")


def normalize_computer_use_actions(
    function_calls: list[dict[str, Any]],
    *,
    screen_width: int,
    screen_height: int,
) -> dict[str, Any]:
    if not function_calls:
        raise BackendCompatibilityError("No Computer Use function calls were provided.")
    normalized = [
        normalize_computer_use_function_call(
            function_call,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        for function_call in function_calls
    ]
    if len(normalized) == 1:
        return normalized[0]
    return {"action_type": "batch", "actions": normalized}


# Claude's key names follow xdotool ("Return", "ctrl+s", "Page_Down"); Playwright's differ.
_CLAUDE_KEY_NAMES = {
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "shift": "Shift",
    "super": "Meta",
    "cmd": "Meta",
    "meta": "Meta",
    "return": "Enter",
    "enter": "Enter",
    "kp_enter": "Enter",
    "escape": "Escape",
    "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "tab": "Tab",
    "space": "Space",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "page_up": "PageUp",
    "prior": "PageUp",
    "page_down": "PageDown",
    "next": "PageDown",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
}

# Pixels per scroll-wheel click, as Chromium scrolls for one notch.
_CLAUDE_SCROLL_STEP_PX = 100


def _claude_key(text: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(text).split("+") if part.strip()]
    if not parts:
        raise BackendCompatibilityError("Claude key action without a key.")
    if len(parts) == 1:
        name = parts[0]
        return {"action_type": "press_key", "key": _CLAUDE_KEY_NAMES.get(name.lower(), name)}
    lowered = "+".join(part.lower() for part in parts)
    if lowered in {"alt+left", "alt+right"}:
        # The executor turns these into history navigation.
        return {"action_type": "hotkey", "key": lowered}
    return {
        "action_type": "hotkey",
        "key": "+".join(_CLAUDE_KEY_NAMES.get(part.lower(), part) for part in parts),
    }


def normalize_claude_computer_calls(
    calls: list[dict[str, Any]],
    *,
    scale_x: float,
    scale_y: float,
    screen_width: int,
    screen_height: int,
) -> dict[str, Any]:
    """Map one turn of Claude computer use calls to an executor action payload.

    Each call is ``{"name": member, "input": {...}}``. Coordinates arrive in the
    pixel space of the screenshot Claude saw and are multiplied by ``scale_x`` /
    ``scale_y`` to reach screen space.
    """

    def point(value: Any) -> tuple[int, int]:
        x, y = value
        return (
            max(0, min(screen_width - 1, round(float(x) * scale_x))),
            max(0, min(screen_height - 1, round(float(y) * scale_y))),
        )

    def centre() -> tuple[int, int]:
        return screen_width // 2, screen_height // 2

    actions: list[dict[str, Any]] = []
    for call in calls:
        name = call.get("name")
        args = call.get("input") or {}
        if name == "screenshot":
            continue  # every answer already carries a fresh screenshot
        if name in {"left_click", "middle_click", "double_click", "right_click"}:
            x, y = point(args["coordinate"]) if args.get("coordinate") else centre()
            action_type = {"left_click": "click", "middle_click": "click"}.get(name, name)
            actions.append({"action_type": action_type, "x": x, "y": y})
        elif name == "mouse_move":
            x, y = point(args["coordinate"])
            actions.append({"action_type": "hover", "x": x, "y": y})
        elif name == "left_click_drag":
            x, y = point(args["start_coordinate"])
            x_end, y_end = point(args["coordinate"])
            actions.append({"action_type": "drag", "x": x, "y": y, "x_end": x_end, "y_end": y_end})
        elif name == "type":
            typed: dict[str, Any] = {
                "action_type": "type",
                "text": args["text"],
                "clear_before_typing": False,
                "press_enter": False,
            }
            previous = actions[-1] if actions else None
            if previous is not None and previous["action_type"] == "click":
                # Focus and type in one atomic TYPE rather than a CLICK then a TYPE.
                actions.pop()
                typed.update(x=previous["x"], y=previous["y"])
            actions.append(typed)
        elif name == "key":
            repeat = max(1, min(100, int(args.get("repeat", 1) or 1)))
            actions.extend(_claude_key(args["text"]) for _ in range(repeat))
        elif name == "scroll":
            x, y = point(args["coordinate"]) if args.get("coordinate") else centre()
            direction = str(args.get("scroll_direction", "down")).lower()
            magnitude = max(1, int(args.get("scroll_amount", 3) or 3)) * _CLAUDE_SCROLL_STEP_PX
            # Executors scroll up for a positive amount.
            amount = magnitude if direction in {"up", "right"} else -magnitude
            actions.append({"action_type": "scroll", "x": x, "y": y, "scroll_amount": amount})
        elif name == "wait":
            seconds = min(float(args.get("duration", 1) or 1), 30.0)
            actions.append({"action_type": "wait", "wait_ms": int(seconds * 1000)})
        elif name == "navigate":
            actions.append({"action_type": "navigate", "url": args["url"]})
        elif name == "launch_app":
            actions.append({"action_type": "launch_app", "text": args["app"]})
        else:
            raise BackendCompatibilityError(f"Unsupported Claude computer use call: {name}")

    if not actions:
        # Only screenshots were asked for: nothing to do on screen this step.
        return {"action_type": "wait", "wait_ms": 250}
    if len(actions) == 1:
        return actions[0]
    return {"action_type": "batch", "actions": actions}


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
