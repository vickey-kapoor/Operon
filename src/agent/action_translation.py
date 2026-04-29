from __future__ import annotations

from typing import Any

from src.agent.fallback_backend import BackendCompatibilityError
from src.models.policy import ActionType, AgentAction, PolicyDecision


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
        amount = 800 if direction in {"down", "right"} else -800
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
        return {
            "action_type": "drag",
            "x": _scale_grid_coord(args["x"], screen_width),
            "y": _scale_grid_coord(args["y"], screen_height),
            "x_end": _scale_grid_coord(args["destination_x"], screen_width),
            "y_end": _scale_grid_coord(args["destination_y"], screen_height),
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


def translate_computer_use_action(action_payload: dict[str, Any]) -> AgentAction:
    action_type = action_payload.get("action_type")

    if action_type == "click":
        return AgentAction(
            action_type=ActionType.CLICK,
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
    from src.models.policy import ExpectedChange
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
