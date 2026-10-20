from __future__ import annotations

import pytest

from operon.agent.actions.translation import (
    build_policy_decision,
    normalize_computer_use_actions,
    normalize_computer_use_function_call,
    translate_computer_use_action,
)
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.models.policy import ActionType


def test_translate_click_action() -> None:
    action = translate_computer_use_action({"action_type": "click", "x": 100, "y": 200})

    assert action.action_type is ActionType.CLICK
    assert action.x == 100
    assert action.y == 200


def test_translate_keypress_hotkey_action() -> None:
    action = translate_computer_use_action({"action_type": "keypress", "key": "ctrl+l"})

    assert action.action_type is ActionType.HOTKEY
    assert action.key == "ctrl+l"


def test_build_policy_decision_uses_defaults() -> None:
    decision = build_policy_decision(
        {
            "action": {"action_type": "wait", "wait_ms": 1500},
        }
    )

    assert decision.action.action_type is ActionType.WAIT
    assert decision.action.wait_ms == 1500
    assert decision.active_subgoal == "browser step"
    assert decision.confidence == 0.7


def test_normalize_click_at_scales_from_documented_grid() -> None:
    payload = normalize_computer_use_function_call(
        {"name": "click_at", "args": {"x": 500, "y": 250}},
        screen_width=1200,
        screen_height=800,
    )

    assert payload["action_type"] == "click"
    assert payload["x"] == 600
    assert payload["y"] == 200


def test_normalize_click_at_clamps_to_viewport_bounds() -> None:
    payload = normalize_computer_use_function_call(
        {"name": "click_at", "args": {"x": 5000, "y": -20}},
        screen_width=1200,
        screen_height=800,
    )

    assert payload["action_type"] == "click"
    assert payload["x"] == 1199
    assert payload["y"] == 0


def test_normalize_type_text_at_preserves_documented_flags() -> None:
    payload = normalize_computer_use_function_call(
        {
            "name": "type_text_at",
            "args": {"x": 100, "y": 200, "text": "hello", "press_enter": False, "clear_before_typing": False},
        },
        screen_width=1000,
        screen_height=1000,
    )

    assert payload["action_type"] == "type"
    assert payload["press_enter"] is False
    assert payload["clear_before_typing"] is False


def test_normalize_multiple_function_calls_to_batch() -> None:
    payload = normalize_computer_use_actions(
        [
            {"name": "navigate", "args": {"url": "https://example.com"}},
            {"name": "wait_5_seconds", "args": {}},
        ],
        screen_width=1200,
        screen_height=800,
    )

    assert payload["action_type"] == "batch"
    assert [item["action_type"] for item in payload["actions"]] == ["navigate", "wait"]


def test_translate_batch_action() -> None:
    action = translate_computer_use_action(
        {
            "action_type": "batch",
            "actions": [
                {"action_type": "navigate", "url": "https://example.com"},
                {"action_type": "wait", "wait_ms": 5000},
            ],
        }
    )

    assert action.action_type is ActionType.BATCH
    assert action.actions is not None
    assert [item.action_type for item in action.actions] == [ActionType.NAVIGATE, ActionType.WAIT]


def test_normalize_safety_confirmation_to_wait_for_user() -> None:
    payload = normalize_computer_use_function_call(
        {
            "name": "click_at",
            "args": {
                "x": 50,
                "y": 50,
                "safety_decision": {
                    "decision": "require_confirmation",
                    "explanation": "Please confirm this action.",
                },
            },
        },
        screen_width=1200,
        screen_height=800,
    )

    assert payload["action_type"] == "wait_for_user"
    assert payload["text"] == "Please confirm this action."


def test_translate_rejects_unknown_action_type() -> None:
    with pytest.raises(ValueError, match="Unsupported computer use action type"):
        translate_computer_use_action({"action_type": "teleport"})


def test_normalize_rejects_unknown_tool_function() -> None:
    with pytest.raises(BackendCompatibilityError, match="Unsupported Computer Use function call"):
        normalize_computer_use_function_call(
            {"name": "mystery_tool", "args": {}},
            screen_width=1200,
            screen_height=800,
        )


@pytest.mark.parametrize(
    ("function_call", "action_type", "key"),
    [
        ({"name": "go_back", "args": {}}, ActionType.HOTKEY, "alt+left"),
        ({"name": "go_forward", "args": {}}, ActionType.HOTKEY, "alt+right"),
        ({"name": "key_combination", "args": {"keys": "Control+A"}}, ActionType.HOTKEY, "Control+A"),
        ({"name": "key_combination", "args": {"keys": "Enter"}}, ActionType.PRESS_KEY, "Enter"),
    ],
)
def test_normalized_key_calls_translate_to_actions(function_call, action_type, key) -> None:
    payload = normalize_computer_use_function_call(function_call, screen_width=1000, screen_height=1000)

    action = translate_computer_use_action(payload)

    assert action.action_type is action_type
    assert action.key == key


@pytest.mark.parametrize(("direction", "sign"), [("down", -1), ("up", 1)])
def test_scroll_document_uses_the_executor_sign_convention(direction, sign) -> None:
    """Executors scroll up for a positive amount; both scroll tools must agree."""
    document = normalize_computer_use_function_call(
        {"name": "scroll_document", "args": {"direction": direction}}, screen_width=1000, screen_height=1000
    )
    at = normalize_computer_use_function_call(
        {"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": direction}}, screen_width=1000, screen_height=1000
    )

    assert document["scroll_amount"] * sign > 0
    assert at["scroll_amount"] * sign > 0


@pytest.mark.parametrize(
    ("function_call", "expected"),
    [
        ({"name": "click", "args": {"x": 500, "y": 500, "intent": "open it"}}, {"action_type": "click"}),
        ({"name": "right_click", "args": {"x": 500, "y": 500, "intent": "menu"}}, {"action_type": "right_click"}),
        ({"name": "move", "args": {"x": 500, "y": 500, "intent": "hover"}}, {"action_type": "hover"}),
        ({"name": "type", "args": {"text": "Ada", "intent": "name"}}, {"action_type": "type", "text": "Ada", "press_enter": False}),
        ({"name": "press_key", "args": {"key": "Enter", "intent": "submit"}}, {"action_type": "press_key", "key": "Enter"}),
        ({"name": "hotkey", "args": {"keys": ["Control", "a"], "intent": "select"}}, {"action_type": "hotkey", "key": "Control+a"}),
        ({"name": "wait", "args": {"seconds": 3, "intent": "load"}}, {"action_type": "wait", "wait_ms": 3000}),
        ({"name": "take_screenshot", "args": {"intent": "look"}}, {"action_type": "wait"}),
        ({"name": "scroll", "args": {"x": 500, "y": 500, "direction": "down", "magnitude_in_pixels": 400, "intent": "more"}}, {"action_type": "scroll", "scroll_amount": -400}),
        ({"name": "drag_and_drop", "args": {"start_x": 0, "start_y": 0, "end_x": 999, "end_y": 999, "intent": "move"}}, {"action_type": "drag", "x": 0, "x_end": 999}),
    ],
)
def test_gemini_3_computer_use_actions_translate(function_call, expected) -> None:
    payload = normalize_computer_use_function_call(function_call, screen_width=1000, screen_height=1000)

    assert {key: payload[key] for key in expected} == expected
    translate_computer_use_action(payload)
