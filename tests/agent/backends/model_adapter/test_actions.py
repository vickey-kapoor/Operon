"""Neutral actions to executor payloads, coordinate spaces and image resizing."""

from __future__ import annotations

import pytest

from operon.agent.backends.model_adapter.actions import (
    CoordinateMapper,
    InvalidAction,
    NeutralAction,
    to_payload,
)
from operon.agent.backends.model_adapter.images import smart_resize, target_size
from operon.agent.backends.model_adapter.registry import ImageRule


def _mapper(space: str = "pixels", sent: tuple[int, int] = (640, 400)) -> CoordinateMapper:
    return CoordinateMapper(space=space, sent_width=sent[0], sent_height=sent[1], screen_width=1280, screen_height=800)


@pytest.mark.parametrize(
    ("space", "point", "expected"),
    [
        ("pixels", (320, 200), (640, 400)),
        ("relative_1000", (500, 250), (640, 200)),
        ("normalized", (0.25, 0.5), (320, 400)),
        ("relative_1000", (1000, 1000), (1279, 799)),  # clamped onto the screen
        ("pixels", (-5, 9999), (0, 799)),
    ],
)
def test_coordinate_spaces_map_to_screen_pixels(space, point, expected) -> None:
    assert _mapper(space).point(*point) == expected


def test_click_then_type_becomes_one_atomic_type() -> None:
    payload = to_payload(
        [NeutralAction("click", {"x": 100, "y": 50}), NeutralAction("type", {"text": "hello", "press_enter": True})],
        _mapper(),
    )
    assert payload == {
        "action_type": "type", "text": "hello", "clear_before_typing": False, "press_enter": True, "x": 200, "y": 100,
    }


def test_several_actions_become_a_batch() -> None:
    payload = to_payload(
        [NeutralAction("key", {"keys": "ctrl+a"}), NeutralAction("scroll", {"direction": "down", "amount": 2})],
        _mapper(),
    )
    assert payload["action_type"] == "batch"
    assert payload["actions"][0] == {"action_type": "hotkey", "key": "Control+a"}
    assert payload["actions"][1] == {"action_type": "scroll", "x": 640, "y": 400, "scroll_amount": -200}


def test_key_names_follow_the_executor() -> None:
    assert to_payload([NeutralAction("key", {"keys": "Return"})], _mapper()) == {"action_type": "press_key", "key": "Enter"}
    assert to_payload([NeutralAction("key", {"keys": ["alt", "Left"]})], _mapper()) == {"action_type": "hotkey", "key": "alt+left"}


def test_click_with_right_button_is_a_right_click() -> None:
    assert to_payload([NeutralAction("click", {"x": 1, "y": 1, "button": "right"})], _mapper())["action_type"] == "right_click"


def test_screenshot_alone_is_a_short_wait() -> None:
    assert to_payload([NeutralAction("screenshot")], _mapper()) == {"action_type": "wait", "wait_ms": 250}


@pytest.mark.parametrize(
    "action",
    [
        NeutralAction("click", {"x": 1}),
        NeutralAction("click", {"x": "left", "y": 2}),
        NeutralAction("type", {"text": ""}),
        NeutralAction("scroll", {"direction": "sideways"}),
        NeutralAction("teleport", {}),
        NeutralAction("drag", {"start_x": 1, "start_y": 1}),
        NeutralAction("done", {}),
    ],
)
def test_unusable_actions_are_invalid_not_crashes(action) -> None:
    with pytest.raises(InvalidAction):
        to_payload([action], _mapper())


def test_long_edge_rule_scales_down_only() -> None:
    rule = ImageRule(resize="long_edge", max_long_edge=1280)
    assert target_size(1920, 1080, rule) == (1280, 720)
    assert target_size(800, 600, rule) == (800, 600)


def test_smart_resize_matches_qwen() -> None:
    # Qwen2.5-VL's reference implementation gives (1092, 1932) for a 1080x1920 frame
    # at factor 28 under its default 12845056 max pixels.
    assert smart_resize(1080, 1920, factor=28, min_pixels=3136, max_pixels=12845056) == (1092, 1932)
    h, w = smart_resize(1080, 1920, factor=32, min_pixels=3136, max_pixels=1310720)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1310720
    assert target_size(1920, 1080, ImageRule(resize="smart", factor=32, max_pixels=1310720)) == (w, h)
