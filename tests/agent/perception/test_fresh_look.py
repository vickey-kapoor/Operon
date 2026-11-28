"""KAN-207: spot_changed and pointer_point (OPERON_FRESH_LOOK), no screen needed."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from operon.agent.perception.fresh_look import (
    CHANGED_THRESHOLD,
    REGION_RADIUS,
    pointer_point,
    spot_changed,
)
from operon.models.policy import ActionType, AgentAction


def _img(size=(400, 300), color=(40, 40, 40)) -> Image.Image:
    return Image.new("RGB", size, color)


def _with_rect(box, color=(240, 240, 240)) -> Image.Image:
    img = _img()
    ImageDraw.Draw(img).rectangle(box, fill=color)
    return img


def test_same_image_is_zero() -> None:
    assert spot_changed(_img(), _img(), 200, 150) == 0.0


def test_popup_over_the_point_is_over_threshold() -> None:
    assert spot_changed(_img(), _with_rect((150, 100, 260, 200)), 200, 150) > CHANGED_THRESHOLD


def test_change_far_from_the_point_is_zero() -> None:
    assert spot_changed(_img(), _with_rect((300, 220, 390, 290)), 50, 50) == 0.0


def test_caret_sized_change_is_under_threshold() -> None:
    caret = _with_rect((199, 140, 201, 160))
    changed = spot_changed(_img(), caret, 200, 150)
    assert changed is not None
    assert 0 < changed < CHANGED_THRESHOLD


def test_out_of_bounds_sizes_and_bad_input_are_none() -> None:
    assert spot_changed(_img(), _img(), 400, 10) is None
    assert spot_changed(_img(), _img(), 10, 300) is None
    assert spot_changed(_img(), _img((200, 100)), 10, 10) is None
    assert spot_changed(b"not an image", _img(), 10, 10) is None
    assert spot_changed("/no/such/file.png", _img(), 10, 10) is None


def test_point_near_the_edge_is_clamped() -> None:
    changed = spot_changed(_img(), _with_rect((0, 0, 30, 30)), 0, 0)
    assert changed is not None and changed > CHANGED_THRESHOLD


def test_accepts_bytes_and_paths(tmp_path) -> None:
    a, b = _img(), _with_rect((150, 100, 260, 200))
    buf = io.BytesIO()
    b.save(buf, format="PNG")
    path = tmp_path / "a.png"
    a.save(path)
    from_bytes = spot_changed(path, buf.getvalue(), 200, 150)
    assert from_bytes is not None and from_bytes > CHANGED_THRESHOLD
    assert spot_changed(str(path), path, 200, 150) == 0.0


def test_monitor_offset_is_the_callers_subtraction() -> None:
    """The frame is monitor-local: the same virtual point minus the origin finds the change."""
    changed = _with_rect((150, 100, 260, 200))
    virtual_x, virtual_y, left, top = 1200 + 200, 150, 1200, 0
    local = spot_changed(_img(), changed, virtual_x - left, virtual_y - top)
    assert local is not None and local > CHANGED_THRESHOLD
    assert spot_changed(_img(), changed, virtual_x, virtual_y) is None


def test_region_is_80_px_square() -> None:
    assert REGION_RADIUS == 40


def test_pointer_point_rules() -> None:
    for kind in (
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.TRIPLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.HOVER,
    ):
        assert pointer_point(AgentAction(action_type=kind, x=5, y=6)) == (5, 6)
    assert pointer_point(AgentAction(action_type=ActionType.DRAG, x=5, y=6, x_end=9, y_end=9)) == (5, 6)
    assert pointer_point(AgentAction(action_type=ActionType.TYPE, text="a", x=5, y=6)) == (5, 6)
    assert pointer_point(AgentAction(action_type=ActionType.TYPE, text="a")) is None
    assert pointer_point(AgentAction(action_type=ActionType.CLICK, target_element_id="e1")) is None
    assert pointer_point(AgentAction(action_type=ActionType.WAIT, wait_ms=10)) is None


def test_pointer_point_batch_uses_first_child_only() -> None:
    click = AgentAction(action_type=ActionType.CLICK, x=7, y=8)
    wait = AgentAction(action_type=ActionType.WAIT, wait_ms=10)
    assert pointer_point(AgentAction(action_type=ActionType.BATCH, actions=[click, wait])) == (7, 8)
    assert pointer_point(AgentAction(action_type=ActionType.BATCH, actions=[wait, click])) is None
