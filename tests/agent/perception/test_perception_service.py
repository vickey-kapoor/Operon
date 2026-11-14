"""Parsing and quality branches of perception.service not covered by test_perception.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from operon.agent.perception.service import (
    PerceptionError,
    PerceptionLowQualityError,
    _fix_spaced_json,
    _low_quality_reason,
    _normalize_visible_elements,
    _strip_json_fence,
    parse_perception_output,
)
from operon.models.common import FailureCategory, StopReason
from operon.models.perception import PageHint


def _element(eid: str, kind: str = "button", *, x: int = 10, y: int = 10, w: int = 50, h: int = 20, **extra) -> dict:
    return {
        "element_id": eid,
        "element_type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "is_interactable": kind in {"button", "input", "link"},
        "confidence": 0.9,
        **extra,
    }


def _raw(elements: list[dict], **fields) -> str:
    return json.dumps(
        {"summary": "A page", "page_hint": "unknown", "confidence": 0.9, "visible_elements": elements, **fields}
    )


def _png(tmp_path: Path, width: int, height: int) -> str:
    path = tmp_path / "frame.png"
    Image.new("RGB", (width, height)).save(path)
    return str(path)


# ── fences and malformed JSON ────────────────────────────────────────────


def test_strip_json_fence_removes_a_language_fence() -> None:
    assert _strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_json_fence_tolerates_a_missing_closing_fence() -> None:
    assert _strip_json_fence('```\n{"a": 1}') == '{"a": 1}'


def test_strip_json_fence_leaves_plain_json_alone() -> None:
    assert _strip_json_fence('  {"a": 1}  ') == '{"a": 1}'


def test_fenced_output_parses() -> None:
    perception = parse_perception_output(f"```json\n{_raw([_element('ok', label='OK')])}\n```", "frame.png")
    assert [e.element_id for e in perception.visible_elements] == ["ok"]


def test_spaced_character_output_is_collapsed() -> None:
    spaced = '{"s u m m a r y": "s e a r c h", "p a g e _ h i n t": "u n k n o w n"}'
    assert json.loads(_fix_spaced_json(spaced)) == {"summary": "search", "page_hint": "unknown"}


def test_normal_output_is_not_touched_by_the_spaced_fix() -> None:
    text = '{"summary": "a b c"}'
    assert _fix_spaced_json(text) is text


def test_invalid_json_raises_perception_error() -> None:
    with pytest.raises(PerceptionError, match="not valid JSON"):
        parse_perception_output("not json", "frame.png")


def test_non_object_json_raises_perception_error() -> None:
    with pytest.raises(PerceptionError, match="JSON object"):
        parse_perception_output("[1, 2]", "frame.png")


# ── element quirks ───────────────────────────────────────────────────────


def test_element_n_key_is_recovered_as_element_id() -> None:
    parsed = {"visible_elements": [{"element_7": "element_7", "element_type": "button"}]}
    _normalize_visible_elements(parsed)
    assert parsed["visible_elements"][0] == {"element_id": "element_7", "element_type": "button"}


def test_numeric_key_is_recovered_as_missing_y() -> None:
    parsed = {"visible_elements": [{"element_id": "a", "x": 5, "120": 40}]}
    _normalize_visible_elements(parsed)
    assert parsed["visible_elements"][0] == {"element_id": "a", "x": 5, "y": 120}


def test_unknown_keys_are_dropped_and_odd_entries_skipped() -> None:
    parsed = {"visible_elements": ["junk", {"element_id": "a", "y": 1, "bogus": True}]}
    _normalize_visible_elements(parsed)
    assert parsed["visible_elements"] == ["junk", {"element_id": "a", "y": 1}]


def test_non_list_visible_elements_is_left_for_validation() -> None:
    parsed = {"visible_elements": "none"}
    _normalize_visible_elements(parsed)
    assert parsed == {"visible_elements": "none"}


# ── coordinate space ─────────────────────────────────────────────────────


def test_scale_ratio_maps_coordinates_back_to_native_pixels() -> None:
    raw = _raw([_element("b", label="Go", x=10, y=20, w=30, h=1)])

    element = parse_perception_output(raw, "frame.png", scale_ratio=0.5).visible_elements[0]

    assert (element.x, element.y, element.width, element.height) == (20, 40, 60, 2)


def test_out_of_bounds_elements_log_a_warning(tmp_path: Path, caplog) -> None:
    frame = _png(tmp_path, 100, 100)
    raw = _raw([_element("inside", label="A", x=0, y=0, w=50, h=50), _element("outside", label="B", x=80, y=0, w=50)])

    with caplog.at_level(logging.WARNING, logger="operon.agent.perception.service"):
        parse_perception_output(raw, frame)

    assert "coord_bounds VIOLATION" in caplog.text
    assert "outside" in caplog.text and "'inside'" not in caplog.text


def test_in_bounds_elements_do_not_warn(tmp_path: Path, caplog) -> None:
    frame = _png(tmp_path, 100, 100)

    with caplog.at_level(logging.WARNING, logger="operon.agent.perception.service"):
        parse_perception_output(_raw([_element("a", label="A")]), frame)
        parse_perception_output(_raw([]), frame)

    assert "VIOLATION" not in caplog.text


# ── quality gate ─────────────────────────────────────────────────────────


def _parsed(elements: list[dict], page_hint: str = "unknown"):
    return parse_perception_output(_raw(elements, page_hint=page_hint), "frame.png")


def test_low_quality_error_carries_its_category_and_stop_reason() -> None:
    error = PerceptionLowQualityError("bad")
    assert error.failure_category is FailureCategory.PERCEPTION_LOW_QUALITY
    assert error.stop_reason is StopReason.PERCEPTION_LOW_QUALITY
    assert isinstance(error, PerceptionError)


def test_no_elements_is_low_quality() -> None:
    assert _low_quality_reason(_parsed([])) == "no visible elements were returned"


def test_only_text_is_low_quality() -> None:
    assert _low_quality_reason(_parsed([_element("t", "text", text="Hello")])) == "zero usable_for_targeting elements"


def test_an_unlabeled_button_alone_is_low_quality() -> None:
    # Interactable, but nothing nearby names it, so nothing can be targeted by name.
    assert _low_quality_reason(_parsed([_element("b")])) == "zero usable candidates after salvage"


def test_a_labeled_button_is_good_enough() -> None:
    assert _low_quality_reason(_parsed([_element("b", label="Submit")])) is None


def test_a_form_page_with_label_and_inputs_is_viable() -> None:
    elements = [
        _element("name-label", "text", text="Name", x=10, y=10, w=80, h=20),
        _element("name", "input", label="Name", x=10, y=40, w=200, h=24),
        _element("email", "input", label="Email", x=10, y=80, w=200, h=24),
    ]
    perception = _parsed(elements, page_hint="form_page")
    assert perception.page_hint is PageHint.FORM_PAGE
    assert _low_quality_reason(perception) is None
