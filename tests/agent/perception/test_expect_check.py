"""Expected-change check (KAN-116): parse, compare and note, all pure."""

import pytest

from operon.agent.perception.expect_check import (
    EXPECTATION_TEXT_CAP,
    NAVIGATION_MIN_CHANGE,
    NOTE_CAP,
    UNEXPECTED_BIG_CHANGE,
    Expectation,
    compare,
    mismatch_note,
    parse_expectation,
)
from operon.agent.perception.screen_diff import SCREEN_CHANGE_THRESHOLD
from operon.models.policy import ExpectedChange


@pytest.mark.parametrize(
    "line",
    [
        "Expect: content - results list appears",
        "expect: CONTENT: results list appears",
        "Expected : content – results list appears",
        "Expect: content — results list appears",
    ],
)
def test_parse_expect_line_variants(line):
    result = parse_expectation(f"Clicking search.\n{line}", ExpectedChange.NONE)
    assert result == Expectation(ExpectedChange.CONTENT, "results list appears", "model_text")


def test_last_expect_line_wins():
    text = "Expect: focus - box\nthinking\nExpect: dialog - a modal opens\n"
    assert parse_expectation(text).kind is ExpectedChange.DIALOG


def test_missing_line_uses_decision_field_only_when_set():
    assert parse_expectation("no line here", ExpectedChange.NAVIGATION) == Expectation(
        ExpectedChange.NAVIGATION, "", "decision_field"
    )
    assert parse_expectation("no line here", ExpectedChange.NONE) is None
    assert parse_expectation(None, ExpectedChange.NONE) is None


def test_expect_line_beats_fallback_and_text_is_capped():
    result = parse_expectation("Expect: none - " + "x" * 500, ExpectedChange.CONTENT)
    assert result.kind is ExpectedChange.NONE
    assert len(result.text) == EXPECTATION_TEXT_CAP


def test_unknown_kind_is_not_an_expectation():
    assert parse_expectation("Expect: banana - x", ExpectedChange.NONE) is None


def _e(kind):
    return Expectation(kind, "t", "model_text")


def test_compare_none_ratio_cannot_judge():
    assert compare(_e(ExpectedChange.CONTENT), None) is None


@pytest.mark.parametrize("kind", [ExpectedChange.CONTENT, ExpectedChange.DIALOG, ExpectedChange.NAVIGATION])
def test_compare_no_change_when_change_expected(kind):
    assert compare(_e(kind), SCREEN_CHANGE_THRESHOLD / 2) == "the screen did not visibly change"
    # At the threshold the screen did change.
    assert compare(_e(ExpectedChange.CONTENT), SCREEN_CHANGE_THRESHOLD) is None


def test_compare_navigation_small_change():
    assert "small part" in compare(_e(ExpectedChange.NAVIGATION), NAVIGATION_MIN_CHANGE - 0.01)
    assert compare(_e(ExpectedChange.NAVIGATION), NAVIGATION_MIN_CHANGE) is None
    assert compare(_e(ExpectedChange.CONTENT), NAVIGATION_MIN_CHANGE - 0.01) is None


@pytest.mark.parametrize("kind", [ExpectedChange.NONE, ExpectedChange.FOCUS])
def test_compare_unexpected_big_change(kind):
    assert compare(_e(kind), UNEXPECTED_BIG_CHANGE + 0.01) == "most of the screen changed"
    assert compare(_e(kind), UNEXPECTED_BIG_CHANGE) is None
    assert compare(_e(kind), 0.0) is None


def test_mismatch_note_text_and_cap():
    note = mismatch_note(_e(ExpectedChange.CONTENT), "the screen did not visibly change")
    assert note.startswith("Expected-change check: before your last action you expected content (t)")
    assert "the screen did not visibly change" in note
    long = Expectation(ExpectedChange.CONTENT, "y" * 120, "model_text")
    assert len(mismatch_note(long, "z" * 500)) == NOTE_CAP


@pytest.mark.parametrize(
    ("line", "kind", "text"),
    [
        ("**Expect:** content - list", ExpectedChange.CONTENT, "list"),
        ("Expect: **content** - list", ExpectedChange.CONTENT, "list"),
        ("`Expect: focus`", ExpectedChange.FOCUS, ""),
        ("- Expect: dialog - a **modal** opens", ExpectedChange.DIALOG, "a modal opens"),
        ("> 1. Expected: navigation - `/cart` loads", ExpectedChange.NAVIGATION, "/cart loads"),
    ],
)
def test_parse_markdown_wrapped_lines(line, kind, text):
    result = parse_expectation(line)
    assert result is not None and result.kind is kind and result.text == text


@pytest.mark.parametrize(
    "line",
    [
        "This was unexpected: none of the buttons work",
        "as expected: content loaded fine",
        "I did not expect: content here",
    ],
)
def test_parse_ignores_prose_that_only_contains_the_word(line):
    assert parse_expectation(line) is None


def test_reliable_field_makes_none_a_real_expectation():
    result = parse_expectation("no line", ExpectedChange.NONE, field_is_reliable=True)
    assert result == Expectation(ExpectedChange.NONE, "", "decision_field")
