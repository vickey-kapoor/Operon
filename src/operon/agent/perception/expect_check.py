"""Expected-change check (KAN-116): compare what the model expected with what changed.

Pure text and arithmetic, no model call and no I/O. The model is asked to write one line
``Expect: <kind> - <what you will see>`` before each action. After the action runs, the
expectation is compared with the screen-change ratio; a mismatch becomes one line of text for
the model's next step. It is only a hint: it never changes a verification result, a retry or
an action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from operon.agent.perception.screen_diff import SCREEN_CHANGE_THRESHOLD
from operon.models.policy import ActionType, ExpectedChange

# A navigation moves most of the page; less than this changed means it may not have moved on.
NAVIGATION_MIN_CHANGE = 0.05
# An action expected to change nothing (or only focus) that changed more than this surprised the model.
UNEXPECTED_BIG_CHANGE = 0.5
# Caps on the text we keep from the model and on the final hint line.
EXPECTATION_TEXT_CAP = 120
NOTE_CAP = 300

# Re-send the instruction at most this often (in steps) when the model wrote no Expect line.
EXPECT_REMINDER_EVERY = 3

# Actions that are not meant to change the screen: no expectation is checked for them.
NO_SCREEN_EFFECT_ACTIONS = frozenset(
    {
        ActionType.STOP,
        ActionType.WAIT,
        ActionType.WAIT_FOR_USER,
        ActionType.TOOL,
        ActionType.READ_CLIPBOARD,
        ActionType.WRITE_CLIPBOARD,
        ActionType.SCREENSHOT_REGION,
        ActionType.READ_TEXT,
    }
)

EXPECT_INSTRUCTION = (
    "Before each action, write one line: Expect: <none|focus|content|navigation|dialog> - "
    "<what you expect to see after it>."
)

_EXPECT_LINE = re.compile(
    r"expect(?:ed)?\s*:\s*(none|focus|content|navigation|dialog)\b[\s:\-\u2013\u2014]*(.*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Expectation:
    kind: ExpectedChange
    text: str
    source: str  # "model_text" (an Expect line) or "decision_field" (PolicyDecision.expected_change)


def parse_expectation(rationale: str | None, fallback: ExpectedChange = ExpectedChange.NONE) -> Expectation | None:
    """The model's expectation: its last ``Expect:`` line, else the decision field when set."""
    if rationale:
        for line in reversed(rationale.splitlines()):
            match = _EXPECT_LINE.search(line)
            if match:
                text = match.group(2).strip()[:EXPECTATION_TEXT_CAP].strip()
                return Expectation(ExpectedChange(match.group(1).lower()), text, "model_text")
    if fallback is not ExpectedChange.NONE:
        return Expectation(fallback, "", "decision_field")
    return None


def compare(expectation: Expectation, change_ratio: float | None) -> str | None:
    """What went against the expectation, as a short phrase, or None when it held (or can't be judged)."""
    if change_ratio is None:
        return None
    kind = expectation.kind
    if kind in (ExpectedChange.CONTENT, ExpectedChange.DIALOG, ExpectedChange.NAVIGATION):
        if change_ratio < SCREEN_CHANGE_THRESHOLD:
            return "the screen did not visibly change"
        if kind is ExpectedChange.NAVIGATION and change_ratio < NAVIGATION_MIN_CHANGE:
            return "only a small part of the screen changed, so the page may not have moved on"
        return None
    if change_ratio > UNEXPECTED_BIG_CHANGE:
        return "most of the screen changed"
    return None


def mismatch_note(expectation: Expectation, observation: str) -> str:
    """One hint line for the model's next step."""
    detail = f" ({expectation.text})" if expectation.text else ""
    note = (
        f"Expected-change check: before your last action you expected {expectation.kind.value}{detail}, "
        f"but {observation}. Check this screenshot before going on."
    )
    return note[:NOTE_CAP]
