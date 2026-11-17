"""Running task note: progress memory for long tasks (KAN-102).

The loop builds one short text note from what it already holds, with no model call: the
task, the plan steps (when ``OPERON_PLANNER`` wrote a plan), the last few actions with their
verification outcome, and the step count. It is sent with each step as an advisory hint
(``source="task_note"``), the same channel as the plan and the change note.

Vision-only is intact: the note is text about what Operon did, never about the page.
It changes what the model sees, not what it does, so the thin harness gets it too.

Off by default (``OPERON_TASK_NOTE``). Rebuilt each step from the run's history, so a run
resumed after a restart gets the same note and no per-run state is kept.
"""

from __future__ import annotations

from collections.abc import Sequence

from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.verification import VerificationResult, VerificationStatus

MAX_NOTE_CHARS = 1200
_MAX_TEXT_CHARS = 40
_MAX_PLAN_STEP_CHARS = 80

_VERBS = {
    ActionType.CLICK: "clicked",
    ActionType.DOUBLE_CLICK: "double-clicked",
    ActionType.TRIPLE_CLICK: "triple-clicked",
    ActionType.RIGHT_CLICK: "right-clicked",
    ActionType.HOVER: "hovered",
}


def _short(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _where(action: AgentAction) -> str:
    if action.x is not None and action.y is not None:
        return f"({action.x},{action.y})"
    if action.target_element_id:
        return f"element {action.target_element_id}"
    return ""


def describe_action(action: AgentAction) -> str:
    """A short human-readable phrase for one action."""
    kind = action.action_type
    where = _where(action)
    if kind in _VERBS:
        return f"{_VERBS[kind]} {where}".strip()
    if kind is ActionType.TYPE:
        # The text itself stays out: it may be a password, and the note is sent and saved.
        what = f"typed {len(action.text)} chars" if action.text else "typed text"
        return f"{what} into field at {where}" if where else what
    if kind is ActionType.NAVIGATE:
        return f"opened {_short(action.url or '', 60)}".strip()
    if kind in (ActionType.PRESS_KEY, ActionType.HOTKEY):
        return f"pressed {action.key or action.text or 'a key'}"
    if kind is ActionType.SCROLL:
        direction = "sideways" if action.horizontal else ("up" if (action.scroll_amount or 0) > 0 else "down")
        return f"scrolled {direction}"
    if kind is ActionType.DRAG:
        return f"dragged from {where}".strip()
    if kind is ActionType.BATCH:
        return f"ran {len(action.actions or [])} actions in a row"
    if kind is ActionType.TOOL:
        return f"used tool {action.tool}"
    if kind is ActionType.LAUNCH_APP:
        return f"launched {_short(action.text or 'an app')}"
    if kind is ActionType.WAIT:
        return "waited"
    return f"{kind.value.replace('_', ' ')} {where}".strip()


def describe_outcome(executed: ExecutedAction, verification: VerificationResult | None) -> str:
    """The verification outcome in a few words."""
    if not executed.success:
        return "failed"
    if verification is None or verification.unchecked:
        return "done"
    status = verification.status
    if status in (VerificationStatus.SUCCESS, VerificationStatus.PROGRESSING_STABLE):
        return "success"
    if status is VerificationStatus.FAILURE:
        return "no visible change" if not verification.expected_outcome_met else "failed"
    return "unclear"


def build_task_note(
    intent: str,
    step_count: int,
    actions: Sequence[ExecutedAction],
    verifications: Sequence[VerificationResult],
    plan_steps: Sequence[str] = (),
    max_items: int = 8,
    max_chars: int = MAX_NOTE_CHARS,
) -> str:
    """The note text; "" when nothing has run yet (the first step needs no memory).

    Lines are numbered by position in the action list. When over ``max_chars``, the oldest action lines go first,
    then the plan, so the task and the latest action always stay.
    """
    if not actions:
        return ""
    recent = list(actions)[-max(1, max_items):]
    # Numbered by position in the action list: a pause bumps step_count without an action.
    first = len(actions) - len(recent) + 1
    lines = []
    for offset, executed in enumerate(recent):
        index = len(actions) - len(recent) + offset
        verification = verifications[index] if index < len(verifications) else None
        lines.append(f"{first + offset}. {describe_action(executed.action)} -> {describe_outcome(executed, verification)}")

    head = f"Task so far ({step_count} steps done): {_short(intent, 160)}"
    plan = ""
    if plan_steps:
        numbered = "; ".join(f"{i}. {_short(s, _MAX_PLAN_STEP_CHARS)}" for i, s in enumerate(plan_steps, 1))
        plan = f"Plan: {numbered}"

    def render(plan_text: str, action_lines: list[str]) -> str:
        parts = [head]
        if plan_text:
            parts.append(plan_text)
        parts.append("Last actions:\n" + "\n".join(action_lines))
        return "\n".join(parts)

    note = render(plan, lines)
    while len(note) > max_chars and len(lines) > 1:
        lines.pop(0)
        note = render(plan, lines)
    if len(note) > max_chars and plan:
        note = render("", lines)
    return note if len(note) <= max_chars else note[: max_chars - 3] + "..."
