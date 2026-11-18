"""The advisory hint a resumed run's model gets in place of its lost conversation (KAN-80).

A restart loses the model conversation and the browser page, so the run resumes "from the
current screen": the task, the actions done so far (from the saved state, typed text left
out) and a fresh screenshot. Text about what Operon did, never about the page, so
vision-only is intact.
"""

from __future__ import annotations

from operon.agent.task_note import describe_action, describe_outcome
from operon.models.state import AgentState

RESUME_HINT_SOURCE = "resume"
_MAX_ACTIONS = 10


def build_resume_hint(state: AgentState) -> str:
    lines = [
        "You are resuming an interrupted task: Operon restarted, so your earlier conversation is gone "
        "and the screen may have changed since your last action. Judge only from the screenshot now.",
        f"Task: {state.intent}",
    ]
    actions = state.action_history
    if actions:
        recent = actions[-_MAX_ACTIONS:]
        first = len(actions) - len(recent)
        # The saved lists are trimmed from the front, so line them up from the end.
        skew = len(actions) - len(state.verification_history)
        lines.append(f"Steps done so far ({state.step_count}); do not repeat what already worked. Last actions:")
        for offset, executed in enumerate(recent):
            index = first + offset
            slot = index - skew
            verification = state.verification_history[slot] if 0 <= slot < len(state.verification_history) else None
            lines.append(f"- {describe_action(executed.action)} -> {describe_outcome(executed, verification)}")
    else:
        lines.append("No steps were completed before the interruption.")
    return "\n".join(lines)
