"""Subgoal-mutation helpers shared between recovery and progress tracking.

Live bug discovered while running the WebArena medium suite: every time a
recovery strategy or stalled-state detection fired, code did:

    state.current_subgoal = f"<prefix>: {state.current_subgoal}"

Consecutive fires therefore prepended the same prefix recursively
("Stalled — choose a completely different approach for: Stalled — choose a
completely different approach for: ..."). After 3-4 hits the string blew past
`MemoryRecord.subgoal`'s 200-char Pydantic cap, raising ValidationError on
the next memory write and crashing the run mid-task.

Two policies enforced together:

1. Strip the well-known prefix from the existing subgoal before re-prepending
   — so consecutive fires don't compound the wrap.
2. Cap the final string at SUBGOAL_MAX_CHARS even if the inner text is
   unexpectedly large. This is a defensive net; (1) is the real fix.
"""

from __future__ import annotations

# Mirrors MemoryRecord.subgoal's max_length=200. Centralised so both layers
# (recovery, progress) honour the same cap and a future schema change has one
# call site to update.
SUBGOAL_MAX_CHARS = 200


def wrap_subgoal(prefix: str, current: str | None, fallback: str = "current step") -> str:
    """Return a subgoal that prepends `prefix` exactly once and stays under cap.

    `prefix` should include any trailing separator the caller wants (e.g.
    ``"Try a different tactic for: "``). If `current` already starts with
    `prefix` (one or more times), the duplicates are stripped before
    re-prepending so the result is idempotent under repeated calls.
    """
    base = current or fallback
    while base.startswith(prefix):
        base = base[len(prefix):]
    wrapped = f"{prefix}{base}"
    if len(wrapped) > SUBGOAL_MAX_CHARS:
        budget = SUBGOAL_MAX_CHARS - len(prefix) - 1
        if budget < 0:
            # Pathological prefix longer than the cap. Truncate the prefix
            # itself rather than crash; the caller's fault, not the runtime's.
            return prefix[: SUBGOAL_MAX_CHARS - 1] + "…"
        wrapped = f"{prefix}{base[:budget]}…"
    return wrapped


def truncate_subgoal(value: str | None) -> str | None:
    """Defensive truncation at the schema boundary.

    Use right before passing a subgoal to a strict-validated model
    (e.g. MemoryRecord) when the upstream chain may not have used wrap_subgoal.
    """
    if value is None:
        return None
    if len(value) <= SUBGOAL_MAX_CHARS:
        return value
    return value[: SUBGOAL_MAX_CHARS - 1] + "…"
