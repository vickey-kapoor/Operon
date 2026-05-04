"""Idempotency + length cap for subgoal mutation helpers.

Live regression: recovery and stalled-state code prepended a prefix on every
fire — consecutive fires recursively wrapped the same prefix, growing the
subgoal past MemoryRecord's 200-char Pydantic cap and crashing the run on the
next memory write. wrap_subgoal() is the dedupe + cap that prevents that.
"""

from __future__ import annotations

from src.agent.subgoal_utils import SUBGOAL_MAX_CHARS, truncate_subgoal, wrap_subgoal

# ---------------------------------------------------------------------------
# wrap_subgoal — idempotency
# ---------------------------------------------------------------------------

def test_wrap_subgoal_prepends_when_prefix_absent():
    assert wrap_subgoal("Stalled — try X for: ", "click submit") == "Stalled — try X for: click submit"


def test_wrap_subgoal_does_not_double_prepend_existing_prefix():
    """Single fire should not stack: pre-existing 'Stalled — ...' prefix is stripped first."""
    prefix = "Stalled — try X for: "
    once = wrap_subgoal(prefix, "click submit")
    twice = wrap_subgoal(prefix, once)
    assert twice == once  # idempotent


def test_wrap_subgoal_strips_three_recursive_prefixes():
    """Real-world failure mode: three consecutive stalls had nested the prefix 3x."""
    prefix = "Stalled — try X for: "
    nested = prefix + prefix + prefix + "click submit"
    assert wrap_subgoal(prefix, nested) == prefix + "click submit"


def test_wrap_subgoal_uses_fallback_when_current_is_none():
    assert wrap_subgoal("Try: ", None) == "Try: current step"
    assert wrap_subgoal("Try: ", None, fallback="initial") == "Try: initial"


def test_wrap_subgoal_uses_fallback_when_current_is_empty():
    assert wrap_subgoal("Try: ", "") == "Try: current step"


def test_wrap_subgoal_caps_at_subgoal_max_chars():
    long_inner = "x" * 500
    out = wrap_subgoal("Prefix: ", long_inner)
    assert len(out) <= SUBGOAL_MAX_CHARS
    assert out.startswith("Prefix: ")
    assert out.endswith("…")


def test_wrap_subgoal_handles_pathological_long_prefix():
    """A prefix longer than the cap shouldn't crash — it gets clipped."""
    huge_prefix = "P" * 250
    out = wrap_subgoal(huge_prefix, "anything")
    assert len(out) <= SUBGOAL_MAX_CHARS


# ---------------------------------------------------------------------------
# truncate_subgoal — schema-boundary safety net
# ---------------------------------------------------------------------------

def test_truncate_subgoal_passes_short_strings_through():
    assert truncate_subgoal("short subgoal") == "short subgoal"


def test_truncate_subgoal_passes_none_through():
    assert truncate_subgoal(None) is None


def test_truncate_subgoal_truncates_at_max_chars():
    long = "x" * 500
    out = truncate_subgoal(long)
    assert len(out) <= SUBGOAL_MAX_CHARS
    assert out.endswith("…")


def test_truncate_subgoal_passes_at_exactly_max_chars():
    """Boundary: at exactly max chars, leave alone — no ellipsis added."""
    exact = "x" * SUBGOAL_MAX_CHARS
    assert truncate_subgoal(exact) == exact


# ---------------------------------------------------------------------------
# Integration: real Pydantic schema accepts wrap_subgoal output across many
# consecutive fires (the actual crash scenario).
# ---------------------------------------------------------------------------

def test_wrap_subgoal_output_validates_against_memory_record_schema():
    """A pre-fix run had this exact recursive nesting trigger ValidationError.
    Each fire must produce a string short enough for MemoryRecord(subgoal=...)."""
    from src.models.memory import MemoryOutcome, MemoryRecord

    subgoal = "anchor_recheck:(0,0)→1"
    prefixes = [
        "Stalled — choose a completely different approach for: ",
        "Try a different tactic for: ",
        "Reset the local page context, then continue: ",
        "Restart the session context, then continue: ",
    ]
    # 10 consecutive fires alternating across the four prefixes — pre-fix this
    # would have produced a 1000+ char string and raised ValidationError on the
    # second iteration of the inner loop.
    for _ in range(10):
        for prefix in prefixes:
            subgoal = wrap_subgoal(prefix, subgoal)
            # Schema-validates without raising
            record = MemoryRecord(
                key="test",
                benchmark="test",
                hint="test",
                outcome=MemoryOutcome.FAILURE,
                subgoal=subgoal,
                success=False,
            )
            assert record.subgoal is not None
            assert len(record.subgoal) <= SUBGOAL_MAX_CHARS
