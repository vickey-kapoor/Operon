"""Best-of-N action selection — RFC 0001, Move 3 (spike).

At uncertain steps, sample N candidate decisions and pick the highest-scored one
*before* executing anything. This is a pure pre-execution selection layer: it does
no speculative execution, no rollback, and leaves the executor, verifier, and
recovery path untouched.

Disabled by default (``OPERON_BESTOFN_N`` defaults to 1 ⇒ byte-for-byte identical
to today's single-shot behavior). When a deterministic policy rule fires, every
sampled candidate is identical, so selection is a harmless no-op — Best-of-N only
does useful work on non-deterministic (LLM-driven) steps, which is exactly the
"hard step" target.

Config:
    OPERON_BESTOFN_N            int,   default 1     master switch; <= 1 disables
    OPERON_BESTOFN_CONFIDENCE   float, default 1.0   only sample when the first
                                                     decision's confidence < this
                                                     (lower it to restrict sampling
                                                     to genuinely uncertain steps)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from operon.models.perception import ScreenPerception
from operon.models.policy import AgentAction, PolicyDecision
from operon.models.state import AgentState

logger = logging.getLogger(__name__)

# An async callable that proposes one decision for the given state + perception.
ProposeFn = Callable[[AgentState, ScreenPerception], Awaitable[PolicyDecision]]


@runtime_checkable
class Critic(Protocol):
    """Scores a *proposed* decision before execution. Higher = more likely correct."""

    def score(
        self, state: AgentState, perception: ScreenPerception, decision: PolicyDecision
    ) -> float: ...


@dataclass(frozen=True)
class ScoredCandidate:
    decision: PolicyDecision
    score: float


def best_of_n_count() -> int:
    """N from ``OPERON_BESTOFN_N`` (default 1 = disabled). Clamps to >= 1."""
    try:
        return max(1, int(os.getenv("OPERON_BESTOFN_N", "1")))
    except ValueError:
        return 1


def best_of_n_confidence_ceiling() -> float:
    """Sampling only fires when the first decision's confidence is below this.

    Default 1.0 ⇒ fires whenever Best-of-N is enabled (confidence is in [0, 1]).
    """
    try:
        return float(os.getenv("OPERON_BESTOFN_CONFIDENCE", "1.0"))
    except ValueError:
        return 1.0


def _same_action(a: AgentAction, b: AgentAction) -> bool:
    """Conservative equality over the fields that define *what* an action does."""
    return (
        a.action_type == b.action_type
        and a.x == b.x
        and a.y == b.y
        and a.selector == b.selector
        and a.target_element_id == b.target_element_id
        and a.text == b.text
        and a.key == b.key
    )


class HeuristicCritic:
    """Baseline, model-free critic scoring a candidate from signals already on hand:

    - the policy's own ``confidence`` (trust the planner, but not blindly),
    - grounding quality: does a coordinate action land on a *high-confidence,
      interactable* element? (a low-confidence or non-interactable hit scores
      lower than a clean one, and empty space is penalized),
    - non-redundancy: does it repeat the immediately-preceding *failed* action?

    Deterministic and cheap. Explicitly swappable for a learned critic later
    (RFC 0001 §8 Q3).
    """

    def __init__(
        self,
        *,
        w_confidence: float = 1.0,
        w_grounding: float = 0.5,
        w_redundant: float = 0.75,
        hit_radius: int = 40,
        noninteractable_factor: float = 0.5,
    ) -> None:
        self._w_conf = w_confidence
        self._w_ground = w_grounding
        self._w_redundant = w_redundant
        self._hit_radius = hit_radius
        # Multiplier applied to grounding quality when the matched element is not
        # flagged interactable — a click there is plausible but less likely correct.
        self._noninteractable_factor = noninteractable_factor

    def score(
        self, state: AgentState, perception: ScreenPerception, decision: PolicyDecision
    ) -> float:
        action = decision.action
        return (
            self._w_conf * float(decision.confidence)
            + self._w_ground * self._grounding_plausibility(action, perception)
            - self._w_redundant * self._redundancy_penalty(action, state)
        )

    def _grounding_plausibility(self, action: AgentAction, perception: ScreenPerception) -> float:
        """Score how well a coordinate action lands on a real, usable target.

        Returns a quality score in [0, 1] when the action lands on a perceived
        element — scaled by the element's confidence and whether it is interactable,
        so the critic prefers high-confidence, clickable targets over a low-confidence
        or non-interactable hit. Returns -1 for a click on empty space, and 0
        (neutral) for non-spatial actions or when no elements are perceived.
        """
        if action.x is None or action.y is None:
            return 0.0
        elements = getattr(perception, "visible_elements", None) or []
        if not elements:
            return 0.0
        match = self._element_under(action.x, action.y, elements)
        if match is None:
            return -1.0
        quality = float(getattr(match, "confidence", 1.0) or 0.0)
        if not getattr(match, "is_interactable", True):
            quality *= self._noninteractable_factor
        return quality

    def _element_under(self, x: int, y: int, elements: list) -> object | None:
        """Return the perceived element whose center is within ``hit_radius`` of
        (x, y), or None. On overlap, prefer the higher-confidence element."""
        hits = [
            el for el in elements
            if abs(el.x - x) <= self._hit_radius and abs(el.y - y) <= self._hit_radius
        ]
        if not hits:
            return None
        return max(hits, key=lambda el: float(getattr(el, "confidence", 0.0) or 0.0))

    def _redundancy_penalty(self, action: AgentAction, state: AgentState) -> float:
        """1.0 if this repeats the immediately-preceding action that did *not*
        succeed (a thrash), else 0.0."""
        history = getattr(state, "action_history", None) or []
        if not history:
            return 0.0
        last = history[-1]
        last_action = getattr(last, "action", None)
        if last_action is None or getattr(last, "success", True):
            return 0.0
        return 1.0 if _same_action(action, last_action) else 0.0


async def select_best_of_n(
    propose: ProposeFn,
    state: AgentState,
    perception: ScreenPerception,
    *,
    n: int,
    critic: Critic,
    first: PolicyDecision | None = None,
) -> tuple[PolicyDecision, list[ScoredCandidate]]:
    """Sample up to ``n`` candidates and return ``(chosen, scored_candidates)``.

    ``first`` (if given) is reused as candidate #1 to avoid re-invoking the policy
    for a decision the caller already computed. With ``n <= 1`` this returns the
    single decision with an empty candidate list and incurs no extra policy calls.
    Ties keep the earliest (highest-priority / already-computed) candidate.
    """
    if n <= 1:
        decision = first if first is not None else await propose(state, perception)
        return decision, []

    candidates: list[ScoredCandidate] = []
    for i in range(n):
        decision = first if (i == 0 and first is not None) else await propose(state, perception)
        candidates.append(ScoredCandidate(decision, critic.score(state, perception, decision)))

    best_index = max(range(len(candidates)), key=lambda i: candidates[i].score)
    return candidates[best_index].decision, candidates
