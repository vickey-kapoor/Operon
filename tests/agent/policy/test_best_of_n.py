"""Tests for Best-of-N pre-execution selection (RFC 0001, Move 3)."""

from __future__ import annotations

import pytest

from operon.agent.policy.best_of_n import (
    HeuristicCritic,
    best_of_n_confidence_ceiling,
    best_of_n_count,
    select_best_of_n,
)
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.perception import (
    PageHint,
    ScreenPerception,
    UIElement,
    UIElementType,
)
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState

# ── builders ──────────────────────────────────────────────────────────────

def _click(x: int, y: int) -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, x=x, y=y)


def _wait() -> AgentAction:
    return AgentAction(action_type=ActionType.WAIT, wait_ms=1000)


def _decision(action: AgentAction, confidence: float, subgoal: str = "advance") -> PolicyDecision:
    return PolicyDecision(action=action, rationale="r", confidence=confidence, active_subgoal=subgoal)


def _element(eid: str, x: int, y: int) -> UIElement:
    return UIElement(
        element_id=eid, element_type=UIElementType.BUTTON, label=eid,
        x=x, y=y, width=40, height=20, is_interactable=True, confidence=0.9,
    )


def _perception(elements: list[UIElement], path: str = "/tmp/run/step_1/before.png") -> ScreenPerception:
    return ScreenPerception(
        summary="s", page_hint=PageHint.UNKNOWN, capture_artifact_path=path, visible_elements=elements,
    )


def _state(action_history: list[ExecutedAction] | None = None) -> AgentState:
    return AgentState(
        run_id="r1", intent="do thing", status=RunStatus.RUNNING,
        action_history=action_history or [],
    )


def _propose_from(decisions: list[PolicyDecision]):
    """Return (propose_fn, calls) where propose yields the given decisions in order."""
    it = iter(decisions)
    calls = {"n": 0}

    async def propose(state, perception):
        calls["n"] += 1
        return next(it)

    return propose, calls


# ── HeuristicCritic ───────────────────────────────────────────────────────

def test_critic_rewards_grounded_click_over_empty_space() -> None:
    critic = HeuristicCritic()
    perception = _perception([_element("btn", 100, 100)])
    state = _state()

    grounded = _decision(_click(100, 100), confidence=0.8)   # lands on the element
    empty = _decision(_click(500, 500), confidence=0.8)      # lands on nothing

    assert critic.score(state, perception, grounded) > critic.score(state, perception, empty)


def test_critic_weights_confidence_for_non_spatial_actions() -> None:
    critic = HeuristicCritic()
    perception = _perception([])
    state = _state()

    high = _decision(_wait(), confidence=0.9)
    low = _decision(_wait(), confidence=0.2)

    # WAIT has no coordinates → grounding neutral, so confidence decides.
    assert critic.score(state, perception, high) > critic.score(state, perception, low)


def test_critic_penalizes_repeating_a_failed_action() -> None:
    critic = HeuristicCritic()
    perception = _perception([_element("a", 100, 100), _element("b", 200, 200)])
    last_failed = ExecutedAction(action=_click(100, 100), success=False, detail="missed")
    state = _state(action_history=[last_failed])

    repeat = _decision(_click(100, 100), confidence=0.8)      # same as the failed action
    alternative = _decision(_click(200, 200), confidence=0.8)  # different, equally grounded

    # Both are grounded with equal confidence; the redundancy penalty breaks the tie.
    assert critic.score(state, perception, alternative) > critic.score(state, perception, repeat)


def test_critic_no_penalty_when_last_action_succeeded() -> None:
    critic = HeuristicCritic()
    perception = _perception([_element("a", 100, 100)])
    succeeded = ExecutedAction(action=_click(100, 100), success=True, detail="ok")
    state = _state(action_history=[succeeded])

    repeat = _decision(_click(100, 100), confidence=0.8)
    # Repeating a *successful* action carries no redundancy penalty.
    no_history_score = critic.score(_state(), perception, repeat)
    assert critic.score(state, perception, repeat) == no_history_score


# ── select_best_of_n ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_returns_first_without_proposing() -> None:
    first = _decision(_wait(), confidence=0.5)
    propose, calls = _propose_from([])

    chosen, candidates = await select_best_of_n(
        propose, _state(), _perception([]), n=1, critic=HeuristicCritic(), first=first,
    )

    assert chosen is first
    assert candidates == []
    assert calls["n"] == 0  # no extra policy calls when disabled


@pytest.mark.asyncio
async def test_reuses_first_and_makes_n_minus_one_calls() -> None:
    perception = _perception([_element("btn", 100, 100)])
    first = _decision(_click(100, 100), confidence=0.5)        # grounded, low conf
    best = _decision(_click(100, 100), confidence=0.9)         # grounded, high conf → winner
    worst = _decision(_click(500, 500), confidence=0.5)        # empty space
    propose, calls = _propose_from([best, worst])

    chosen, candidates = await select_best_of_n(
        propose, _state(), perception, n=3, critic=HeuristicCritic(), first=first,
    )

    assert chosen is best
    assert len(candidates) == 3
    assert calls["n"] == 2  # first reused; only N-1 fresh proposals


@pytest.mark.asyncio
async def test_tie_keeps_earliest_candidate() -> None:
    perception = _perception([])
    a = _decision(_wait(), confidence=0.7)
    b = _decision(_wait(), confidence=0.7)  # identical score
    propose, _ = _propose_from([b])

    chosen, candidates = await select_best_of_n(
        propose, _state(), perception, n=2, critic=HeuristicCritic(), first=a,
    )

    assert chosen is a  # ties resolve to the earliest (already-computed) candidate
    assert len(candidates) == 2


# ── config readers ────────────────────────────────────────────────────────

def test_best_of_n_count_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_BESTOFN_N", raising=False)
    assert best_of_n_count() == 1               # default disabled
    monkeypatch.setenv("OPERON_BESTOFN_N", "3")
    assert best_of_n_count() == 3
    monkeypatch.setenv("OPERON_BESTOFN_N", "0")
    assert best_of_n_count() == 1               # clamps to >= 1
    monkeypatch.setenv("OPERON_BESTOFN_N", "garbage")
    assert best_of_n_count() == 1               # invalid → disabled


def test_best_of_n_confidence_ceiling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_BESTOFN_CONFIDENCE", raising=False)
    assert best_of_n_confidence_ceiling() == 1.0
    monkeypatch.setenv("OPERON_BESTOFN_CONFIDENCE", "0.5")
    assert best_of_n_confidence_ceiling() == 0.5
    monkeypatch.setenv("OPERON_BESTOFN_CONFIDENCE", "garbage")
    assert best_of_n_confidence_ceiling() == 1.0
