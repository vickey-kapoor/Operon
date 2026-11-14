"""KAN-154: full-harness fixes from the external audit."""

from __future__ import annotations

from pathlib import Path

from operon.agent.policy.best_of_n import HeuristicCritic, _same_action
from operon.agent.policy.coordinator import PolicyCoordinator
from operon.agent.policy.progress_tracker import action_signature
from operon.agent.policy.trust_gate import TrustGate, TrustVerdict
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.clients.gemini import PlaceholderGeminiClient
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.store.memory import FileBackedMemoryStore


def _click(x: int, y: int) -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, x=x, y=y)


def _batch(*actions: AgentAction) -> AgentAction:
    return AgentAction(action_type=ActionType.BATCH, actions=list(actions))


def _perception(tmp_path: Path, summary: str = "s", elements: list[UIElement] | None = None) -> ScreenPerception:
    return ScreenPerception(
        summary=summary, page_hint=PageHint.UNKNOWN, visible_elements=elements or [],
        capture_artifact_path=str(tmp_path / "before.png"),
    )


def test_two_different_batches_have_different_signatures() -> None:
    assert action_signature(_batch(_click(10, 10))) != action_signature(_batch(_click(400, 400)))
    assert action_signature(_batch(_click(10, 10))) == action_signature(_batch(_click(10, 10)))


def test_best_of_n_tells_apart_scrolls_and_batches() -> None:
    down = AgentAction(action_type=ActionType.SCROLL, x=1, y=1, scroll_amount=-500)
    up = AgentAction(action_type=ActionType.SCROLL, x=1, y=1, scroll_amount=500)
    assert not _same_action(down, up)
    assert not _same_action(_batch(_click(1, 1)), _batch(_click(9, 9)))


def test_the_critic_counts_a_click_in_the_middle_of_a_wide_element_as_a_hit(tmp_path: Path) -> None:
    wide = UIElement(
        element_id="search", element_type=UIElementType.INPUT, label="Search",
        x=100, y=100, width=600, height=40, is_interactable=True, confidence=0.9,
    )
    critic = HeuristicCritic()
    state = AgentState(run_id="r", intent="i", status=RunStatus.RUNNING)
    decision = PolicyDecision(action=_click(400, 120), rationale="r", confidence=0.8, active_subgoal="s")

    assert critic._element_under(400, 120, [wide]) is wide
    assert critic.score(state, _perception(tmp_path, elements=[wide]), decision) > 0


def test_the_trust_gate_sees_a_hotkeys_keys() -> None:
    gate = TrustGate(enabled=True, deny=("ctrl+w",))
    assert gate.evaluate(AgentAction(action_type=ActionType.HOTKEY, key="ctrl+w")).verdict is TrustVerdict.DENY


async def test_a_failed_action_does_not_end_the_run_on_a_keyword_seen_before_it(tmp_path: Path) -> None:
    verifier = DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())
    state = AgentState(
        run_id="r", intent="Search until the result is found", status=RunStatus.RUNNING,
        observation_history=[_perception(tmp_path, summary="The search result is shown")],
    )
    action = _click(5, 5)
    decision = PolicyDecision(action=action, rationale="r", confidence=0.8, active_subgoal="s")

    failed = await verifier.verify(state, decision, ExecutedAction(action=action, success=False, detail="missed"))

    assert failed.stop_condition_met is False


class _Delegate:
    """A Computer Use-like delegate: one decision per screen, no second sample."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.hints: list[str] = []
        self.calls = 0

    def supports_resampling(self, run_id: str = "") -> bool:
        return False

    def add_advisory_hints(self, hints, source="", run_id=""):
        self.hints.extend(hints)

    async def choose_action(self, state, perception):
        self.calls += 1
        return self.decision


async def test_a_success_stop_is_confirmed_by_the_models_own_decision_without_a_leftover_question(tmp_path: Path) -> None:
    stop = PolicyDecision(action=AgentAction(action_type=ActionType.STOP), rationale="done", confidence=0.9, active_subgoal="done")
    delegate = _Delegate(stop)
    coordinator = PolicyCoordinator(delegate=delegate, memory_store=FileBackedMemoryStore(tmp_path / "runs"))
    state = AgentState(run_id="r", intent="Send the form", status=RunStatus.RUNNING)
    rule_stop = stop.model_copy(update={"rule_name": "_task_success_stop_rule"})

    confirmed = await coordinator._confirm_success(state, _perception(tmp_path), rule_stop)

    assert confirmed is rule_stop
    assert delegate.calls == 1
    assert not any("CONFIRMATION REQUIRED" in hint for hint in delegate.hints)


def test_memory_hints_and_rule_trace_are_queued_once_per_step(tmp_path: Path) -> None:
    delegate = _Delegate(PolicyDecision(action=_click(1, 1), rationale="r", confidence=0.5, active_subgoal="s"))
    coordinator = PolicyCoordinator(delegate=delegate, memory_store=FileBackedMemoryStore(tmp_path / "runs"))
    state = AgentState(run_id="r", intent="i", status=RunStatus.RUNNING, last_rule_trace="rule X fired")

    coordinator.prepare_hints(state, None)

    assert delegate.hints.count("rule X fired") == 1


def test_a_form_with_only_text_values_submits_instead_of_scrolling_up_forever(tmp_path: Path) -> None:
    from operon.agent.policy.rules import PolicyRuleEngine

    engine = PolicyRuleEngine()
    state = AgentState(
        run_id="r",
        intent='Fill the contact form with name "Jane Doe", email "jane@example.com" and message "Hi there", then submit',
        status=RunStatus.RUNNING,
    )
    engine._form_fields_filled["r"] = {"name", "email", "message"}
    submit = UIElement(
        element_id="submit", element_type=UIElementType.BUTTON, label="Submit",
        x=300, y=400, width=100, height=30, is_interactable=True, confidence=0.9,
    )
    perception = ScreenPerception(
        summary="Contact form", page_hint=PageHint.FORM_PAGE, visible_elements=[submit],
        capture_artifact_path=str(tmp_path / "before.png"),
    )

    decision = engine._form_visible_field_fill_rule(state, perception)

    assert decision is not None
    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "submit"
