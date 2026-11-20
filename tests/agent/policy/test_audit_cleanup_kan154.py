"""KAN-154: fixes from the external audit that outlive the Full harness."""

from __future__ import annotations

from operon.agent.policy.progress_tracker import action_signature
from operon.agent.policy.trust_gate import TrustGate, TrustVerdict
from operon.models.policy import ActionType, AgentAction


def _click(x: int, y: int) -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, x=x, y=y)


def _batch(*actions: AgentAction) -> AgentAction:
    return AgentAction(action_type=ActionType.BATCH, actions=list(actions))


def test_two_different_batches_have_different_signatures() -> None:
    assert action_signature(_batch(_click(10, 10))) != action_signature(_batch(_click(400, 400)))
    assert action_signature(_batch(_click(10, 10))) == action_signature(_batch(_click(10, 10)))


def test_the_trust_gate_sees_a_hotkeys_keys() -> None:
    gate = TrustGate(enabled=True, deny=("ctrl+w",))
    assert gate.evaluate(AgentAction(action_type=ActionType.HOTKEY, key="ctrl+w")).verdict is TrustVerdict.DENY


def test_a_click_the_model_says_places_an_order_asks_for_approval() -> None:
    """KAN-153: a Computer Use click has no element name; the model's own words do."""
    gate = TrustGate(enabled=True, confirm=("place order",))

    said = gate.evaluate(_click(5, 5), stated_reason="I'll click Place order to finish checkout.")
    quiet = gate.evaluate(_click(5, 5), stated_reason="I'll open the cart.")
    typed = gate.evaluate(
        AgentAction(action_type=ActionType.TYPE, text="hello", x=1, y=1), stated_reason="before I place order"
    )

    assert said.verdict is TrustVerdict.CONFIRM and said.matched == "place order"
    assert quiet.verdict is TrustVerdict.ALLOW
    assert typed.verdict is TrustVerdict.ALLOW  # typed text is judged by itself


def test_the_models_words_never_block_a_step_on_their_own() -> None:
    gate = TrustGate(enabled=True, deny=("delete account",))

    result = gate.evaluate(_click(5, 5), stated_reason="This will delete account data.")

    assert result.verdict is TrustVerdict.CONFIRM
