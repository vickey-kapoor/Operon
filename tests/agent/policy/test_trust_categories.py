"""Named risk categories in the trust gate (KAN-103, OPERON_TRUST_CATEGORIES)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.agent.policy.trust_gate import (
    RISK_CATEGORIES,
    TrustGate,
    TrustVerdict,
    apply_permission_mode,
    make_trust_gate,
    match_reason_category,
    match_risk_category,
)
from operon.api import frontdoor
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from tests.fixtures.frontdoor import CLICK_SAVE, build_loop

ALL = tuple(RISK_CATEGORIES)


def _click(name: str | None = None, text: str | None = None, kind: str = "click"):
    ctx = SimpleNamespace(original_target=SimpleNamespace(primary_name=name)) if name is not None else None
    return SimpleNamespace(
        action_type=SimpleNamespace(value=kind), text=text, url=None, key=None, target_context=ctx
    )


@pytest.mark.parametrize(
    ("label", "category"),
    [
        ("Pay now", "payment"),
        ("PAY", "payment"),
        ("Buy now", "payment"),
        ("Submit order", "send"),  # "submit" is a send phrase; "order" alone is not payment
        ("Place   order", "payment"),
        ("Transfer", "payment"),
        ("Donate", "payment"),
        ("Send", "send"),
        ("Reply all", "send"),
        ("Publish", "send"),
        ("Delete", "delete"),
        ("Empty trash", "delete"),
        ("Delete permanently", "delete"),
        ("Uninstall", "delete"),
        ("Format disk", "delete"),
        ("Sign out of all devices", "account"),
        ("Change password", "account"),
        ("Revoke", "account"),
        ("Install", "install"),
        ("Run as administrator", "install"),
        ("Allow", "install"),
        ("Grant access", "install"),
    ],
)
def test_category_positives(label: str, category: str) -> None:
    assert match_risk_category(label, ALL)[0] == category


@pytest.mark.parametrize(
    "label",
    [
        "Payday",  # word boundary
        "Payments history",
        "Resend code",
        "Postal code",
        "Allowance",
        "Information",
        "Format",
        "Email",
        "Don't allow",
        "Do not allow",
        "Cancel payment",
        "Search",
        "Save",
        "Send us feedback about the new design",  # too long to be a button label
        "",
    ],
)
def test_category_negatives(label: str) -> None:
    assert match_risk_category(label, ALL) == ("", "")


ORDINARY_REASONS = [
    "I will click on the email field",
    "remove the price filter",
    "submit the search",
    "pay attention to the sidebar",
    "Let me change the date format dropdown",
    "I'll click the Send us feedback link in the footer of the page",
]


@pytest.mark.parametrize("reason", ORDINARY_REASONS)
def test_ordinary_reasons_never_fire_on_a_nameless_click(reason: str) -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    assert match_reason_category(reason, ALL) == ("", "")
    assert gate.evaluate(_click(), stated_reason=reason).verdict is TrustVerdict.ALLOW


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("Click 'Pay now'", "payment"),
        ("I'll click \"Delete\" to remove it", "delete"),
        ("Clicking ‘Send’", "send"),
        ("I'll click the Place order button at the bottom of the page", "payment"),
        ("Time to permanently delete the file", "delete"),
        ("Now sign out of all devices", "account"),
    ],
)
def test_reason_positives_quoted_target_or_multiword_phrase(reason: str, category: str) -> None:
    result = TrustGate(enabled=True, categories=ALL).evaluate(_click(), stated_reason=reason)
    assert (result.verdict, result.category) == (TrustVerdict.CONFIRM, category)


def test_quoted_reason_target_obeys_label_rules() -> None:
    assert match_reason_category("Click 'Payments history'", ALL) == ("", "")
    assert match_reason_category("Click 'Don't allow'", ALL) == ("", "")
    assert match_reason_category("Click 'Send us feedback about the new design'", ALL) == ("", "")


def test_enter_may_use_the_reason_but_other_keys_never_do() -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    reason = "Press the key to place order"
    enter = _click(kind="press_key")
    enter.key = "Enter"
    ctrl_a = _click(kind="hotkey")
    ctrl_a.key = "ctrl+a"
    assert gate.evaluate(enter, stated_reason=reason).category == "payment"
    assert gate.evaluate(ctrl_a, stated_reason=reason).verdict is TrustVerdict.ALLOW


def test_gate_confirms_a_category_label_with_the_category_on_the_result() -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    result = gate.evaluate(_click("Pay now"))
    assert result.verdict is TrustVerdict.CONFIRM
    assert result.category == "payment"
    assert result.matched == "pay"
    assert gate.evaluate(_click("Payments history")).verdict is TrustVerdict.ALLOW


def test_typed_text_is_not_judged_as_a_category() -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    assert gate.evaluate(_click("Message", text="please send it", kind="type")).verdict is TrustVerdict.ALLOW


def test_stated_reason_used_only_when_no_target_is_named() -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    nameless = gate.evaluate(_click(), stated_reason="Click 'Delete'")
    assert (nameless.verdict, nameless.category) == (TrustVerdict.CONFIRM, "delete")
    named = gate.evaluate(_click("Save"), stated_reason="Click 'Delete'")
    assert named.verdict is TrustVerdict.ALLOW


def test_flag_off_decisions_are_unchanged() -> None:
    off = TrustGate(enabled=True)
    for label in ("Pay now", "Send", "Delete", "Install"):
        assert off.evaluate(_click(label)).verdict is TrustVerdict.ALLOW
    # the phrase list still works exactly as before, with an empty category
    legacy = TrustGate(enabled=True, confirm=("buy now",))
    result = legacy.evaluate(_click("Buy now"))
    assert (result.verdict, result.reason, result.category) == (TrustVerdict.CONFIRM, "matched high-risk rule", "")


def test_phrase_list_wins_and_deny_still_blocks() -> None:
    gate = TrustGate(enabled=True, confirm=("pay now",), deny=("delete",), categories=ALL)
    assert gate.evaluate(_click("Pay now")).category == ""
    assert gate.evaluate(_click("Delete")).verdict is TrustVerdict.DENY


def test_gate_off_bypasses_categories() -> None:
    assert TrustGate(enabled=False, categories=ALL).evaluate(_click("Pay now")).verdict is TrustVerdict.ALLOW


def test_auto_mode_skips_category_confirmations_ask_and_risky_keep_them() -> None:
    gate = TrustGate(enabled=True, categories=ALL)
    action = _click("Delete")
    result = gate.evaluate(action)
    assert apply_permission_mode(result, action, "auto").verdict is TrustVerdict.ALLOW
    assert apply_permission_mode(result, action, "risky").verdict is TrustVerdict.CONFIRM
    assert apply_permission_mode(result, action, "ask").verdict is TrustVerdict.CONFIRM


def test_settings_flag_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_TRUST_CATEGORIES", raising=False)
    monkeypatch.delenv("OPERON_TRUST_CATEGORY_LIST", raising=False)
    assert make_trust_gate().categories == ()
    monkeypatch.setenv("OPERON_TRUST_CATEGORIES", "on")
    assert make_trust_gate().categories == ALL
    monkeypatch.setenv("OPERON_TRUST_CATEGORY_LIST", "Payment, delete, bogus")
    gate = make_trust_gate()
    assert gate.categories == ("payment", "delete")
    assert gate.evaluate(_click("Send")).verdict is TrustVerdict.ALLOW
    assert gate.evaluate(_click("Delete")).category == "delete"
    monkeypatch.setenv("OPERON_TRUST_CATEGORY_LIST", "bogus")
    assert make_trust_gate().categories == ALL


async def test_category_reaches_the_pause_and_the_waiting_payload(monkeypatch, tmp_path) -> None:
    pay = CLICK_SAVE.model_copy(
        update={"target_context": SimpleNamespace(original_target=SimpleNamespace(primary_name="Pay now"))}
    )
    loop, executor = build_loop(monkeypatch, tmp_path, [pay], gate=TrustGate(enabled=True, categories=ALL))
    run_id = (await loop.start_run(RunTaskRequest(intent="Do the task", max_steps=5))).run_id

    response = await loop.step_run(StepRequest(run_id=run_id))

    assert response.status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []
    assert loop.pending_gate_category(run_id) == "payment"
    record = await loop.run_store.get_run(run_id)
    assert record.pause_reason.startswith("Payment: approve click 'Pay now'?")
    assert frontdoor._waiting(loop, record)["category"] == "payment"

    await loop.resume_run(run_id)
    assert loop.pending_gate_category(run_id) is None


async def test_waiting_payload_has_no_category_key_for_ordinary_pauses(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = (await loop.start_run(RunTaskRequest(intent="Do the task", permission_mode="ask", max_steps=5))).run_id
    await loop.step_run(StepRequest(run_id=run_id))
    record = await loop.run_store.get_run(run_id)
    assert "category" not in frontdoor._waiting(loop, record)
