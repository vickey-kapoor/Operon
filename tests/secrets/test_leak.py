"""KAN-103 leak test: a scripted run types ``{{secret:pw}}`` and the value must appear in no
file of the run directory, no log line and no prompt sent to the model."""

from __future__ import annotations

import logging

import pytest

from evals import harness
from evals.harness import (
    Element,
    Scenario,
    Screen,
    Transition,
    cu_call,
    cu_text,
    run_scenario,
    types,
)
from operon.agent.policy.trust_gate import (
    PermissionMode,
    TrustGate,
    TrustVerdict,
    apply_permission_mode,
)
from operon.core import vault
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction

SECRET = "hunter2-SECRET"
PASSWORD = Element("password", "Password", 40, 100, width=300, element_type="input")


class _TypingExecutor(harness.FakeExecutor):
    """The fake screen's executor, doing what the real executors do for a TYPE: swap in the
    value in a local, type it, and report with the action untouched."""

    instances: list[_TypingExecutor] = []

    def __init__(self, world) -> None:
        super().__init__(world)
        self.typed: list[str] = []
        _TypingExecutor.instances.append(self)

    async def execute(self, action: AgentAction) -> ExecutedAction:
        if action.action_type is ActionType.TYPE and action.text is not None:
            try:
                text, _values = vault.resolve_text(action.text)
            except vault.SecretError as exc:
                return ExecutedAction(action=action, success=False, detail=f"type failed: {exc}")
            self.typed.append(text)
            self.actions.append(action)
            changed = self._world.apply(action)
            return ExecutedAction(
                action=action,
                success=True,
                detail=f"Typed '{action.text}' ({'changed' if changed else 'no change'})",
                artifact_path=self._world.write_frame(),
            )
        return await super().execute(action)


def _scenario(env: dict[str, str]) -> Scenario:
    login = Screen("login", "Sign in page with a password field", (PASSWORD,))
    done = Screen("done", "Signed in", (), goal=True)
    x, y = PASSWORD.center
    return Scenario(
        name="secret_login",
        intent="Sign in with my saved password",
        backend="computer_use",
        screens=[login, done],
        transitions=[Transition("login", types("{{secret:pw}}"), "done")],
        model={
            "login": cu_call("type_text_at", x=x, y=y, text="{{secret:pw}}", press_enter=False),
            "done": cu_text("I signed in. The task is complete."),
        },
        on_pause="approve",
        env=env,
    )


def _all_bytes(root):
    for path in root.rglob("*"):
        if path.is_file():
            yield path, path.read_bytes()


async def test_the_value_never_reaches_any_record(vault_dir, tmp_path, monkeypatch, caplog) -> None:
    vault.set_secret("pw", SECRET)
    monkeypatch.setattr(harness, "FakeExecutor", _TypingExecutor)
    _TypingExecutor.instances.clear()
    caplog.set_level(logging.DEBUG)
    root = tmp_path / "run"

    result = await run_scenario(_scenario({"OPERON_SECRETS": "on"}), root=root)

    # The run did what it should: it paused for approval (typing a secret is risky), was
    # approved, and the executor typed the real value.
    assert result.status == "succeeded", result.summary()
    assert result.pauses >= 1
    assert _TypingExecutor.instances[-1].typed == [SECRET]

    files = list(_all_bytes(root))
    assert any(p.name == "run.jsonl" or p.suffix == ".json" for p, _ in files), "no run files were written"
    leaks = [str(p) for p, data in files if SECRET.encode() in data or SECRET.encode("utf-16-le") in data]
    assert leaks == []
    # The placeholder is what was recorded.
    assert any(b"{{secret:pw}}" in data for _, data in files)

    assert SECRET not in caplog.text
    assert SECRET not in str(result.model_requests)
    assert SECRET not in result.summary() and SECRET not in repr(result)
    # The model was told the NAME and the syntax.
    assert "Secrets are available by name: pw" in str(result.model_requests)


async def test_flag_off_the_placeholder_fails_the_step_and_nothing_is_typed(vault_dir, tmp_path, monkeypatch) -> None:
    vault.set_secret("pw", SECRET)
    monkeypatch.setenv("OPERON_SECRETS", "off")
    monkeypatch.setattr(harness, "FakeExecutor", _TypingExecutor)
    _TypingExecutor.instances.clear()
    root = tmp_path / "off"
    result = await run_scenario(_scenario({"OPERON_SECRETS": "off"}), root=root)
    assert _TypingExecutor.instances[-1].typed == []
    assert result.final_screen == "login"
    assert "Secrets are available" not in str(result.model_requests)
    assert not [p for p, data in _all_bytes(root) if SECRET.encode() in data]


# -- the trust gate -----------------------------------------------------------------------------


def _typed() -> AgentAction:
    return AgentAction(action_type=ActionType.TYPE, text="{{secret:pw}}", x=1, y=1)


def test_typing_a_secret_needs_approval(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "on")
    gate = TrustGate(enabled=True)
    verdict = gate.evaluate(_typed())
    assert verdict.verdict is TrustVerdict.CONFIRM and verdict.category == "account"
    batch = AgentAction(action_type=ActionType.BATCH, actions=[_typed()])
    assert gate.evaluate(batch).verdict is TrustVerdict.CONFIRM
    assert gate.evaluate(AgentAction(action_type=ActionType.TYPE, text="hello", x=1, y=1)).verdict is TrustVerdict.ALLOW


def test_modes_apply_to_the_secret_confirm(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "on")
    result = TrustGate(enabled=True).evaluate(_typed())
    assert apply_permission_mode(result, _typed(), PermissionMode.RISKY).verdict is TrustVerdict.CONFIRM
    assert apply_permission_mode(result, _typed(), PermissionMode.ASK).verdict is TrustVerdict.CONFIRM
    # Auto mode may not skip it: typing a secret always needs a person.
    assert apply_permission_mode(result, _typed(), PermissionMode.AUTO).verdict is TrustVerdict.CONFIRM


def test_other_confirms_are_still_skipped_in_auto_mode() -> None:
    from operon.agent.policy.trust_gate import TrustResult

    result = TrustResult(TrustVerdict.CONFIRM, "matched high-risk rule", "delete account")
    assert apply_permission_mode(result, _typed(), PermissionMode.AUTO).verdict is TrustVerdict.ALLOW


def test_an_allowlisted_page_does_not_need_approval(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "on")
    gate = TrustGate(enabled=True, allow_domains=("bank.example",))
    on_list = gate.evaluate(_typed(), current_url="https://login.bank.example/signin")
    assert on_list.verdict is TrustVerdict.ALLOW
    # Off the list the gate denies the page outright; with no page (a desktop app) it confirms.
    assert gate.evaluate(_typed(), current_url="https://evil.example/").verdict is TrustVerdict.DENY
    assert gate.evaluate(_typed(), current_url=None).verdict is TrustVerdict.CONFIRM
    assert "page:" not in gate.evaluate(_typed(), current_url=None).reason
    assert gate.evaluate(_typed(), current_url="about:blank").verdict is TrustVerdict.CONFIRM
    # No allowlist configured: an https page does not help.
    open_gate = TrustGate(enabled=True)
    assert open_gate.evaluate(_typed(), current_url="https://bank.example/").verdict is TrustVerdict.CONFIRM


def test_flag_off_the_gate_does_not_look(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "off")
    assert TrustGate(enabled=True).evaluate(_typed()).verdict is TrustVerdict.ALLOW


def test_with_the_trust_gate_off_a_secret_type_fails(vault_dir, monkeypatch) -> None:
    vault.set_secret("pw", SECRET)
    monkeypatch.setenv("OPERON_TRUST_GATE", "off")
    with pytest.raises(vault.SecretError, match="secrets need the trust gate on"):
        vault.resolve_text("{{secret:pw}}")
    monkeypatch.setenv("OPERON_TRUST_GATE", "on")
    assert vault.resolve_text("{{secret:pw}}")[0] == SECRET


def test_no_hint_when_the_vault_is_empty(vault_dir) -> None:
    assert vault.available_names() == []


def test_the_approval_text_names_the_page_host(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "on")
    result = TrustGate(enabled=True).evaluate(_typed(), current_url="https://bank.example/login?x=1")
    assert result.verdict is TrustVerdict.CONFIRM and "(page: bank.example)" in result.reason
    assert SECRET not in result.reason
