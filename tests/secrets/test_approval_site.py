"""KAN-103: an approval to type a stored secret names the site it goes into."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.trust_gate import TrustGate
from operon.core.runtime import current_runtime
from operon.models.policy import ActionType, AgentAction, PolicyDecision

pytestmark = pytest.mark.usefixtures("vault_dir")

VALUE = "hunter2-real-value"
TYPE = AgentAction(action_type=ActionType.TYPE, x=10, y=20, text="{{secret:pw}}")


class _Browser:
    def __init__(self, url: str | None) -> None:
        self.url = url

    async def current_url_for_run(self, run_id: str) -> str | None:
        return self.url


def _loop(executor, gate: TrustGate) -> AgentLoop:
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=Mock(),
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    loop._trust_gate = gate
    return loop


def _text(loop: AgentLoop, action: AgentAction = TYPE) -> str | None:
    decision = PolicyDecision(action=action, rationale="log in", confidence=0.9, active_subgoal="s")
    url = asyncio.run(loop._current_url_for_trust_gate("r", secret=True))
    return loop._pre_execution_gate_text(decision, intent="x", run_id="r", current_url=url)


@pytest.fixture(autouse=True)
def _no_threshold(monkeypatch):
    monkeypatch.setattr(current_runtime().control, "confidence_threshold", 0.0)


def test_no_allowlist_still_names_the_host() -> None:
    text = _text(_loop(_Browser("https://accounts.example.com/login"), TrustGate(enabled=True)))
    assert text is not None and "on accounts.example.com" in text
    assert "{{secret:pw}}" in text and VALUE not in text


def test_unreadable_url_says_unknown_site() -> None:
    text = _text(_loop(_Browser(None), TrustGate(enabled=True)))
    assert text is not None and "on an unknown site" in text


def test_allowlist_off_page_still_names_host() -> None:
    gate = TrustGate(enabled=True, allow_domains=("bank.example",))
    text = _text(_loop(_Browser("https://evil.test/x"), gate))
    assert text is not None and "Blocked by policy" in text  # allowlist semantics unchanged


def test_allowlisted_page_needs_no_approval() -> None:
    gate = TrustGate(enabled=True, allow_domains=("bank.example",))
    assert _text(_loop(_Browser("https://www.bank.example/login"), gate)) is None


def test_desktop_message_is_unchanged() -> None:
    text = _text(_loop(SimpleNamespace(), TrustGate(enabled=True)))
    assert text is not None and "site" not in text and text.startswith("Account: approve type")


def test_url_not_fetched_for_non_secret_action() -> None:
    loop = _loop(_Browser("https://a.test/"), TrustGate(enabled=True))
    assert asyncio.run(loop._current_url_for_trust_gate("r")) is None


async def test_sdk_callback_receives_the_host(monkeypatch, tmp_path) -> None:
    from operon import Agent
    from operon.sdk import ApprovalRequest
    from tests.fixtures.frontdoor import build_loop
    from tests.test_sdk import STOP, _use

    loop, executor = build_loop(monkeypatch, tmp_path, [TYPE, STOP], gate=TrustGate(enabled=True))
    executor.current_url_for_run = _Browser("https://accounts.example.com/signin").current_url_for_run
    _use(monkeypatch, loop)
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    await Agent(on_approval=approve, poll_s=0.01).arun("Log in")

    assert len(seen) == 1
    assert "on accounts.example.com" in seen[0].reason and VALUE not in seen[0].reason


# --- batches: the approval shows the secret and the site, and the site is re-checked ---


def _nav(url: str) -> AgentAction:
    return AgentAction(action_type=ActionType.NAVIGATE, url=url)


def _batch(*children: AgentAction) -> AgentAction:
    return AgentAction(action_type=ActionType.BATCH, actions=list(children))


class _BatchBrowser(_Browser):
    """Runs a batch like the browser executor: the child check before every later child."""

    def __init__(self, url: str | None) -> None:
        super().__init__(url)
        self.batch_child_check = None
        self.typed: list[str] = []

    async def run(self, batch: AgentAction):
        for index, child in enumerate(batch.actions):
            if index > 0 and self.batch_child_check is not None:
                blocked = await self.batch_child_check(child)
                if blocked is not None:
                    return index, blocked
            if child.action_type is ActionType.NAVIGATE:
                self.url = child.url
            elif child.action_type is ActionType.TYPE:
                self.typed.append(child.text)
        return None, None


def _paused(url: str, batch: AgentAction, gate: TrustGate, mode: str | None = None):
    browser = _BatchBrowser(url)
    loop = _loop(browser, gate)

    async def _alive(run_id: str) -> bool:
        return False

    loop._run_cancelled = _alive  # type: ignore[method-assign]
    decision = PolicyDecision(action=batch, rationale="log in", confidence=0.9, active_subgoal="s")
    page = asyncio.run(loop._current_url_for_trust_gate("r", secret=True))
    text = loop._pre_execution_gate_text(decision, intent="x", run_id="r", current_url=page, permission_mode=mode)
    return loop, browser, text


@pytest.mark.parametrize(
    ("first", "gate", "mode"),
    [
        (_nav("https://evil.test/login"), TrustGate(enabled=True), "risky"),
        (AgentAction(action_type=ActionType.CLICK, x=1, y=2), TrustGate(enabled=True), "ask"),
        (_nav("https://x.test/checkout"), TrustGate(enabled=True, confirm=("checkout",)), "risky"),
    ],
)
def test_batch_approval_shows_the_secret_and_site(first, gate, mode) -> None:
    _, _, text = _paused("https://accounts.example.com/", _batch(first, TYPE), gate, mode)
    assert text is not None
    assert "{{secret:pw}}" in text and "on accounts.example.com" in text and "re-checked before typing" in text
    assert VALUE not in text


def test_batch_stops_before_the_secret_when_the_site_changes() -> None:
    loop, browser, text = _paused(
        "https://accounts.example.com/", _batch(_nav("https://evil.test/login"), TYPE), TrustGate(enabled=True)
    )
    assert text is not None
    loop._install_batch_child_check("r", _batch(_nav("https://evil.test/login"), TYPE))
    index, blocked = asyncio.run(browser.run(_batch(_nav("https://evil.test/login"), TYPE)))
    assert index == 1 and browser.typed == []
    assert "nothing was typed" in blocked and VALUE not in blocked


def test_batch_types_the_secret_on_the_approved_site() -> None:
    batch = _batch(_nav("https://accounts.example.com/step2"), TYPE)
    loop, browser, text = _paused("https://accounts.example.com/", batch, TrustGate(enabled=True))
    assert text is not None
    loop._install_batch_child_check("r", batch)
    assert asyncio.run(browser.run(batch)) == (None, None)
    assert browser.typed == ["{{secret:pw}}"]


def test_batch_stops_when_no_host_was_known_at_approval() -> None:
    batch = _batch(AgentAction(action_type=ActionType.CLICK, x=1, y=2), TYPE)
    loop, browser, text = _paused(None, batch, TrustGate(enabled=True))
    assert text is not None and "on an unknown site" in text
    browser.url = "https://appeared.test/"  # a host appearing mid-batch was never approved
    loop._install_batch_child_check("r", batch)
    index, blocked = asyncio.run(browser.run(batch))
    assert index == 1 and browser.typed == [] and blocked
