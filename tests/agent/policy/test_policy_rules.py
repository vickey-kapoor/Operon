"""Focused tests for rule-first policy coordination."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.policy.coordinator import PolicyCoordinator
from operon.models.common import RunStatus
from operon.models.perception import ScreenPerception, UIElement, UIElementType
from operon.models.policy import ActionType
from operon.models.state import AgentState
from operon.store.memory import FileBackedMemoryStore
from tests.fixtures.paths import unique_artifact_dir
from tests.fixtures.policy import PromptPolicyStub


class StubGeminiClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.calls = 0

    async def generate_policy(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        raise NotImplementedError


def _local_test_dir(name: str) -> Path:
    return unique_artifact_dir(name)


def _prompt_path(root: Path) -> Path:
    path = root / "policy_prompt.txt"
    path.write_text(
        "Intent: {intent}\nSubgoal: {current_subgoal}\nStep: {step_count}\nRetry: {retry_counts}\nPerception: {perception_json}",
        encoding="utf-8",
    )
    return path


def _input_element(element_id: str, label: str, x: int = 320, y: int = 180) -> UIElement:
    return UIElement(
        element_id=element_id,
        element_type=UIElementType.INPUT,
        label=label,
        x=x,
        y=y,
        width=300,
        height=28,
        is_interactable=True,
        confidence=0.95,
    )


def _form_perception(root: Path, page_hint: str = "form_page", focused_element_id: str | None = None) -> ScreenPerception:
    path = root / "run-1" / "step_1" / "before.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return ScreenPerception(
        summary="Form page visible.",
        page_hint=page_hint,
        visible_elements=[
            _input_element("name-input", "Name"),
            _input_element("email-input", "Email", y=220),
            _input_element("message-input", "Message", y=260),
            UIElement(
                element_id="submit-button",
                element_type=UIElementType.BUTTON,
                label="Submit",
                x=320,
                y=320,
                width=120,
                height=32,
                is_interactable=True,
                confidence=0.95,
            ),
        ],
        focused_element_id=focused_element_id,
        capture_artifact_path=str(path),
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_llm_policy_used_only_when_no_rule_matches() -> None:
    root = _local_test_dir("test-policy-coordinator-llm-fallback")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"help-link"},"rationale":"Inspect the help link.","confidence":0.9,"active_subgoal":"inspect page","expected_change":"navigation"}'
    )
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="inspect page",
        step_count=1,
    )
    perception = ScreenPerception(
        summary="A page with a help link is visible.",
        page_hint="unknown",
        visible_elements=[
            UIElement(
                element_id="help-link",
                element_type=UIElementType.LINK,
                label="Help",
                x=24,
                y=116,
                width=108,
                height=36,
                is_interactable=True,
                confidence=0.97,
            )
        ],
        capture_artifact_path=str(root / "run-1" / "step_1" / "before.png"),
        confidence=0.92,
    )
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await coordinator.choose_action(state, perception)

    assert decision.action.action_type is ActionType.CLICK
    assert client.calls == 1
    assert client.last_prompt is not None
    assert "Advisory memory hints" in client.last_prompt


@pytest.mark.asyncio
async def test_type_action_reaches_executor_without_rule_interception() -> None:
    # The focus-before-type rule was removed: click+type is now handled atomically
    # in DesktopExecutor._exec_type (it clicks at the target coords before typing).
    # This test confirms the policy engine passes TYPE through to the LLM and that
    # the returned TYPE decision is not intercepted by a pre-emptive CLICK rule.
    root = _local_test_dir("test-policy-coordinator-focus")
    client = StubGeminiClient(
        response='{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Fill name.","confidence":0.8,"active_subgoal":"fill_name","expected_change":"focus"}'
    )
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )

    decision = await coordinator.choose_action(state, _form_perception(root))

    # Rule engine passes through; LLM is called and its TYPE decision is used directly.
    assert decision.action.action_type is ActionType.TYPE
    assert decision.action.target_element_id == "name-input"
    assert decision.action.text == "Alice"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_form_success_stop_confirmed_when_llm_agrees() -> None:
    """Rule fires success stop; LLM also returns STOP → run terminates."""
    root = _local_test_dir("test-policy-coordinator-form-success-confirmed")
    client = StubGeminiClient(
        response='{"action":{"action_type":"stop"},"rationale":"Task complete.","confidence":1.0,"active_subgoal":"verify_success","expected_change":"none"}'
    )
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="verify_success",
    )
    perception = _form_perception(root, page_hint="form_success")

    decision = await coordinator.choose_action(state, perception)

    # LLM confirmed — original rule decision is returned (preserves rationale/subgoal)
    assert decision.action.action_type is ActionType.STOP
    assert decision.active_subgoal == "verify_success"
    assert client.calls == 1  # LLM was consulted exactly once


@pytest.mark.asyncio
async def test_form_success_stop_overridden_when_llm_sees_error() -> None:
    """Rule fires success stop; LLM returns a recovery action → stop is rejected."""
    root = _local_test_dir("test-policy-coordinator-form-success-override")
    client = StubGeminiClient(
        response='{"action":{"action_type":"wait","wait_ms":500},"rationale":"Error page visible, not success.","confidence":0.9,"active_subgoal":"reassess","expected_change":"none"}'
    )
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="verify_success",
    )
    perception = _form_perception(root, page_hint="form_success")

    decision = await coordinator.choose_action(state, perception)

    # LLM overrode — the LLM's recovery action is returned, not STOP
    assert decision.action.action_type is ActionType.WAIT
    assert decision.active_subgoal == "reassess"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_browser_navigation_summary_does_not_trigger_success_stop_rule() -> None:
    root = _local_test_dir("test-policy-coordinator-browser-navigation")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","x":229,"y":257},"rationale":"Click the Learn more link.","confidence":0.8,"active_subgoal":"click_link","expected_change":"navigation"}'
    )
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Open example.com and click the More information link.",
        status=RunStatus.RUNNING,
        current_subgoal="click link",
        step_count=2,
    )
    perception = ScreenPerception(
        summary='I have successfully navigated to example.com and can see the "Learn more" link.',
        page_hint="unknown",
        visible_elements=[],
        capture_artifact_path=str(root / "run-1" / "step_2" / "before.png"),
        confidence=0.7,
    )
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    decision = await coordinator.choose_action(state, perception)

    assert decision.action.action_type is ActionType.CLICK
    assert client.calls == 1


# ── coordinator invariant tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_reset_run_context_clears_hint_cache() -> None:
    """Per project invariants: reset_run_context must clear the hint cache so context from a previous task does not bleed across runs.
    Auth state is intentionally preserved at the executor layer — not asserted here."""
    root = _local_test_dir("test-reset-run-context")
    client = StubGeminiClient(response='{"action":{"action_type":"wait"},"rationale":"x","confidence":0.1,"active_subgoal":"x","expected_change":"none"}')
    coordinator = PolicyCoordinator(
        delegate=PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )

    # Simulate state populated during a prior run. Sentinel objects are sufficient
    # — we only assert the fields are cleared back to None / empty.
    coordinator._cached_hints = []
    coordinator._cached_hints_key = ("benchmark-prior", "form_page", None, None)

    coordinator.reset_run_context("run-prior")

    assert coordinator._cached_hints is None
    assert coordinator._cached_hints_key is None


@pytest.mark.asyncio
async def test_last_rule_trace_injected_as_advisory_hint_to_llm() -> None:
    """Rule-Augmented Generation: when a rule fired in the previous step,
    state.last_rule_trace must be injected into the next LLM call so the
    planner knows what was tried and whether it worked."""
    root = _local_test_dir("test-rule-trace-injection")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"help-link"},"rationale":"x","confidence":0.9,"active_subgoal":"y","expected_change":"none"}'
    )
    delegate = PromptPolicyStub(gemini_client=client, prompt_path=_prompt_path(root))
    coordinator = PolicyCoordinator(
        delegate=delegate,
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    rule_trace = "[RULE TRACE] Rule 'task_success_stop_rule' fired at step 3 | action=stop | outcome=success"
    state = AgentState(
        run_id="run-1",
        intent="A bare-LLM task with no rule match.",
        status=RunStatus.RUNNING,
        current_subgoal="explore",
        step_count=4,
        last_rule_trace=rule_trace,
    )
    perception = ScreenPerception(
        summary="A page with a help link is visible.",
        page_hint="unknown",
        visible_elements=[
            UIElement(
                element_id="help-link",
                element_type=UIElementType.LINK,
                label="Help",
                x=24, y=116, width=108, height=36,
                is_interactable=True, confidence=0.97,
            )
        ],
        capture_artifact_path=str(root / "run-1" / "step_4" / "before.png"),
        confidence=0.92,
    )
    Path(perception.capture_artifact_path).parent.mkdir(parents=True, exist_ok=True)

    await coordinator.choose_action(state, perception)

    # The rule trace must have been rendered into the prompt sent to the LLM.
    assert client.last_prompt is not None
    assert rule_trace in client.last_prompt


class _RecordingDelegate(PromptPolicyStub):
    """A delegate that, like the model-adapter backends, wants to hear when the
    full harness ran something else in place of its decision (KAN-68)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.replaced: list[tuple[str, str]] = []

    def note_replaced(self, run_id: str, action: str) -> None:
        self.replaced.append((run_id, action))


def _recording_coordinator(root: Path, response: str) -> tuple[PolicyCoordinator, _RecordingDelegate]:
    delegate = _RecordingDelegate(gemini_client=StubGeminiClient(response=response), prompt_path=_prompt_path(root))
    return PolicyCoordinator(delegate=delegate, memory_store=FileBackedMemoryStore(root_dir=root / "runs")), delegate


@pytest.mark.asyncio
async def test_a_rule_action_is_reported_to_the_delegate_as_replacing_its_own() -> None:
    root = _local_test_dir("test-policy-coordinator-note-replaced-rule")
    coordinator, delegate = _recording_coordinator(
        root, '{"action":{"action_type":"click","target_element_id":"irrelevant"},"rationale":"unused.","confidence":0.1,"active_subgoal":"unused","expected_change":"none"}'
    )
    state = AgentState(run_id="run-1", intent="Open Notepad and write hello",
                       status=RunStatus.RUNNING)
    decision = await coordinator.choose_action(state, _computer_use_perception(root))
    assert decision.rule_name == "_prefer_launch_app_rule" and delegate.replaced
    run_id, action = delegate.replaced[0]
    assert run_id == "run-1" and action.startswith("launch_app") and f"rule {decision.rule_name}" in action


@pytest.mark.asyncio
async def test_the_delegates_own_action_and_a_confirmed_stop_are_not_reported_as_replaced() -> None:
    root = _local_test_dir("test-policy-coordinator-note-replaced-own")
    coordinator, delegate = _recording_coordinator(
        root, '{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Fill name.","confidence":0.8,"active_subgoal":"fill_name","expected_change":"focus"}'
    )
    state = AgentState(run_id="run-1", intent="Complete the auth-free form and submit it successfully.",
                       status=RunStatus.RUNNING, current_subgoal="fill_name")
    await coordinator.choose_action(state, _form_perception(root, focused_element_id="name-input"))
    assert delegate.replaced == []

    root = _local_test_dir("test-policy-coordinator-note-replaced-stop")
    coordinator, delegate = _recording_coordinator(
        root, '{"action":{"action_type":"stop"},"rationale":"Task complete.","confidence":1.0,"active_subgoal":"verify_success","expected_change":"none"}'
    )
    state = state.model_copy(update={"current_subgoal": "verify_success"})
    decision = await coordinator.choose_action(state, _form_perception(root, page_hint="form_success"))
    assert decision.action.action_type is ActionType.STOP and delegate.replaced == []


class _ScriptedDelegate:
    """A model-adapter-like delegate: hands out queued decisions and records what the
    full harness tells it about them (KAN-68)."""

    def __init__(self, *decisions) -> None:
        self.decisions = list(decisions)
        self.replaced: list[str] = []
        self.rejected: list[str] = []

    async def choose_action(self, state, perception):
        return self.decisions.pop(0)

    def note_replaced(self, run_id: str, action: str) -> None:
        self.replaced.append(action)

    def note_rejected(self, run_id: str, reason: str) -> None:
        self.rejected.append(reason)


def _decision(action: dict, rationale: str = "Because."):
    from operon.agent.actions.translation import build_policy_decision

    return build_policy_decision({"action": action, "rationale": rationale, "confidence": 0.8, "active_subgoal": "x"})


@pytest.mark.asyncio
async def test_a_guard_never_replaces_the_models_request_for_a_human(monkeypatch) -> None:
    root = _local_test_dir("test-policy-coordinator-keeps-wait-for-user")
    wait = _decision({"action_type": "wait_for_user", "text": "OpenAI safety check: confirm"})
    delegate = _ScriptedDelegate(wait)
    coordinator = PolicyCoordinator(delegate=delegate, memory_store=FileBackedMemoryStore(root_dir=root / "runs"))

    async def guard_replaces(state, perception):
        await coordinator._ask_delegate(state, perception)
        return _decision({"action_type": "launch_app", "text": "notepad"})

    monkeypatch.setattr(coordinator, "_choose_action", guard_replaces)
    state = AgentState(run_id="run-1", intent="Open Notepad", status=RunStatus.RUNNING)
    decision = await coordinator.choose_action(state, _form_perception(root))
    assert decision is wait and delegate.replaced == []


@pytest.mark.asyncio
async def test_asking_the_model_again_tells_it_the_last_proposal_did_not_run(monkeypatch) -> None:
    root = _local_test_dir("test-policy-coordinator-note-rejected")
    first, second = _decision({"action_type": "stop"}), _decision({"action_type": "wait", "wait_ms": 500})
    delegate = _ScriptedDelegate(first, second)
    coordinator = PolicyCoordinator(delegate=delegate, memory_store=FileBackedMemoryStore(root_dir=root / "runs"))

    async def guard_asks_again(state, perception):
        await coordinator._ask_delegate(state, perception)
        return await coordinator._ask_delegate(state, perception)

    monkeypatch.setattr(coordinator, "_choose_action", guard_asks_again)
    state = AgentState(run_id="run-1", intent="Open Notepad", status=RunStatus.RUNNING)
    decision = await coordinator.choose_action(state, _form_perception(root))
    assert decision is second
    assert len(delegate.rejected) == 1 and "asked you again" in delegate.rejected[0]
    assert delegate.replaced == []


def _computer_use_perception(root: Path, summary: str = "A desktop with a taskbar.") -> ScreenPerception:
    """What Gemini Computer Use gives the harness: no elements and no page type."""
    return ScreenPerception(summary=summary, page_hint="unknown",
                            capture_artifact_path=str(root / "step_1" / "before.png"), visible_elements=[],
                            elements_reported=False)


def test_engine_keeps_only_rules_that_fire_under_computer_use() -> None:
    """KAN-161: rules that need an element list or a page type never fired under
    Gemini Computer Use and were removed. The two left work from the task text and
    the model's own words."""
    from operon.agent.policy.rules import PolicyRuleEngine

    root = _local_test_dir("test-policy-rules-computer-use")
    engine = PolicyRuleEngine(launch_apps=False)
    plain = AgentState(run_id="run-cu", intent="Search for Operon on github.com", status=RunStatus.RUNNING)
    assert engine.choose_action(plain, _computer_use_perception(root), []) is None

    done = engine.choose_action(plain, _computer_use_perception(root, "Message sent to Dana."), [])
    assert done is not None and done.action.action_type is ActionType.STOP
    assert engine.last_fired_rule_name() == "_task_success_stop_rule"

    desktop = PolicyRuleEngine(launch_apps=True)
    launch_state = AgentState(run_id="run-cu", intent="Open Notepad and write hello", status=RunStatus.RUNNING)
    launch = desktop.choose_action(launch_state, _computer_use_perception(root), [])
    assert launch is not None and launch.action.action_type is ActionType.LAUNCH_APP
    assert desktop.last_fired_rule_name() == "_prefer_launch_app_rule"
