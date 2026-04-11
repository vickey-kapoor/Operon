"""Focused tests for rule-first policy coordination."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.agent.policy import GeminiPolicyService
from src.agent.policy_coordinator import PolicyCoordinator
from src.models.common import FailureCategory, LoopStage, RunStatus
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.store.memory import FileBackedMemoryStore


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
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _search_perception(root: Path, *, focused_element_id: str | None = None) -> ScreenPerception:
    path = root / "run-1" / "step_1" / "before.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return ScreenPerception(
        summary="Search page with a single search input.",
        page_hint="form_page",
        visible_elements=[
            UIElement(
                element_id="search-input",
                element_type=UIElementType.INPUT,
                label="Search or jump to...",
                x=700,
                y=40,
                width=220,
                height=30,
                is_interactable=True,
                confidence=0.95,
            )
        ],
        focused_element_id=focused_element_id,
        capture_artifact_path=str(path),
        confidence=0.95,
    )


@pytest.mark.asyncio
async def test_rule_first_policy_takes_precedence_for_login_guardrail() -> None:
    root = _local_test_dir("test-policy-coordinator-login")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"submit-button"},"rationale":"Submit the form.","confidence":0.9,"active_subgoal":"submit_form"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Create a Gmail draft and stop before send.",
        status=RunStatus.RUNNING,
        current_subgoal="open compose",
    )
    perception = _form_perception(root, page_hint="google_sign_in")

    decision = await coordinator.choose_action(state, perception)

    assert decision.action.action_type is ActionType.STOP
    assert "authenticated Gmail start state" in decision.rationale
    assert client.calls == 0


@pytest.mark.asyncio
async def test_llm_policy_used_only_when_no_rule_matches() -> None:
    root = _local_test_dir("test-policy-coordinator-llm-fallback")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"help-link"},"rationale":"Inspect the help link.","confidence":0.9,"active_subgoal":"inspect page"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
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
async def test_click_before_type_is_enforced_through_policy_path() -> None:
    root = _local_test_dir("test-policy-coordinator-focus")
    client = StubGeminiClient(
        response='{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Fill name.","confidence":0.8,"active_subgoal":"fill_name"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )

    decision = await coordinator.choose_action(state, _form_perception(root))

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "name-input"
    assert client.calls == 0
    debug_artifacts = coordinator.latest_debug_artifacts()
    assert debug_artifacts is not None
    assert debug_artifacts.selector_trace_artifact_path is not None
    trace_path = Path(debug_artifacts.selector_trace_artifact_path)
    assert trace_path.exists()
    assert '"selected_element_id": "name-input"' in trace_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_search_query_rule_clicks_search_input_before_typing() -> None:
    root = _local_test_dir("test-policy-coordinator-search-click")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"irrelevant"},"rationale":"unused.","confidence":0.1,"active_subgoal":"unused"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Open Chrome, navigate to github.com, and search for Operon",
        status=RunStatus.RUNNING,
        current_subgoal="search",
    )

    decision = await coordinator.choose_action(state, _search_perception(root))

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "search-input"
    assert client.calls == 0


@pytest.mark.asyncio
async def test_search_query_rule_types_and_submits_when_search_input_focused() -> None:
    root = _local_test_dir("test-policy-coordinator-search-type")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","target_element_id":"irrelevant"},"rationale":"unused.","confidence":0.1,"active_subgoal":"unused"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    state = AgentState(
        run_id="run-1",
        intent="Open Chrome, navigate to github.com, and search for Operon",
        status=RunStatus.RUNNING,
        current_subgoal="search",
    )

    decision = await coordinator.choose_action(
        state,
        _search_perception(root, focused_element_id="search-input"),
    )

    assert decision.action.action_type is ActionType.TYPE
    assert decision.action.target_element_id == "search-input"
    assert decision.action.text == "Operon"
    assert decision.action.press_enter is True
    assert client.calls == 0


@pytest.mark.asyncio
async def test_no_identical_type_retry_is_enforced_after_failed_type() -> None:
    root = _local_test_dir("test-policy-coordinator-no-identical-retry")
    client = StubGeminiClient(
        response='{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Retry name.","confidence":0.8,"active_subgoal":"fill_name"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    prior_type = AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice")
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
        action_history=[
            ExecutedAction(
                action=prior_type,
                success=False,
                detail="Execution failed: target not editable.",
                failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
                failure_stage=LoopStage.EXECUTE,
            )
        ],
        verification_history=[
            VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Type failed.",
                failure_type=VerificationFailureType.ACTION_FAILED,
                failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
                failure_stage=LoopStage.EXECUTE,
            )
        ],
        retry_counts={"fill_name:action_failed": 1},
        target_failure_counts={
            f"type:name-input:{FailureCategory.EXECUTION_TARGET_NOT_EDITABLE.value}": 1,
        },
    )

    decision = await coordinator.choose_action(state, _form_perception(root))

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "name-input"
    assert client.calls == 0


@pytest.mark.asyncio
async def test_failed_type_target_not_found_forces_click_on_next_step() -> None:
    root = _local_test_dir("test-policy-coordinator-target-not-found-click")
    client = StubGeminiClient(
        response='{"action":{"action_type":"type","target_element_id":"name-input","text":"Alice"},"rationale":"Retry name.","confidence":0.8,"active_subgoal":"fill_name"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
        memory_store=FileBackedMemoryStore(root_dir=root / "runs"),
    )
    prior_type = AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice")
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="focus name-input",
        action_history=[
            ExecutedAction(
                action=prior_type,
                success=False,
                detail="Execution failed: target not found.",
                failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                failure_stage=LoopStage.EXECUTE,
            )
        ],
        verification_history=[
            VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Type failed.",
                failure_type=VerificationFailureType.ACTION_FAILED,
                failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                failure_stage=LoopStage.EXECUTE,
            )
        ],
        target_failure_counts={
            f"type:name-input:{FailureCategory.EXECUTION_TARGET_NOT_FOUND.value}": 1,
        },
    )

    decision = await coordinator.choose_action(state, _form_perception(root))

    assert decision.action.action_type is ActionType.CLICK
    assert decision.action.target_element_id == "name-input"
    assert client.calls == 0


@pytest.mark.asyncio
async def test_form_success_visible_stops_successfully_before_llm() -> None:
    root = _local_test_dir("test-policy-coordinator-form-success")
    client = StubGeminiClient(
        response='{"action":{"action_type":"wait","wait_ms":1000},"rationale":"Wait.","confidence":0.8,"active_subgoal":"wait"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
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

    assert decision.action.action_type is ActionType.STOP
    assert decision.active_subgoal == "verify_success"
    assert client.calls == 0


@pytest.mark.asyncio
async def test_browser_navigation_summary_does_not_trigger_success_stop_rule() -> None:
    root = _local_test_dir("test-policy-coordinator-browser-navigation")
    client = StubGeminiClient(
        response='{"action":{"action_type":"click","x":229,"y":257},"rationale":"Click the Learn more link.","confidence":0.8,"active_subgoal":"click_link"}'
    )
    coordinator = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=client, prompt_path=_prompt_path(root)),
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
