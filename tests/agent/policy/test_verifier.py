"""Focused tests for deterministic verifier behavior."""

import json
from pathlib import Path

import pytest

from operon.agent.policy.verifier import DeterministicVerifierService
from operon.clients.gemini import PlaceholderGeminiClient
from operon.models.common import FailureCategory, LoopStage, StopReason
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception, UIElement, UIElementType
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationFailureType, VerificationStatus


def _service() -> DeterministicVerifierService:
    return DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())


class StubVerificationClient:
    def __init__(self, response: str | None = None, *, raises: bool = False) -> None:
        self.response = response
        self.raises = raises

    async def generate_verification(self, prompt: str, screenshot_path: str) -> str:
        if self.raises:
            raise RuntimeError("verification unavailable")
        if self.response is None:
            raise RuntimeError("missing response")
        return self.response


def _prompt_path(tmp_path: Path) -> Path:
    path = tmp_path / "critic_prompt.txt"
    path.write_text(
        "intent={intent}\nsubgoal={current_subgoal}\naction={action_json}\nreason={rationale}\nconfidence={confidence}\n"
        "detail={execution_detail}\nprevious={previous_summary}",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_verifier_success_case() -> None:
    state = AgentState(run_id="run-1", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.9, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True
    assert result.stop_condition_met is False


@pytest.mark.asyncio
async def test_verifier_expected_outcome_not_met() -> None:
    state = AgentState(run_id="run-2", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="subject", text="Hello")
    decision = PolicyDecision(action=action, rationale="Fill subject.", confidence=0.8, active_subgoal="fill subject")
    executed = ExecutedAction(action=action, success=False, detail="typing failed")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.FAILURE
    assert result.expected_outcome_met is False
    assert result.failure_type is VerificationFailureType.ACTION_FAILED


@pytest.mark.asyncio
async def test_verifier_stop_before_send_boundary() -> None:
    state = AgentState(run_id="run-3", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.STOP)
    decision = PolicyDecision(action=action, rationale="Stop before send.", confidence=1.0, active_subgoal="stop before send")
    executed = ExecutedAction(action=action, success=True, detail="stopped before send")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True
    assert result.stop_condition_met is True
    assert result.failure_type is VerificationFailureType.STOP_BOUNDARY_REACHED
    assert result.stop_reason is StopReason.TASK_COMPLETED


@pytest.mark.asyncio
async def test_verifier_distinguishes_benchmark_precondition_stop() -> None:
    state = AgentState(run_id="run-setup", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.STOP)
    decision = PolicyDecision(
        action=action,
        rationale="Benchmark requires an authenticated start state; pre-auth screens are out of scope.",
        confidence=1.0,
        active_subgoal="stop for benchmark setup",
    )
    executed = ExecutedAction(action=action, success=True, detail="stopped for benchmark setup")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.FAILURE
    assert result.expected_outcome_met is False
    assert result.stop_condition_met is True
    assert result.failure_type is VerificationFailureType.BENCHMARK_PRECONDITION_FAILED
    assert result.failure_category is FailureCategory.BENCHMARK_PRECONDITION_FAILED
    assert result.failure_stage is LoopStage.CHOOSE_ACTION
    assert result.stop_reason is StopReason.BENCHMARK_PRECONDITION_FAILED


@pytest.mark.asyncio
async def test_verifier_uncertain_case() -> None:
    state = AgentState(run_id="run-4", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.2, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.expected_outcome_met is False
    assert result.stop_condition_met is False
    assert result.failure_type is VerificationFailureType.UNCERTAIN_SCREEN_STATE


@pytest.mark.asyncio
async def test_verifier_detects_form_success_from_page_hint() -> None:
    state = AgentState(
        run_id="run-form",
        intent="Complete the auth-free form and submit it successfully.",
        status="running",
        observation_history=[
            ScreenPerception(
                summary="Thank you, your form was submitted successfully.",
                page_hint="form_success",
                capture_artifact_path="runs/run-form/step_1/before.png",
                visible_elements=[],
            )
        ],
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button")
    decision = PolicyDecision(action=action, rationale="Submit form.", confidence=0.9, active_subgoal="submit_form")
    executed = ExecutedAction(action=action, success=True, detail="clicked submit")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is True
    assert result.stop_reason is StopReason.FORM_SUBMITTED_SUCCESS


@pytest.mark.asyncio
async def test_verifier_uses_model_backed_critic_success(tmp_path: Path) -> None:
    state = AgentState(run_id="run-5", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.2, active_subgoal="open compose")
    screenshot = tmp_path / "after.png"
    screenshot.write_bytes(b"fake")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose", artifact_path=str(screenshot))
    client = StubVerificationClient(
        response=json.dumps(
            {
                "status": "success",
                "expected_outcome_met": True,
                "stop_condition_met": False,
                "reason": "Compose view is visible after the click.",
            }
        )
    )

    service = DeterministicVerifierService(gemini_client=client, prompt_path=_prompt_path(tmp_path))
    result = await service.verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True
    assert result.reason == "Compose view is visible after the click."
    assert result.critic_model_used is True
    assert result.critic_fallback_reason is None
    debug = service.latest_debug_artifacts()
    assert debug is not None
    assert debug.prompt_artifact_path.endswith("verification_prompt.txt")
    assert debug.raw_response_artifact_path.endswith("verification_raw.txt")
    assert debug.parsed_artifact_path.endswith("verification_result.json")


@pytest.mark.asyncio
async def test_verifier_uses_model_backed_critic_failure(tmp_path: Path) -> None:
    state = AgentState(run_id="run-6", intent="Fill subject", status="running")
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="subject", text="Hello")
    decision = PolicyDecision(action=action, rationale="Fill subject.", confidence=0.9, active_subgoal="fill subject")
    screenshot = tmp_path / "after.png"
    screenshot.write_bytes(b"fake")
    executed = ExecutedAction(action=action, success=True, detail="typed subject", artifact_path=str(screenshot))
    client = StubVerificationClient(
        response=json.dumps(
            {
                "status": "failure",
                "expected_outcome_met": False,
                "stop_condition_met": False,
                "reason": "The subject field still appears empty.",
            }
        )
    )

    service = DeterministicVerifierService(gemini_client=client, prompt_path=_prompt_path(tmp_path))
    result = await service.verify(state, decision, executed)

    assert result.status is VerificationStatus.FAILURE
    assert result.failure_type is VerificationFailureType.EXPECTED_OUTCOME_NOT_MET
    assert result.failure_category is FailureCategory.EXPECTED_OUTCOME_NOT_MET
    assert result.failure_stage is LoopStage.VERIFY


@pytest.mark.asyncio
async def test_verifier_falls_back_when_model_critic_unavailable(tmp_path: Path) -> None:
    state = AgentState(run_id="run-7", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.2, active_subgoal="open compose")
    screenshot = tmp_path / "after.png"
    screenshot.write_bytes(b"fake")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose", artifact_path=str(screenshot))

    service = DeterministicVerifierService(
        gemini_client=StubVerificationClient(raises=True),
        prompt_path=_prompt_path(tmp_path),
    )
    result = await service.verify(state, decision, executed)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.failure_type is VerificationFailureType.UNCERTAIN_SCREEN_STATE
    assert result.critic_model_used is False
    assert result.critic_fallback_reason == "critic_unavailable_or_unusable"
    debug = service.latest_debug_artifacts()
    assert debug is not None
    assert debug.diagnostics_artifact_path is not None


@pytest.mark.asyncio
async def test_verifier_critic_failure_is_logged_and_persisted(tmp_path: Path, caplog) -> None:
    # A dead critic falls back to SUCCESS at confidence >= 0.5, so the failure
    # reason must reach both the log and the step's diagnostics file.
    state = AgentState(run_id="run-9", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.9, active_subgoal="open compose")
    screenshot = tmp_path / "after.png"
    screenshot.write_bytes(b"fake")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose", artifact_path=str(screenshot))
    service = DeterministicVerifierService(
        gemini_client=StubVerificationClient(raises=True),
        prompt_path=_prompt_path(tmp_path),
    )

    with caplog.at_level("WARNING", logger="operon.agent.policy.verifier"):
        result = await service.verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.critic_fallback_reason == "critic_unavailable_or_unusable"
    assert "Critic call failed" in caplog.text
    assert "verification unavailable" in caplog.text
    diagnostics = json.loads((tmp_path / "verification_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["critic_fallback_reason"] == "critic_error: verification unavailable"
    assert (tmp_path / "verification_prompt.txt").exists()


@pytest.mark.asyncio
async def test_verifier_does_not_treat_input_label_as_typed_value() -> None:
    state = AgentState(
        run_id="run-8",
        intent="Find the best MacBook under $2000",
        status="running",
    )
    state.progress_state.target_value_history["id:search_input"] = "MacBook under $2000"
    state.observation_history.append(
        ScreenPerception(
            summary="Search results for busboys movie review.",
            page_hint="search_results",
            capture_artifact_path="runs/run-8/step_3/after.png",
            visible_elements=[],
        )
    )
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="search_button")
    decision = PolicyDecision(action=action, rationale="Submit the search.", confidence=0.9, active_subgoal="submit search")
    executed = ExecutedAction(action=action, success=True, detail="clicked search")
    state.observation_history[-1] = ScreenPerception(
        summary="Search results page with the old query still visible.",
        page_hint="search_results",
        capture_artifact_path="runs/run-8/step_3/after.png",
        visible_elements=[
            UIElement(
                element_id="search_input",
                element_type=UIElementType.INPUT,
                label="busboys movie review",
                x=100,
                y=20,
                width=400,
                height=40,
                is_interactable=True,
                confidence=1.0,
            )
        ],
    )

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True


def _write_step_frames(step_dir: Path, *, changed: bool) -> Path:
    from PIL import Image

    step_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (480, 270), "white").save(step_dir / "before.png")
    after = Image.new("RGB", (480, 270), "white")
    if changed:
        after.paste((0, 0, 0), (0, 0, 240, 135))
    after.save(step_dir / "after.png")
    return step_dir / "after.png"


def _stable_wait_inputs(after_path: Path):
    state = AgentState(run_id="run-sw", intent="Open compose", status="running")
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=10)
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.9, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked", artifact_path=str(after_path))
    return state, decision, executed


@pytest.mark.asyncio
async def test_verifier_returns_stable_wait_when_action_changed_screen(tmp_path: Path) -> None:
    after = _write_step_frames(tmp_path / "step_1", changed=True)

    result = await _service().verify(*_stable_wait_inputs(after))

    assert result.status is VerificationStatus.STABLE_WAIT


@pytest.mark.asyncio
async def test_verifier_skips_stable_wait_when_screen_unchanged(tmp_path: Path) -> None:
    after = _write_step_frames(tmp_path / "step_1", changed=False)

    result = await _service().verify(*_stable_wait_inputs(after))

    assert result.status is not VerificationStatus.STABLE_WAIT


@pytest.mark.asyncio
async def test_verifier_settle_reverify_never_returns_stable_wait(tmp_path: Path) -> None:
    # The loop's settle re-verify passes allow_stable_wait=False; the unchanged
    # before.png would otherwise make the verdict STABLE_WAIT forever.
    after = _write_step_frames(tmp_path / "step_1", changed=True)

    result = await _service().verify(*_stable_wait_inputs(after), allow_stable_wait=False)

    assert result.status is not VerificationStatus.STABLE_WAIT
