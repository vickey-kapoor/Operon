"""Adapters that translate legacy Operon runtime objects into unified contracts."""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.actor import ActorAction, ActorOutput, ActorStatus, ExecutorChoice
from core.contracts.critic import CriticOutcome, CriticOutput, FailureType
from core.contracts.perception import Environment, PerceptionOutput, VisibleTarget
from core.contracts.planner import ActionType as ContractActionType
from core.contracts.planner import PlannerAction, PlannerOutput
from src.models.common import FailureCategory
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception, UIElement
from src.models.policy import AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus


def _target_label(perception: ScreenPerception, target_element_id: str | None) -> str | None:
    if target_element_id is None:
        return None
    for element in perception.visible_elements:
        if element.element_id == target_element_id:
            return element.primary_name
    return None


def _map_visible_target(element: UIElement) -> VisibleTarget:
    return VisibleTarget(
        target_id=element.element_id,
        role=element.element_type.value,
        label=element.label or element.name or element.primary_name,
        text=element.text,
        confidence=element.confidence,
    )


def _map_action_type(action: AgentAction) -> ContractActionType:
    if action.action_type.value == "click":
        return ContractActionType.CLICK
    if action.action_type.value == "double_click":
        return ContractActionType.DOUBLE_CLICK
    if action.action_type.value == "type":
        return ContractActionType.TYPE_TEXT
    if action.action_type.value in {"hotkey", "press_key"}:
        return ContractActionType.PRESS_HOTKEY
    if action.action_type.value == "scroll":
        return ContractActionType.SCROLL
    if action.action_type.value == "wait":
        return ContractActionType.WAIT
    if action.action_type.value == "launch_app":
        return ContractActionType.LAUNCH_APP
    if action.action_type.value == "navigate":
        return ContractActionType.NAVIGATE
    if action.action_type.value == "upload_file_native":
        return ContractActionType.UPLOAD_FILE_NATIVE
    raise ValueError(f"Legacy action {action.action_type.value!r} is not supported by the unified contract")


def _map_planner_action(action: AgentAction, perception: ScreenPerception) -> PlannerAction:
    contract_action_type = _map_action_type(action)
    payload: dict[str, object] = {"action_type": contract_action_type}

    if contract_action_type in {ContractActionType.CLICK, ContractActionType.DOUBLE_CLICK}:
        payload["target_id"] = action.target_element_id
        payload["target_label"] = _target_label(perception, action.target_element_id)
    elif contract_action_type is ContractActionType.TYPE_TEXT:
        payload["target_id"] = action.target_element_id
        payload["target_label"] = _target_label(perception, action.target_element_id)
        payload["text"] = action.text
    elif contract_action_type is ContractActionType.PRESS_HOTKEY:
        key = action.key or ""
        keys = [part.strip().lower() for part in key.split("+") if part.strip()] or ["unknown"]
        payload["hotkey"] = keys
    elif contract_action_type is ContractActionType.SCROLL:
        amount = action.scroll_amount or 0
        payload["scroll_direction"] = "down" if amount >= 0 else "up"
        payload["scroll_amount"] = abs(amount) or 1
    elif contract_action_type is ContractActionType.WAIT:
        payload["wait_ms"] = action.wait_ms or 500
    elif contract_action_type is ContractActionType.LAUNCH_APP:
        payload["app_name"] = action.text
    elif contract_action_type is ContractActionType.NAVIGATE:
        payload["url"] = action.url
    elif contract_action_type is ContractActionType.UPLOAD_FILE_NATIVE:
        payload["target_id"] = action.target_element_id or action.selector or "upload_control"
        payload["target_label"] = _target_label(perception, action.target_element_id)
        payload["file_path"] = action.text
        payload["picker_title"] = action.text

    return PlannerAction.model_validate(payload)


def _map_failure_type(executed_action: ExecutedAction, verification: VerificationResult) -> FailureType | None:
    category = executed_action.failure_category or verification.failure_category
    if category in {
        FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        FailureCategory.SELECTOR_NO_CANDIDATES_AFTER_FILTERING,
        FailureCategory.TARGET_RERESOLUTION_FAILED,
        FailureCategory.TARGET_LOST_BEFORE_ACTION,
    }:
        return FailureType.TARGET_NOT_FOUND
    if category in {
        FailureCategory.FOCUS_VERIFICATION_FAILED,
        FailureCategory.CLICK_BEFORE_TYPE_FAILED,
    }:
        return FailureType.WRONG_WINDOW_ACTIVE
    if category in {
        FailureCategory.TYPE_VERIFICATION_FAILED,
        FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
    }:
        return FailureType.TEXT_NOT_ENTERED
    if category in {
        FailureCategory.STALE_TARGET_BEFORE_ACTION,
        FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
    }:
        return FailureType.UI_CHANGED
    if category in {
        FailureCategory.UNCERTAIN_SCREEN_STATE,
        FailureCategory.PERCEPTION_LOW_QUALITY,
        FailureCategory.AMBIGUOUS_TARGET_CANDIDATES,
    }:
        return FailureType.AMBIGUOUS_PERCEPTION
    if category is FailureCategory.PICKER_NOT_DETECTED:
        return FailureType.PICKER_NOT_DETECTED
    if category is FailureCategory.FILE_NOT_REFLECTED:
        return FailureType.FILE_NOT_REFLECTED
    if verification.status is VerificationStatus.FAILURE:
        return FailureType.TIMING_ISSUE
    if verification.status is VerificationStatus.UNCERTAIN:
        return FailureType.AMBIGUOUS_PERCEPTION
    return None


@dataclass(slots=True)
class LegacyContractBundle:
    """One translated unified-contract step."""

    perception: PerceptionOutput
    planner: PlannerOutput
    actor: ActorOutput
    critic: CriticOutput


class LegacyOperonContractAdapter:
    """Translate legacy Operon runtime objects into unified contracts."""

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def perception_output(
        self,
        state: AgentState,
        perception: ScreenPerception,
        attempt_index: int,
    ) -> PerceptionOutput:
        return PerceptionOutput(
            environment=self.environment,
            observation_id=f"{state.run_id}:obs:{state.step_count}:{attempt_index}",
            summary=perception.summary,
            context_label=perception.page_hint.value,
            active_app=perception.page_hint.value,
            current_url=state.start_url if self.environment is Environment.BROWSER else None,
            visible_targets=[_map_visible_target(element) for element in perception.visible_elements],
            focused_target_id=perception.focused_element_id,
            notes=[],
        )

    def planner_output(
        self,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
        attempt_index: int,
    ) -> PlannerOutput:
        observation_id = f"{state.run_id}:obs:{state.step_count}:{attempt_index}"
        return PlannerOutput(
            environment=self.environment,
            observation_id=observation_id,
            plan_id=f"{state.run_id}:plan:{state.step_count}:{attempt_index}",
            subgoal=decision.active_subgoal,
            rationale=decision.rationale,
            action=_map_planner_action(decision.action, perception),
            expected_outcome=decision.rationale,
        )

    def actor_output(
        self,
        planner: PlannerOutput,
        executed_action: ExecutedAction,
        attempt_index: int,
    ) -> ActorOutput:
        executor = ExecutorChoice.BROWSER if self.environment is Environment.BROWSER else ExecutorChoice.DESKTOP
        status = ActorStatus.SUCCESS if executed_action.success else ActorStatus.FAILED
        failure_type = _map_failure_type(executed_action, VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="placeholder",
        )) if not executed_action.success else None
        return ActorOutput(
            environment=self.environment,
            observation_id=planner.observation_id,
            plan_id=planner.plan_id,
            attempt_id=f"{planner.plan_id}:attempt:{attempt_index}",
            executor=executor,
            action=ActorAction.model_validate(planner.action.model_dump()),
            status=status,
            failure_type=failure_type,
            details=executed_action.detail,
        )

    def critic_output(
        self,
        planner: PlannerOutput,
        actor: ActorOutput,
        executed_action: ExecutedAction,
        verification: VerificationResult,
    ) -> CriticOutput:
        failure_type = _map_failure_type(executed_action, verification)
        if verification.status is VerificationStatus.SUCCESS:
            outcome = CriticOutcome.SUCCESS
        elif failure_type in {
            FailureType.TIMING_ISSUE,
            FailureType.WRONG_WINDOW_ACTIVE,
            FailureType.TARGET_NOT_FOUND,
            FailureType.AMBIGUOUS_PERCEPTION,
        }:
            outcome = CriticOutcome.RETRY
        else:
            outcome = CriticOutcome.FAILURE

        return CriticOutput(
            environment=self.environment,
            observation_id=planner.observation_id,
            plan_id=planner.plan_id,
            attempt_id=actor.attempt_id,
            outcome=outcome,
            failure_type=failure_type,
            judgment=verification.reason,
        )

    def bundle(
        self,
        *,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
        attempt_index: int,
    ) -> LegacyContractBundle:
        perception_output = self.perception_output(state, perception, attempt_index)
        planner_output = self.planner_output(state, perception, decision, attempt_index)
        actor_output = self.actor_output(planner_output, executed_action, attempt_index)
        critic_output = self.critic_output(planner_output, actor_output, executed_action, verification)
        return LegacyContractBundle(
            perception=perception_output,
            planner=planner_output,
            actor=actor_output,
            critic=critic_output,
        )
