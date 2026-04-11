"""Contract-only orchestrator for the unified Phase 2 agent flow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from core.contracts.actor import ActorOutput
from core.contracts.critic import CriticOutcome, CriticOutput, FailureType
from core.contracts.perception import PerceptionOutput
from core.contracts.planner import ActionType, PlannerOutput
from core.router import (
    RoutingError,
    validate_actor_for_state,
    validate_environment_transition,
    validate_plan_route,
)
from runtime.state import AgentRuntimeState, StepState


class UnifiedOrchestrator(BaseModel):
    """One shared loop that validates and simulates browser/desktop steps."""

    model_config = ConfigDict(extra="forbid")
    max_retries: int = 3

    @staticmethod
    def adaptation_strategy_for(failure_type: FailureType | None) -> str | None:
        """Return the deterministic adaptation strategy for a failure type."""

        if failure_type is FailureType.TIMING_ISSUE:
            return "wait_then_retry"
        if failure_type is FailureType.WRONG_WINDOW_ACTIVE:
            return "focus_correction_then_retry"
        if failure_type is FailureType.TARGET_NOT_FOUND:
            return "reperceive_and_replan"
        if failure_type is FailureType.AMBIGUOUS_PERCEPTION:
            return "refresh_state_and_replan"
        if failure_type is FailureType.PICKER_NOT_DETECTED:
            return "wait_then_retry"
        if failure_type is FailureType.FILE_NOT_REFLECTED:
            return "reperceive_and_replan"
        return None

    @staticmethod
    def detect_file_picker(perception: PerceptionOutput) -> bool:
        """Return True if the perception signals that a native OS file picker is active."""

        picker_signals = {"open", "file", "save", "browse", "choose", "select file"}
        label = perception.context_label.lower()
        if any(signal in label for signal in picker_signals):
            return True
        return any(
            any(signal in note.lower() for signal in picker_signals)
            for note in perception.notes
        )

    def process_step(
        self,
        *,
        perception: PerceptionOutput,
        planner: PlannerOutput,
        actor: ActorOutput,
        critic: CriticOutput,
        current_state: AgentRuntimeState | None = None,
    ) -> StepState:
        """Validate one step and return the resulting shared state."""

        validate_plan_route(planner)
        validate_environment_transition(current_state, perception.environment)

        if planner.observation_id != perception.observation_id:
            raise RoutingError("planner observation_id must match perception observation_id")
        if planner.environment != perception.environment:
            raise RoutingError("planner environment must match perception environment")
        if actor.plan_id != planner.plan_id:
            raise RoutingError("actor plan_id must match planner plan_id")
        if actor.observation_id != planner.observation_id:
            raise RoutingError("actor observation_id must match planner observation_id")
        if critic.attempt_id != actor.attempt_id:
            raise RoutingError("critic attempt_id must match actor attempt_id")
        if critic.plan_id != actor.plan_id:
            raise RoutingError("critic plan_id must match actor plan_id")
        if critic.environment != actor.environment:
            raise RoutingError("critic environment must match actor environment")

        validate_actor_for_state(current_state, planner, actor)

        base_state = current_state or AgentRuntimeState.from_perception(perception)
        after_state = base_state.apply_step(
            perception=perception,
            planner=planner,
            actor=actor,
            critic=critic,
        )
        if planner.action.action_type is ActionType.UPLOAD_FILE_NATIVE:
            picker_active = self.detect_file_picker(perception)
            after_state = after_state.model_copy(update={"file_picker_active": picker_active})
        return StepState(
            before=current_state,
            after=after_state,
            perception=perception,
            planner=planner,
            actor=actor,
            critic=critic,
            retry_count=after_state.retry_count,
        )

    def build_state(
        self,
        *,
        perception: PerceptionOutput,
        planner: PlannerOutput,
        actor: ActorOutput | None = None,
        critic: CriticOutput | None = None,
        current_state: AgentRuntimeState | None = None,
    ) -> StepState:
        """Backward-compatible single-step validator used by Phase 1 tests."""

        if actor is None or critic is None:
            validate_plan_route(planner)
            validate_environment_transition(current_state, perception.environment)
            if planner.observation_id != perception.observation_id:
                raise RoutingError("planner observation_id must match perception observation_id")
            if planner.environment != perception.environment:
                raise RoutingError("planner environment must match perception environment")
            base_state = current_state or AgentRuntimeState.from_perception(perception)
            after_state = base_state.model_copy(
                update={
                    "environment": perception.environment,
                    "active_app": perception.active_app or perception.context_label,
                    "current_url": perception.current_url,
                    "visible_elements": list(perception.visible_targets),
                    "uncertainties": list(perception.notes),
                    "latest_observation_id": perception.observation_id,
                    "latest_plan_id": planner.plan_id,
                    "goal_progress": base_state.goal_progress.model_copy(
                        update={"subgoal": planner.subgoal}
                    ),
                }
            )
            return StepState(
                before=current_state,
                after=after_state,
                perception=perception,
                planner=planner,
                actor=actor,
                critic=critic,
                retry_count=after_state.retry_count,
            )
        return self.process_step(
            perception=perception,
            planner=planner,
            actor=actor,
            critic=critic,
            current_state=current_state,
        )

    def simulate_flow(self, steps: list[dict]) -> list[StepState]:
        """Simulate a multi-step browser, desktop, or cross-environment flow."""

        current_state: AgentRuntimeState | None = None
        results: list[StepState] = []
        for step in steps:
            result = self.process_step(
                perception=PerceptionOutput.model_validate(step["perception"]),
                planner=PlannerOutput.model_validate(step["planner"]),
                actor=ActorOutput.model_validate(step["actor"]),
                critic=CriticOutput.model_validate(step["critic"]),
                current_state=current_state,
            )
            results.append(result)
            current_state = result.after
        return results

    def process_step_with_adaptation(
        self,
        *,
        attempts: list[dict],
        current_state: AgentRuntimeState | None = None,
    ) -> StepState:
        """Simulate one step with deterministic failure-driven adaptation."""

        if not attempts:
            raise RoutingError("at least one attempt is required")

        working_state = current_state
        adaptation_trace: list[str] = []
        retries_used = 0
        final_result: StepState | None = None

        for index, attempt in enumerate(attempts):
            result = self.process_step(
                perception=PerceptionOutput.model_validate(attempt["perception"]),
                planner=PlannerOutput.model_validate(attempt["planner"]),
                actor=ActorOutput.model_validate(attempt["actor"]),
                critic=CriticOutput.model_validate(attempt["critic"]),
                current_state=working_state,
            )
            final_result = result

            if result.critic.outcome is CriticOutcome.SUCCESS:
                success_state = result.after.model_copy(
                    update={
                        "retry_count": retries_used,
                        "last_failure_type": None,
                        "last_strategy": adaptation_trace[-1] if adaptation_trace else None,
                    }
                )
                return result.model_copy(
                    update={
                        "after": success_state,
                        "retry_count": retries_used,
                        "adaptation_trace": adaptation_trace,
                    }
                )

            strategy = self.adaptation_strategy_for(result.critic.failure_type)
            if strategy is None:
                failed_state = result.after.model_copy(
                    update={
                        "retry_count": retries_used,
                        "last_failure_type": result.critic.failure_type,
                        "last_strategy": None,
                    }
                )
                return result.model_copy(
                    update={
                        "after": failed_state,
                        "retry_count": retries_used,
                        "adaptation_trace": adaptation_trace,
                    }
                )

            if retries_used >= self.max_retries or index == len(attempts) - 1:
                failed_state = result.after.model_copy(
                    update={
                        "retry_count": retries_used,
                        "last_failure_type": result.critic.failure_type,
                        "last_strategy": strategy,
                    }
                )
                return result.model_copy(
                    update={
                        "after": failed_state,
                        "retry_count": retries_used,
                        "adaptation_trace": adaptation_trace,
                    }
                )

            retries_used += 1
            adaptation_trace.append(strategy)
            working_state = result.after.apply_retry_feedback(
                perception=result.perception,
                planner=result.planner,
                actor=result.actor,
                critic=result.critic,
                retry_count=retries_used,
                strategy=strategy,
            )

        if final_result is None:
            raise RoutingError("unable to produce a final step result")
        return final_result.model_copy(
            update={"retry_count": retries_used, "adaptation_trace": adaptation_trace}
        )


Phase1Orchestrator = UnifiedOrchestrator
