"""Shared runtime state for the unified browser and desktop agent."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.actor import ActorAction, ActorOutput
from src.core.contracts.critic import CriticOutput, FailureType
from src.core.contracts.perception import Environment, PerceptionOutput, VisibleTarget
from src.core.contracts.planner import PlannerOutput


class GoalProgressState(BaseModel):
    """Minimal goal-progress summary for the shared loop."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="unknown", min_length=1, max_length=50)
    subgoal: str | None = Field(default=None, min_length=1, max_length=300)
    summary: str | None = Field(default=None, min_length=1, max_length=500)


class AgentRuntimeState(BaseModel):
    """Shared state model for browser and desktop flows."""

    model_config = ConfigDict(extra="forbid")

    environment: Environment
    active_app: str | None = Field(default=None, min_length=1, max_length=200)
    current_url: str | None = Field(default=None, min_length=1, max_length=500)
    visible_elements: list[VisibleTarget] = Field(default_factory=list)
    goal_progress: GoalProgressState = Field(default_factory=GoalProgressState)
    last_action: ActorAction | None = None
    last_critic_result: CriticOutput | None = None
    retry_count: int = Field(default=0, ge=0)
    last_failure_type: FailureType | None = None
    last_strategy: str | None = Field(default=None, min_length=1, max_length=100)
    file_picker_active: bool = False
    uncertainties: list[str] = Field(default_factory=list)
    latest_observation_id: str | None = Field(default=None, min_length=1, max_length=100)
    latest_plan_id: str | None = Field(default=None, min_length=1, max_length=100)
    latest_attempt_id: str | None = Field(default=None, min_length=1, max_length=100)

    @classmethod
    def from_perception(cls, perception: PerceptionOutput) -> "AgentRuntimeState":
        """Create initial runtime state from a perception contract."""

        return cls(
            environment=perception.environment,
            active_app=perception.active_app or perception.context_label,
            current_url=perception.current_url,
            visible_elements=list(perception.visible_targets),
            retry_count=0,
            last_failure_type=None,
            last_strategy=None,
            uncertainties=list(perception.notes),
            latest_observation_id=perception.observation_id,
        )

    def apply_step(
        self,
        *,
        perception: PerceptionOutput,
        planner: PlannerOutput,
        actor: ActorOutput,
        critic: CriticOutput,
    ) -> "AgentRuntimeState":
        """Return the next state after one simulated step."""

        progress_status = "advanced" if critic.outcome.value == "success" else "blocked"
        summary = critic.judgment
        return self.model_copy(
            update={
                "environment": perception.environment,
                "active_app": perception.active_app or perception.context_label,
                "current_url": perception.current_url,
                "visible_elements": list(perception.visible_targets),
                "goal_progress": GoalProgressState(
                    status=progress_status,
                    subgoal=planner.subgoal,
                    summary=summary,
                ),
                "last_action": actor.action,
                "last_critic_result": critic,
                "retry_count": 0,
                "last_failure_type": None,
                "uncertainties": list(perception.notes),
                "latest_observation_id": perception.observation_id,
                "latest_plan_id": planner.plan_id,
                "latest_attempt_id": actor.attempt_id,
            }
        )

    def apply_retry_feedback(
        self,
        *,
        perception: PerceptionOutput,
        planner: PlannerOutput,
        actor: ActorOutput,
        critic: CriticOutput,
        retry_count: int,
        strategy: str,
    ) -> "AgentRuntimeState":
        """Return state updated after a failed attempt that will be retried."""

        return self.model_copy(
            update={
                "environment": perception.environment,
                "active_app": perception.active_app or perception.context_label,
                "current_url": perception.current_url,
                "visible_elements": list(perception.visible_targets),
                "goal_progress": GoalProgressState(
                    status="blocked",
                    subgoal=planner.subgoal,
                    summary=critic.judgment,
                ),
                "last_action": actor.action,
                "last_critic_result": critic,
                "retry_count": retry_count,
                "last_failure_type": critic.failure_type,
                "last_strategy": strategy,
                "uncertainties": list(perception.notes),
                "latest_observation_id": perception.observation_id,
                "latest_plan_id": planner.plan_id,
                "latest_attempt_id": actor.attempt_id,
            }
        )


class StepState(BaseModel):
    """Validated step bundle plus pre/post shared state."""

    model_config = ConfigDict(extra="forbid")

    before: AgentRuntimeState | None = None
    after: AgentRuntimeState
    perception: PerceptionOutput
    planner: PlannerOutput
    actor: ActorOutput
    critic: CriticOutput
    retry_count: int = Field(default=0, ge=0)
    adaptation_trace: list[str] = Field(default_factory=list)
