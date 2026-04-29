"""Routing helpers for the shared unified agent core."""

from __future__ import annotations

from src.core.contracts.actor import ActorOutput, ExecutorChoice
from src.core.contracts.perception import Environment
from src.core.contracts.planner import ActionType, PlannerOutput
from src.runtime.state import AgentRuntimeState

BROWSER_ACTIONS = {
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.TYPE_TEXT,
    ActionType.PRESS_HOTKEY,
    ActionType.SCROLL,
    ActionType.WAIT,
    ActionType.NAVIGATE,
    ActionType.UPLOAD_FILE_NATIVE,
}

DESKTOP_ACTIONS = {
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.TYPE_TEXT,
    ActionType.PRESS_HOTKEY,
    ActionType.SCROLL,
    ActionType.WAIT,
    ActionType.LAUNCH_APP,
}

CROSS_ENVIRONMENT_ACTIONS = {
    ActionType.UPLOAD_FILE_NATIVE,
}

ALLOWED_TRANSITIONS: dict[Environment, set[Environment]] = {
    Environment.BROWSER: {Environment.BROWSER, Environment.DESKTOP},
    Environment.DESKTOP: {Environment.DESKTOP, Environment.BROWSER},
}


class RoutingError(ValueError):
    """Raised when a plan or actor violates environment routing rules."""


def is_cross_environment_action(action_type: ActionType) -> bool:
    """Return True if the action spans both browser and desktop environments."""

    return action_type in CROSS_ENVIRONMENT_ACTIONS


def _allowed_actions_for(environment: Environment) -> set[ActionType]:
    if environment is Environment.BROWSER:
        return BROWSER_ACTIONS
    return DESKTOP_ACTIONS


def validate_plan_route(plan: PlannerOutput) -> None:
    """Validate that a plan action is compatible with its target environment."""

    allowed = _allowed_actions_for(plan.environment)
    action_type = plan.action.action_type
    if action_type not in allowed:
        raise RoutingError(
            f"Action {action_type.value!r} is not allowed in {plan.environment.value!r} environment"
        )


def route_plan(plan: PlannerOutput) -> ExecutorChoice:
    """Return the explicit executor for a validated plan."""

    validate_plan_route(plan)
    return ExecutorChoice.BROWSER if plan.environment is Environment.BROWSER else ExecutorChoice.DESKTOP


def validate_environment_transition(
    current_state: AgentRuntimeState | None,
    next_environment: Environment,
) -> None:
    """Validate that an environment transition is explicitly allowed."""

    if current_state is None:
        return
    if next_environment not in ALLOWED_TRANSITIONS[current_state.environment]:
        raise RoutingError(
            f"Transition from {current_state.environment.value!r} to {next_environment.value!r} is not allowed"
        )
    if current_state.environment is not next_environment:
        if current_state.goal_progress.status != "advanced":
            raise RoutingError(
                "Cross-environment transition requires the current state to show advanced goal progress"
            )


def validate_actor_for_state(
    current_state: AgentRuntimeState | None,
    plan: PlannerOutput,
    actor: ActorOutput,
) -> None:
    """Validate that the actor output is compatible with the current routed state."""

    validate_plan_route(plan)
    validate_environment_transition(current_state, plan.environment)

    expected_executor = route_plan(plan)
    if actor.executor is not expected_executor:
        raise RoutingError(
            f"Actor executor {actor.executor.value!r} does not match expected executor {expected_executor.value!r}"
        )
    if actor.environment is not plan.environment:
        raise RoutingError("Actor environment must match plan environment")
    if actor.action.action_type is not plan.action.action_type:
        raise RoutingError("Actor action type must match planned action type")
    if current_state is not None:
        if current_state.environment is plan.environment:
            if current_state.environment is Environment.BROWSER and actor.executor is ExecutorChoice.DESKTOP:
                raise RoutingError("Desktop executor cannot be used for a browser-only routed step")
            if current_state.environment is Environment.DESKTOP and actor.executor is ExecutorChoice.BROWSER:
                raise RoutingError("Browser executor cannot be used for a desktop-only routed step")
