"""Phase 2 tests for routing and environment transitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.contracts.actor import ActorOutput
from core.contracts.critic import CriticOutput
from core.contracts.perception import Environment, PerceptionOutput
from core.contracts.planner import PlannerOutput
from core.router import (
    RoutingError,
    route_plan,
    validate_actor_for_state,
    validate_plan_route,
)
from runtime.orchestrator import UnifiedOrchestrator

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "contracts"


def _load_fixture(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_valid_browser_routing() -> None:
    fixture = _load_fixture("browser_flow.json")
    planner = PlannerOutput.model_validate(fixture["planner"])
    actor = ActorOutput.model_validate(fixture["actor"])
    validate_plan_route(planner)
    validate_actor_for_state(None, planner, actor)
    assert route_plan(planner).value == "browser"


def test_valid_desktop_routing() -> None:
    fixture = _load_fixture("desktop_flow.json")
    planner = PlannerOutput.model_validate(fixture["planner"])
    actor = ActorOutput.model_validate(fixture["actor"])
    validate_plan_route(planner)
    validate_actor_for_state(None, planner, actor)
    assert route_plan(planner).value == "desktop"


def test_valid_browser_to_desktop_transition() -> None:
    fixture = _load_fixture("cross_environment_flow.json")
    results = UnifiedOrchestrator().simulate_flow(fixture["steps"])
    assert len(results) == 2
    assert results[0].after.environment is Environment.BROWSER
    assert results[1].after.environment is Environment.DESKTOP
    assert results[1].after.goal_progress.status == "advanced"


def test_invalid_desktop_action_in_browser_only_context() -> None:
    browser_fixture = _load_fixture("browser_state_transition.json")
    orchestrator = UnifiedOrchestrator()
    first = browser_fixture["steps"][0]
    state = orchestrator.process_step(
        perception=PerceptionOutput.model_validate(first["perception"]),
        planner=PlannerOutput.model_validate(first["planner"]),
        actor=ActorOutput.model_validate(first["actor"]),
        critic=CriticOutput.model_validate(first["critic"]),
    ).after

    invalid_plan = PlannerOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "browser",
            "observation_id": "obs_browser_invalid_1",
            "plan_id": "plan_browser_invalid_1",
            "subgoal": "Open Notepad.",
            "rationale": "Invalid desktop action in browser context.",
            "action": {"action_type": "launch_app", "app_name": "Notepad"},
            "expected_outcome": "Notepad opens."
        }
    )
    invalid_actor = ActorOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "browser",
            "observation_id": "obs_browser_invalid_1",
            "plan_id": "plan_browser_invalid_1",
            "attempt_id": "attempt_browser_invalid_1",
            "executor": "browser",
            "action": {"action_type": "launch_app", "app_name": "Notepad", "hotkey": []},
            "status": "success",
            "failure_type": None,
            "details": "Invalid browser-native launch_app."
        }
    )

    with pytest.raises(RoutingError):
        validate_actor_for_state(state, invalid_plan, invalid_actor)


def test_invalid_browser_action_in_desktop_only_context() -> None:
    desktop_fixture = _load_fixture("desktop_state_transition.json")
    orchestrator = UnifiedOrchestrator()
    first = desktop_fixture["steps"][0]
    state = orchestrator.process_step(
        perception=PerceptionOutput.model_validate(first["perception"]),
        planner=PlannerOutput.model_validate(first["planner"]),
        actor=ActorOutput.model_validate(first["actor"]),
        critic=CriticOutput.model_validate(first["critic"]),
    ).after

    invalid_plan = PlannerOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "desktop",
            "observation_id": "obs_desktop_invalid_1",
            "plan_id": "plan_desktop_invalid_1",
            "subgoal": "Navigate to a web page.",
            "rationale": "Invalid browser action in desktop context.",
            "action": {"action_type": "navigate", "url": "https://example.com"},
            "expected_outcome": "The browser page opens."
        }
    )
    invalid_actor = ActorOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "desktop",
            "observation_id": "obs_desktop_invalid_1",
            "plan_id": "plan_desktop_invalid_1",
            "attempt_id": "attempt_desktop_invalid_1",
            "executor": "desktop",
            "action": {"action_type": "navigate", "url": "https://example.com", "hotkey": []},
            "status": "success",
            "failure_type": None,
            "details": "Invalid desktop-native navigate."
        }
    )

    with pytest.raises(RoutingError):
        validate_actor_for_state(state, invalid_plan, invalid_actor)


def test_invalid_cross_environment_transition() -> None:
    browser_fixture = _load_fixture("browser_flow.json")
    browser_step = UnifiedOrchestrator().process_step(
        perception=PerceptionOutput.model_validate(browser_fixture["perception"]),
        planner=PlannerOutput.model_validate(browser_fixture["planner"]),
        actor=ActorOutput.model_validate(browser_fixture["actor"]),
        critic=CriticOutput.model_validate(browser_fixture["critic"]),
    )
    blocked_state = browser_step.after.model_copy(
        update={
            "goal_progress": browser_step.after.goal_progress.model_copy(update={"status": "blocked"})
        }
    )

    cross_fixture = _load_fixture("cross_environment_flow.json")
    desktop_step = cross_fixture["steps"][1]
    with pytest.raises(RoutingError):
        UnifiedOrchestrator().process_step(
            perception=PerceptionOutput.model_validate(desktop_step["perception"]),
            planner=PlannerOutput.model_validate(desktop_step["planner"]),
            actor=ActorOutput.model_validate(desktop_step["actor"]),
            critic=CriticOutput.model_validate(desktop_step["critic"]),
            current_state=blocked_state,
        )
