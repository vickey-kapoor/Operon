"""Schema validation tests for the Phase 1 unified agent contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.contracts.actor import ActorOutput
from src.core.contracts.critic import CriticOutput
from src.core.contracts.perception import PerceptionOutput
from src.core.contracts.planner import PlannerOutput
from src.core.router import route_plan, validate_plan_route
from src.runtime.orchestrator import Phase1Orchestrator

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "contracts"


def _load_fixture(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_browser_fixture_is_valid() -> None:
    fixture = _load_fixture("browser_flow.json")
    perception = PerceptionOutput.model_validate(fixture["perception"])
    planner = PlannerOutput.model_validate(fixture["planner"])
    actor = ActorOutput.model_validate(fixture["actor"])
    critic = CriticOutput.model_validate(fixture["critic"])
    state = Phase1Orchestrator().build_state(
        perception=perception,
        planner=planner,
        actor=actor,
        critic=critic,
    )
    assert state.actor is not None
    assert route_plan(planner).value == "browser"


def test_desktop_fixture_is_valid() -> None:
    fixture = _load_fixture("desktop_flow.json")
    perception = PerceptionOutput.model_validate(fixture["perception"])
    planner = PlannerOutput.model_validate(fixture["planner"])
    actor = ActorOutput.model_validate(fixture["actor"])
    critic = CriticOutput.model_validate(fixture["critic"])
    state = Phase1Orchestrator().build_state(
        perception=perception,
        planner=planner,
        actor=actor,
        critic=critic,
    )
    assert state.critic is not None
    assert route_plan(planner).value == "desktop"


def test_cross_environment_fixture_steps_are_valid() -> None:
    fixture = _load_fixture("cross_environment_flow.json")
    orchestrator = Phase1Orchestrator()
    for step in fixture["steps"]:
        perception = PerceptionOutput.model_validate(step["perception"])
        planner = PlannerOutput.model_validate(step["planner"])
        actor = ActorOutput.model_validate(step["actor"])
        critic = CriticOutput.model_validate(step["critic"])
        state = orchestrator.build_state(
            perception=perception,
            planner=planner,
            actor=actor,
            critic=critic,
        )
        assert state.planner is not None


def test_planner_rejects_coordinates() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(
            {
                "contract_version": "phase1",
                "environment": "browser",
                "observation_id": "obs_bad_1",
                "plan_id": "plan_bad_1",
                "subgoal": "Click the settings button.",
                "rationale": "It is visible.",
                "action": {
                    "action_type": "click",
                    "target_id": "target_settings",
                    "x": 100,
                    "y": 200
                },
                "expected_outcome": "Settings opens."
            }
        )


def test_browser_plan_rejects_launch_app() -> None:
    planner = PlannerOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "browser",
            "observation_id": "obs_bad_2",
            "plan_id": "plan_bad_2",
            "subgoal": "Launch Notepad.",
            "rationale": "Test invalid browser action.",
            "action": {
                "action_type": "launch_app",
                "app_name": "Notepad"
            },
            "expected_outcome": "Notepad opens."
        }
    )
    with pytest.raises(ValueError):
        validate_plan_route(planner)


def test_actor_rejects_wrong_executor_for_environment() -> None:
    with pytest.raises(ValidationError):
        ActorOutput.model_validate(
            {
                "contract_version": "phase1",
                "environment": "browser",
                "observation_id": "obs_bad_3",
                "plan_id": "plan_bad_3",
                "attempt_id": "attempt_bad_3",
                "executor": "desktop",
                "action": {
                    "action_type": "click",
                    "target_id": "target_settings",
                    "target_label": "Settings",
                    "hotkey": []
                },
                "status": "success",
                    "failure_type": None,
                "details": "Wrong executor."
            }
        )


def test_critic_requires_failure_type_for_failure() -> None:
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(
            {
                "contract_version": "phase1",
                "environment": "desktop",
                "observation_id": "obs_bad_4",
                "plan_id": "plan_bad_4",
                "attempt_id": "attempt_bad_4",
                "outcome": "failure",
                "judgment": "The app did not open."
            }
        )


def test_orchestrator_rejects_mismatched_observation_ids() -> None:
    perception = PerceptionOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "browser",
            "observation_id": "obs_one",
            "summary": "A page is visible.",
            "context_label": "Page",
            "visible_targets": [],
                "focused_target_id": None,
            "notes": []
        }
    )
    planner = PlannerOutput.model_validate(
        {
            "contract_version": "phase1",
            "environment": "browser",
            "observation_id": "obs_two",
            "plan_id": "plan_one",
            "subgoal": "Wait.",
            "rationale": "This is a mismatch test.",
            "action": {
                "action_type": "wait",
                "wait_ms": 500
            },
            "expected_outcome": "Nothing changes yet."
        }
    )
    with pytest.raises(ValueError):
        Phase1Orchestrator().build_state(perception=perception, planner=planner)
