"""Phase 3 tests for deterministic failure-driven adaptation."""

from __future__ import annotations

from core.contracts.perception import Environment
from runtime.orchestrator import UnifiedOrchestrator


def _attempt(
    *,
    environment: str,
    observation_id: str,
    plan_id: str,
    attempt_id: str,
    subgoal: str,
    action: dict,
    outcome: str,
    judgment: str,
    failure_type: str | None = None,
    context_label: str = "Context",
    active_app: str | None = None,
    current_url: str | None = None,
    visible_targets: list[dict] | None = None,
) -> dict:
    return {
        "perception": {
            "contract_version": "phase1",
            "environment": environment,
            "observation_id": observation_id,
            "summary": "Synthetic perception for adaptation testing.",
            "context_label": context_label,
            "active_app": active_app,
            "current_url": current_url,
            "visible_targets": visible_targets or [],
            "focused_target_id": None,
            "notes": [],
        },
        "planner": {
            "contract_version": "phase1",
            "environment": environment,
            "observation_id": observation_id,
            "plan_id": plan_id,
            "subgoal": subgoal,
            "rationale": "Synthetic plan for adaptation testing.",
            "action": action,
            "expected_outcome": "The goal advances.",
        },
        "actor": {
            "contract_version": "phase1",
            "environment": environment,
            "observation_id": observation_id,
            "plan_id": plan_id,
            "attempt_id": attempt_id,
            "executor": environment,
            "action": {
                **action,
                **({} if "hotkey" in action else {"hotkey": []}),
            },
            "status": "success",
            "failure_type": None,
            "details": "Stub actor recorded an action request.",
        },
        "critic": {
            "contract_version": "phase1",
            "environment": environment,
            "observation_id": observation_id,
            "plan_id": plan_id,
            "attempt_id": attempt_id,
            "outcome": outcome,
            "failure_type": failure_type,
            "judgment": judgment,
        },
    }


def test_retry_success_after_timing_issue() -> None:
    attempts = [
        _attempt(
            environment="browser",
            observation_id="obs_1",
            plan_id="plan_1",
            attempt_id="attempt_1",
            subgoal="Open settings.",
            action={"action_type": "click", "target_id": "target_settings", "target_label": "Settings"},
            outcome="retry",
            failure_type="timing_issue",
            judgment="The page was still loading.",
            context_label="Dashboard",
            active_app="Example App",
            current_url="https://app.example.com/dashboard",
        ),
        _attempt(
            environment="browser",
            observation_id="obs_1b",
            plan_id="plan_1b",
            attempt_id="attempt_1b",
            subgoal="Open settings.",
            action={"action_type": "click", "target_id": "target_settings", "target_label": "Settings"},
            outcome="success",
            judgment="The page is now ready and the click can succeed.",
            context_label="Settings",
            active_app="Example App",
            current_url="https://app.example.com/settings",
        ),
    ]

    result = UnifiedOrchestrator().process_step_with_adaptation(attempts=attempts)
    assert result.retry_count == 1
    assert result.after.retry_count == 1
    assert result.after.last_strategy == "wait_then_retry"
    assert result.after.last_failure_type is None
    assert result.after.goal_progress.status == "advanced"


def test_replan_after_target_not_found() -> None:
    attempts = [
        _attempt(
            environment="browser",
            observation_id="obs_replan_1",
            plan_id="plan_replan_1",
            attempt_id="attempt_replan_1",
            subgoal="Open settings.",
            action={"action_type": "click", "target_id": "target_missing", "target_label": "Settings"},
            outcome="retry",
            failure_type="target_not_found",
            judgment="The target is no longer present.",
            context_label="Dashboard",
            active_app="Example App",
            current_url="https://app.example.com/dashboard",
        ),
        _attempt(
            environment="browser",
            observation_id="obs_replan_2",
            plan_id="plan_replan_2",
            attempt_id="attempt_replan_2",
            subgoal="Open settings using the refreshed target.",
            action={"action_type": "click", "target_id": "target_settings_v2", "target_label": "Settings"},
            outcome="success",
            judgment="The refreshed target was used successfully.",
            context_label="Settings",
            active_app="Example App",
            current_url="https://app.example.com/settings",
            visible_targets=[{
                "target_id": "target_settings_v2",
                "role": "button",
                "label": "Settings",
                "text": "Settings",
                "confidence": 0.99,
            }],
        ),
    ]

    result = UnifiedOrchestrator().process_step_with_adaptation(attempts=attempts)
    assert result.retry_count == 1
    assert result.after.last_strategy == "reperceive_and_replan"
    assert result.after.latest_observation_id == "obs_replan_2"
    assert result.after.latest_plan_id == "plan_replan_2"
    assert result.after.goal_progress.subgoal == "Open settings using the refreshed target."


def test_failure_after_max_retries() -> None:
    attempts = [
        _attempt(
            environment="desktop",
            observation_id=f"obs_fail_{i}",
            plan_id=f"plan_fail_{i}",
            attempt_id=f"attempt_fail_{i}",
            subgoal="Open Notepad.",
            action={"action_type": "launch_app", "app_name": "Notepad"},
            outcome="retry",
            failure_type="timing_issue",
            judgment="The window did not appear yet.",
            context_label="Desktop",
            active_app="Windows Desktop",
        )
        for i in range(1, 5)
    ]

    result = UnifiedOrchestrator().process_step_with_adaptation(attempts=attempts)
    assert result.retry_count == 3
    assert result.after.retry_count == 3
    assert result.after.last_failure_type is not None
    assert result.after.last_failure_type.value == "timing_issue"
    assert result.after.last_strategy == "wait_then_retry"
    assert result.after.goal_progress.status == "blocked"
    assert len(result.adaptation_trace) == 3


def test_correct_state_updates_across_retries() -> None:
    attempts = [
        _attempt(
            environment="desktop",
            observation_id="obs_state_1",
            plan_id="plan_state_1",
            attempt_id="attempt_state_1",
            subgoal="Bring Notepad to the foreground.",
            action={"action_type": "press_hotkey", "hotkey": ["alt", "tab"]},
            outcome="retry",
            failure_type="wrong_window_active",
            judgment="A different app still has focus.",
            context_label="Desktop",
            active_app="Browser",
        ),
        _attempt(
            environment="desktop",
            observation_id="obs_state_2",
            plan_id="plan_state_2",
            attempt_id="attempt_state_2",
            subgoal="Refresh the visible app state.",
            action={"action_type": "wait", "wait_ms": 200},
            outcome="retry",
            failure_type="ambiguous_perception",
            judgment="The focused window is still unclear.",
            context_label="Desktop",
            active_app="Notepad",
        ),
        _attempt(
            environment="desktop",
            observation_id="obs_state_3",
            plan_id="plan_state_3",
            attempt_id="attempt_state_3",
            subgoal="Continue in Notepad.",
            action={"action_type": "wait", "wait_ms": 200},
            outcome="success",
            judgment="The state is now stable.",
            context_label="Notepad",
            active_app="Notepad",
        ),
    ]

    result = UnifiedOrchestrator().process_step_with_adaptation(attempts=attempts)
    assert result.after.environment is Environment.DESKTOP
    assert result.after.active_app == "Notepad"
    assert result.after.retry_count == 2
    assert result.after.last_failure_type is None
    assert result.after.last_strategy == "refresh_state_and_replan"
    assert result.after.goal_progress.status == "advanced"
    assert result.adaptation_trace == [
        "focus_correction_then_retry",
        "refresh_state_and_replan",
    ]
