"""Running task note (KAN-102): building it, and sending it with the next step."""

from __future__ import annotations

import dataclasses

from evals.harness import run_scenario
from evals.scenarios import cu_click_then_done
from operon.agent.task_note import MAX_NOTE_CHARS, build_task_note
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.verification import (
    VerificationResult,
    VerificationStatus,
)


def _executed(action: AgentAction, success: bool = True) -> ExecutedAction:
    return ExecutedAction(action=action, success=success, detail="ok")


def _verdict(status: VerificationStatus, met: bool = True) -> VerificationResult:
    return VerificationResult(status=status, expected_outcome_met=met, stop_condition_met=False, reason="r")


def _click(x: int, y: int) -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, x=x, y=y)


def test_no_note_before_the_first_action() -> None:
    assert build_task_note("do it", 0, [], []) == ""


def test_lines_describe_action_and_outcome() -> None:
    actions = [
        _executed(AgentAction(action_type=ActionType.TYPE, text="hello", x=412, y=220)),
        _executed(_click(900, 40)),
    ]
    verdicts = [_verdict(VerificationStatus.SUCCESS), _verdict(VerificationStatus.FAILURE, met=False)]

    note = build_task_note("fill the form", 2, actions, verdicts)

    assert "Task so far (2 steps done): fill the form" in note
    assert "1. typed 5 chars into field at (412,220) -> success" in note
    assert "2. clicked (900,40) -> no visible change" in note
    assert "Plan:" not in note
    assert "hello" not in note


def test_failed_execution_and_unchecked_outcomes() -> None:
    actions = [_executed(_click(1, 2), success=False), _executed(_click(3, 4))]
    unchecked = VerificationResult(
        status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="r", unchecked=True
    )
    note = build_task_note("t", 2, actions, [_verdict(VerificationStatus.SUCCESS), unchecked])

    assert "1. clicked (1,2) -> failed" in note
    assert "2. clicked (3,4) -> done" in note


def test_lines_are_numbered_by_action_not_by_step_count() -> None:
    # Two pause steps counted without an action.
    note = build_task_note("t", 5, [_executed(_click(1, 1)), _executed(_click(2, 2))], [_verdict(VerificationStatus.SUCCESS)] * 2)

    assert "(5 steps done)" in note
    assert "1. clicked (1,1)" in note and "2. clicked (2,2)" in note


def test_only_the_last_max_items_actions_are_listed_with_true_step_numbers() -> None:
    actions = [_executed(_click(i, i)) for i in range(1, 13)]
    verdicts = [_verdict(VerificationStatus.SUCCESS)] * 12

    note = build_task_note("t", 12, actions, verdicts, max_items=3)

    assert "10. clicked (10,10)" in note and "12. clicked (12,12)" in note
    assert "9. clicked" not in note


def test_plan_steps_are_included_when_present() -> None:
    note = build_task_note(
        "t", 1, [_executed(_click(1, 1))], [_verdict(VerificationStatus.SUCCESS)], plan_steps=["open the page", "search"]
    )

    assert "Plan: 1. open the page; 2. search" in note


def test_long_notes_drop_the_oldest_lines_first_and_stay_capped() -> None:
    actions = [
        _executed(AgentAction(action_type=ActionType.TYPE, text="x" * 200, x=i, y=i)) for i in range(1, 51)
    ]
    verdicts = [_verdict(VerificationStatus.SUCCESS)] * 50

    note = build_task_note("t", 50, actions, verdicts, plan_steps=["step " * 20] * 8, max_items=50)

    assert len(note) <= MAX_NOTE_CHARS
    assert "50. typed" in note
    assert "1. typed" not in note


def _prompt_text(request) -> str:
    return str(request)


async def test_flag_on_sends_the_first_action_with_the_second_step() -> None:
    scenario = dataclasses.replace(cu_click_then_done(), env={"OPERON_TASK_NOTE": "on"})

    result = await run_scenario(scenario)

    assert result.outcome == "success", result.summary()
    assert "Task so far" not in _prompt_text(result.model_requests[0])
    assert "1. clicked" in _prompt_text(result.model_requests[1])


async def test_flag_off_prompts_are_unchanged() -> None:
    off = await run_scenario(cu_click_then_done())
    explicit_off = await run_scenario(dataclasses.replace(cu_click_then_done(), env={"OPERON_TASK_NOTE": "off"}))
    on = await run_scenario(dataclasses.replace(cu_click_then_done(), env={"OPERON_TASK_NOTE": "on"}))

    assert [_prompt_text(r) for r in off.model_requests] == [_prompt_text(r) for r in explicit_off.model_requests]
    assert all("Task so far" not in _prompt_text(r) for r in off.model_requests)
    # The same first prompt either way; only later steps differ.
    assert _prompt_text(off.model_requests[0]) == _prompt_text(on.model_requests[0])
    assert _prompt_text(off.model_requests[1]) != _prompt_text(on.model_requests[1])


async def test_task_note_file_is_saved_and_plan_json_is_read(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from operon.agent.loop import AgentLoop

    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "plan.json").write_text(json.dumps({"steps": ["open the page", 7, "search"]}), encoding="utf-8")
    hints: list[tuple] = []
    perception = SimpleNamespace(add_advisory_hints=lambda h, source="", run_id="": hints.append((h, source, run_id)))
    loop = AgentLoop.__new__(AgentLoop)
    loop.run_store = SimpleNamespace(root_dir=tmp_path)
    loop.perception_service = perception
    assert loop._saved_plan_steps("r1") == ["open the page", "search"]
    assert loop._saved_plan_steps("missing") == []

    from unittest.mock import patch

    from operon.core.settings import Settings

    record = SimpleNamespace(
        intent="do it", step_count=1, action_history=[_executed(_click(5, 6))],
        verification_history=[_verdict(VerificationStatus.SUCCESS)],
    )
    ctx = SimpleNamespace(record=record, run_id="r1", step_index=2)
    with patch("operon.agent.loop.get_settings", return_value=Settings.from_env({"OPERON_TASK_NOTE": "on"})):
        loop._add_task_note(ctx)

    saved = (tmp_path / "r1" / "step_2" / "task_note.txt").read_text(encoding="utf-8")
    assert "Plan: 1. open the page; 2. search" in saved and "1. clicked (5,6)" in saved
    assert hints == [([saved], "task_note", "r1")]


def test_a_new_task_note_replaces_the_queued_one_in_every_hint_store() -> None:
    from operon.agent.backends.browser_claude_computer_use import (
        BrowserClaudeComputerUseBackend,
    )
    from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
    from operon.agent.backends.model_adapter.backend import ModelAdapterBackend
    from operon.agent.backends.model_adapter.openai_native import (
        OpenAIComputerUseBackend,
    )

    for cls in (
        BrowserComputerUseBackend,
        BrowserClaudeComputerUseBackend,
        ModelAdapterBackend,
        OpenAIComputerUseBackend,
    ):
        backend = cls.__new__(cls)
        attr = "_hints" if cls in (ModelAdapterBackend, OpenAIComputerUseBackend) else "_advisory_hints"
        setattr(backend, attr, {})
        backend._corrected = set()
        backend.add_advisory_hints(["old note"], source="task_note", run_id="r")
        backend.add_advisory_hints(["plan"], source="planner", run_id="r")
        backend.add_advisory_hints(["new note"], source="task_note", run_id="r")
        texts = [h for h, _ in getattr(backend, attr)["r"]]
        assert texts == ["plan", "new note"], cls.__name__
