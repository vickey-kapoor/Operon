"""Step-efficiency score (KAN-85): pure function, saved summary, eval table."""

from __future__ import annotations

import json

from evals.__main__ import waste_cell
from evals.harness import run_scenario
from evals.scenarios import cu_click_then_done
from operon.agent.efficiency import (
    compute_efficiency,
    efficiency_for_response,
    run_efficiency,
)


def _step(action: dict, *, status: str = "success", ok: bool = True, category: str | None = None,
          source: str | None = "[LLM] gemini") -> dict:
    row = {
        "executed_action": {"action": action, "success": ok, "failure_category": category},
        "verification_result": {"status": status},
    }
    if source is not None:
        row["decision_source"] = source
    return row


CLICK_A = {"action_type": "click", "x": 1, "y": 2}
CLICK_B = {"action_type": "click", "x": 9, "y": 9}
WAIT = {"action_type": "wait", "wait_ms": 500}


def test_empty_run() -> None:
    result = compute_efficiency([])
    assert result["total_steps"] == 0
    assert result["wasted_steps"] == 0
    assert result["wasted_ratio"] is None
    assert result["model_calls"] is None and result["actions_per_model_call"] is None
    assert result["wall_seconds"] is None and result["seconds_per_step"] is None


def test_all_good_run_with_timing() -> None:
    result = compute_efficiency([_step(CLICK_A), _step(CLICK_B)], [1000, 3000])
    assert result["wasted_steps"] == 0 and result["wasted_ratio"] == 0.0
    assert result["model_calls"] == 2
    assert result["wall_seconds"] == 4.0 and result["seconds_per_step"] == 2.0


def test_repeat_without_change_is_wasted_but_repeat_that_progressed_is_not() -> None:
    wasted = compute_efficiency([_step(CLICK_A), _step(CLICK_A, status="uncertain")])
    assert wasted["wasted_steps"] == 1 and wasted["wasted_repeat"] == 1
    fine = compute_efficiency([_step(CLICK_A), _step(CLICK_A, status="progressing_stable")])
    assert fine["wasted_steps"] == 0


def test_blocked_and_failed_counted_once() -> None:
    blocked = _step(CLICK_A, ok=False, category="repeated_action_without_progress", status="failure")
    failed = _step(CLICK_B, ok=False, status="failure")
    result = compute_efficiency([_step(CLICK_A), blocked, failed])
    assert result["wasted_steps"] == 2
    assert result["wasted_blocked"] == 1 and result["wasted_failed"] == 2
    assert result["wasted_ratio"] == round(2 / 3, 3)


def test_noop_wait_and_pre_step_failure() -> None:
    result = compute_efficiency([_step(WAIT, status="uncertain"), {"record_type": "pre_step_failure"}])
    assert result["wasted_steps"] == 2 and result["wasted_noop_wait"] == 1


def test_batching_raises_actions_per_model_call() -> None:
    batch = {"action_type": "batch", "actions": [CLICK_A, CLICK_B, WAIT]}
    result = compute_efficiency([_step(batch), _step(CLICK_A, source="[RULE] x")])
    assert result["model_calls"] == 1
    assert result["actions_per_model_call"] == 4.0


def test_old_run_missing_fields_never_crashes() -> None:
    result = compute_efficiency([{}, {"executed_action": None}, {"verification_result": "junk"}, "junk"])  # type: ignore[list-item]
    assert result["total_steps"] == 3
    assert result["model_calls"] is None and result["wall_seconds"] is None


def test_run_efficiency_reads_files_and_tolerates_missing(tmp_path) -> None:
    assert run_efficiency(tmp_path / "nope")["total_steps"] == 0
    (tmp_path / "run.jsonl").write_text(json.dumps(_step(CLICK_A)) + "\nnot json\n", encoding="utf-8")
    (tmp_path / "step_1").mkdir()
    (tmp_path / "step_1" / "step_timing.json").write_text('{"total_ms": 2500}', encoding="utf-8")
    result = run_efficiency(tmp_path)
    assert result["total_steps"] == 1 and result["wall_seconds"] == 2.5


def test_response_helper_skips_running_runs(tmp_path) -> None:
    class Status:
        value = "running"

    class Run:
        run_id = "r"
        efficiency = None
        status = Status()

    class Store:
        root_dir = tmp_path

    assert efficiency_for_response(Run(), Store()) is None
    Status.value = "failed"
    assert efficiency_for_response(Run(), Store())["total_steps"] == 0


async def test_finished_scenario_run_has_efficiency_in_saved_summary() -> None:
    result = await run_scenario(cu_click_then_done())
    assert result.efficiency is not None
    assert result.efficiency["total_steps"] == result.steps
    assert result.efficiency["wasted_steps"] is not None
    assert waste_cell(result.efficiency).endswith(f"/{result.steps}")


def test_waste_cell_handles_missing() -> None:
    assert waste_cell(None) == "-"
    assert waste_cell({"wasted_steps": 1, "total_steps": 4}) == "1/4"


def test_evals_table_renders(capsys) -> None:
    from evals.__main__ import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "waste" in out.splitlines()[0] and "correct" in out
