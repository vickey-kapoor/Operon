"""Step-efficiency score (KAN-85): pure function, saved summary, eval table."""

from __future__ import annotations

import json

from evals.__main__ import waste_cell
from evals.harness import run_scenario
from evals.scenarios import cu_click_then_done, cu_relaunch_runaway
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


def test_evals_table_renders(capsys, monkeypatch) -> None:
    import logging

    import evals.__main__ as evals_main

    monkeypatch.setattr(evals_main, "SCENARIOS", [cu_click_then_done])
    monkeypatch.setattr(logging, "disable", lambda *_a, **_k: None)
    monkeypatch.setenv("OPERON_STRICT_ERRORS", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    assert evals_main.main([]) == 0
    out = capsys.readouterr().out
    assert "waste" in out.splitlines()[0] and "1/1 correct" in out


async def test_unfinished_run_is_still_scored_by_the_harness() -> None:
    result = await run_scenario(cu_relaunch_runaway())
    assert result.efficiency is not None and result.efficiency["total_steps"] == result.steps
    assert waste_cell(result.efficiency) != "-"


async def test_step_budget_exhaustion_saves_efficiency(tmp_path, monkeypatch) -> None:
    from evals import harness
    from operon.models.common import RunStatus, RunTaskRequest, StepRequest

    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    scenario = cu_click_then_done()
    world = harness._World(scenario, tmp_path)
    loop, _executor, _client = harness._build_loop(scenario, world, tmp_path)
    run = await loop.start_run(RunTaskRequest(intent=scenario.intent, max_steps=1))
    first = await loop.step_run(StepRequest(run_id=run.run_id))
    assert first.status is RunStatus.RUNNING
    final = await loop.step_run(StepRequest(run_id=run.run_id))  # budget used: the run fails here
    assert final.status is RunStatus.FAILED

    state = await loop.run_store.get_run(run.run_id)
    assert state.efficiency is not None and state.efficiency["total_steps"] == 1
    assert (tmp_path / "runs" / run.run_id / "efficiency.json").exists()


async def test_efficiency_json_is_written(monkeypatch) -> None:
    import tempfile
    from pathlib import Path

    seen: dict = {}

    class Keep(tempfile.TemporaryDirectory):  # keep the files so the test can read them
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen["root"] = self.name
            seen["keep_alive"] = self  # its finalizer would delete the directory on garbage collection

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr("evals.harness.tempfile.TemporaryDirectory", Keep)
    result = await run_scenario(cu_click_then_done())
    files = list(Path(seen["root"]).glob("runs/*/efficiency.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8")) == result.efficiency


def test_get_run_endpoints_carry_efficiency(tmp_path) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    from operon.api.server import app
    from operon.models.common import RunStatus
    from operon.models.state import AgentState

    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(json.dumps(_step(CLICK_A)) + "\n", encoding="utf-8")
    client = TestClient(app)
    for module, getter, url in (
        ("browser", "get_agent_loop", "/run/r1"),
        ("desktop", "get_desktop_agent_loop", "/desktop/run/r1"),
    ):
        for status, expect in ((RunStatus.FAILED, True), (RunStatus.RUNNING, False)):
            state = AgentState(run_id="r1", intent="x", status=status)
            store = SimpleNamespace(get_run=AsyncMock(return_value=state), root_dir=tmp_path)
            with patch(f"operon.api.routes.{module}.{getter}") as loop:
                loop.return_value.run_store = store
                body = client.get(url).json()
            assert (body["efficiency"] is not None) is expect, (module, status)


async def test_runaway_revisits_are_wasted_and_honest_scrolls_are_not() -> None:
    from evals.scenarios import (
        cu_multi_field_form,
        cu_scroll_twice,
        cu_scroll_twice_revisit,
    )

    runaway = (await run_scenario(cu_relaunch_runaway())).efficiency
    assert runaway is not None
    assert runaway["wasted_steps"] >= 9 and runaway["wasted_revisit"] >= 9
    for factory in (cu_scroll_twice, cu_scroll_twice_revisit, cu_multi_field_form):
        efficiency = (await run_scenario(factory())).efficiency
        assert efficiency is not None and efficiency["wasted_steps"] == 0, factory.__name__


def test_revisit_list_is_counted_and_missing_screens_fail_open(tmp_path) -> None:
    rows = [_step(CLICK_A), _step(CLICK_B), _step(CLICK_A)]
    result = compute_efficiency(rows, revisits=[None, False, True])
    assert result["wasted_revisit"] == 1 and result["wasted_steps"] == 1
    # no screenshots on disk: run_efficiency never counts a revisit
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert run_efficiency(tmp_path)["wasted_revisit"] == 0


def _count_saves(monkeypatch) -> list[str]:
    from operon.agent.loop import AgentLoop

    calls: list[str] = []
    original = AgentLoop._save_efficiency

    async def spy(self, state) -> None:
        calls.append(state.run_id)
        await original(self, state)

    monkeypatch.setattr(AgentLoop, "_save_efficiency", spy)
    return calls


async def test_save_efficiency_runs_once_on_gate_cancel(tmp_path, monkeypatch) -> None:
    from evals import harness
    from operon.models.common import RunStatus, RunTaskRequest, StepRequest

    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    calls = _count_saves(monkeypatch)
    scenario = cu_click_then_done()
    world = harness._World(scenario, tmp_path)
    loop, _executor, client = harness._build_loop(scenario, world, tmp_path)
    run = await loop.start_run(RunTaskRequest(intent=scenario.intent, max_steps=5))
    original_turn = client.generate_turn

    async def stop_while_thinking(*, contents):
        await loop.run_store.set_status(run.run_id, RunStatus.CANCELLED)
        return await original_turn(contents=contents)

    client.generate_turn = stop_while_thinking
    response = await loop.step_run(StepRequest(run_id=run.run_id))
    assert response.status is RunStatus.CANCELLED
    assert calls == [run.run_id]


async def test_save_efficiency_runs_once_on_pre_step_failure(tmp_path, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from evals import harness
    from operon.agent.perception import PerceptionLowQualityError
    from operon.models.common import RunStatus, RunTaskRequest, StepRequest

    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    calls = _count_saves(monkeypatch)
    scenario = cu_click_then_done()
    world = harness._World(scenario, tmp_path)
    loop, _executor, _client = harness._build_loop(scenario, world, tmp_path)
    run = await loop.start_run(RunTaskRequest(intent=scenario.intent, max_steps=5))
    monkeypatch.setattr(loop, "_perceive_with_liveness_retry", AsyncMock(side_effect=PerceptionLowQualityError("no elements")))
    response = await loop.step_run(StepRequest(run_id=run.run_id))
    assert response.status is RunStatus.FAILED
    assert calls == [run.run_id]


async def test_save_efficiency_never_runs_on_a_pause(monkeypatch) -> None:
    from evals.scenarios import trust_gate_declined

    calls = _count_saves(monkeypatch)
    result = await run_scenario(trust_gate_declined())
    assert result.pauses == 1 and result.status == "waiting_for_user"
    assert calls == []
