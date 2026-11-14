"""StepContext: the typed state one agent-loop step passes between its stages (KAN-155)."""

import pytest

from operon.agent.step_context import StepContext
from operon.models.common import RunStatus
from operon.models.state import AgentState


def _ctx() -> StepContext:
    record = AgentState(run_id="run-1", intent="do it", status=RunStatus.RUNNING)
    return StepContext(
        record=record,
        step_index=1,
        before_artifact_path="before.png",
        after_artifact_path="after.png",
    )


def test_run_id_comes_from_the_record():
    assert _ctx().run_id == "run-1"


def test_record_stage_accumulates_milliseconds(monkeypatch):
    ctx = _ctx()
    clock = iter([10.5, 11.0])
    monkeypatch.setattr("operon.agent.step_context.time.perf_counter", lambda: next(clock))
    assert ctx.record_stage("execute", 10.0) == pytest.approx(0.5)
    ctx.record_stage("execute", 10.0)
    assert ctx.stage_timings == {"execute": pytest.approx(1500.0)}


def test_stage_fields_are_unset_until_their_stage_runs():
    ctx = _ctx()
    with pytest.raises(AttributeError):
        _ = ctx.perception
    with pytest.raises(AttributeError):
        _ = ctx.last_attempt
    assert ctx.executed_action is None
    assert ctx.final_status is RunStatus.RUNNING
    assert ctx.earlier_attempts == [] and ctx.adaptation_trace == []
