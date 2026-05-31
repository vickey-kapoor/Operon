"""Shared AgentState builders."""

from __future__ import annotations

from operon.models.common import RunStatus
from operon.models.state import AgentState


def agent_state(
    run_id: str = "run-1",
    *,
    intent: str = "Test task",
    status: RunStatus = RunStatus.RUNNING,
    step_count: int = 0,
) -> AgentState:
    return AgentState(run_id=run_id, intent=intent, status=status, step_count=step_count)

