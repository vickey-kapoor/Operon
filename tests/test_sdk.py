"""The public Python SDK (KAN-35), offline: a real AgentLoop with scripted policy and a fake executor."""

from __future__ import annotations

import subprocess
import sys

import pytest

from operon import Agent, RunResult
from operon.api import ws_stream
from operon.api.runtime import loops
from operon.models.common import StopReason
from operon.models.policy import ActionType, AgentAction
from operon.models.verification import VerificationResult, VerificationStatus
from operon.sdk import ApprovalRequest
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop

STOP = AgentAction(action_type=ActionType.STOP, text="It is sunny in Paris.")


class _StopVerifier:
    """Succeeds on a STOP action, else UNCERTAIN (the fixture verifier is always UNCERTAIN)."""

    async def verify(self, state, decision, executed_action) -> VerificationResult:
        done = executed_action.action.action_type is ActionType.STOP
        return VerificationResult(
            status=VerificationStatus.SUCCESS if done else VerificationStatus.UNCERTAIN,
            expected_outcome_met=done,
            stop_condition_met=done,
            stop_reason=StopReason.TASK_COMPLETED if done else None,
            reason="done" if done else "c",
        )

    def latest_debug_artifacts(self):
        return None


def _use(monkeypatch: pytest.MonkeyPatch, loop) -> None:
    loop.verifier_service = _StopVerifier()
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)


async def test_success_result_fields(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE, STOP])
    _use(monkeypatch, loop)

    result = await Agent(mode="auto", poll_s=0.01).arun("Find the weather", url="https://example.test")

    assert isinstance(result, RunResult)
    assert result.status == "succeeded"
    assert result.final_answer == "It is sunny in Paris."
    assert result.steps >= 1
    assert result.run_id and str(result.run_id) in (result.artifacts_dir or "")
    assert len(executor.executed) >= 1
    with pytest.raises(AttributeError):
        result.status = "x"  # type: ignore[misc]  # frozen


async def test_approval_callback_approves(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, STOP])
    _use(monkeypatch, loop)
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    result = await Agent(on_approval=approve, poll_s=0.01).arun("Buy it")

    assert len(seen) == 1 and seen[0].kind == "approval" and seen[0].gate
    assert result.status == "succeeded"


async def test_approval_callback_denies(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, STOP])
    _use(monkeypatch, loop)

    async def deny(request: ApprovalRequest) -> bool:  # async callbacks work too
        return False

    result = await Agent(on_approval=deny, poll_s=0.01).arun("Buy it")

    assert result.status == "denied" and result.denied_reason
    assert executor.executed == []


async def test_no_callback_denies_by_default(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, STOP])
    _use(monkeypatch, loop)

    result = await Agent(poll_s=0.01).arun("Buy it")

    assert result.status == "denied"
    assert executor.executed == []


async def test_run_inside_a_running_loop_raises() -> None:
    with pytest.raises(RuntimeError, match="arun"):
        Agent().run("anything")


def test_run_works_without_a_running_loop(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    assert Agent(mode="auto", poll_s=0.01).run("Find the weather").status == "succeeded"


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        Agent(mode="yolo")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Agent(env="windows-sandbox")


def test_import_operon_is_lazy() -> None:
    code = (
        "import sys, operon;"
        "assert 'operon.agent.loop' not in sys.modules and 'playwright' not in sys.modules;"
        "assert operon.Agent.__name__ == 'Agent'"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
