"""The public Python SDK (KAN-35), offline: a real AgentLoop with scripted policy and a fake executor."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from operon import Agent, RunResult
from operon.api import ws_stream
from operon.api.runtime import loops
from operon.core.runtime import current_runtime
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
    assert result.final_answer == "It is sunny in Paris." == result.message
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
    assert result.final_answer is None
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
        Agent(env="windows-sandbox")  # env needs desktop=True
    with pytest.raises(ValueError, match="unknown env"):
        Agent(desktop=True, env="no-such-env")
    Agent(desktop=True, env="windows-sandbox")


async def test_cancelling_arun_stops_the_run(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    real_execute = executor.execute

    async def slow(action):
        await asyncio.sleep(0.3)
        return await real_execute(action)

    executor.execute = slow  # type: ignore[method-assign]
    events: list[dict] = []
    monkeypatch.setattr(ws_stream, "publish_event", events.append)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(Agent(mode="auto", poll_s=0.01).arun("Loop forever"), timeout=0.5)
    await asyncio.sleep(0.5)  # a step already in flight may finish
    seen = len(executor.executed)
    await asyncio.sleep(0.8)

    assert len(executor.executed) == seen  # nothing runs after the cancel
    run_id = next(e["run_id"] for e in events if e["type"] == "run_started")
    assert (await loop.run_store.get_run(run_id)).status.value == "cancelled"
    assert not current_runtime().sdk_run_active


async def test_second_concurrent_run_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(current_runtime(), "sdk_run_active", True)
    with pytest.raises(RuntimeError, match="one run at a time"):
        await Agent().arun("x")


@pytest.mark.parametrize("field", ["browser_loop", "desktop_loop", "desktop_provider"])
async def test_loops_the_sdk_did_not_build_are_not_torn_down(monkeypatch, field) -> None:
    runtime = current_runtime()
    sentinel = object()
    monkeypatch.setattr(runtime, field, sentinel)
    with pytest.raises(RuntimeError, match="already holds"):
        await Agent().arun("x")
    assert getattr(runtime, field) is sentinel and not runtime.sdk_run_active


async def test_loads_the_saved_user_env(monkeypatch, tmp_path) -> None:
    from operon.core import paths

    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)
    calls: list[int] = []
    monkeypatch.setattr(paths, "load_user_env", lambda: calls.append(1) or True)
    await Agent(mode="auto", poll_s=0.01).arun("x")
    assert calls == [1]


class _EnvClient:
    """Records OPERON_ENVIRONMENT as the run starts; finishes at once."""

    seen: list[str | None] = []

    async def start(self, payload):
        self.seen.append(os.environ.get("OPERON_ENVIRONMENT"))
        return {"run_id": "r", "done": True, "waiting": None}

    async def report(self, run_id):
        return {"status": "succeeded", "steps": 0, "final_message": "ok"}

    async def stop(self, run_id):
        return {}

    async def close(self):
        pass


async def test_env_is_set_per_run_and_restored(monkeypatch) -> None:
    from operon import cli

    _EnvClient.seen = []
    monkeypatch.setattr(cli, "LocalClient", _EnvClient)
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    await Agent(desktop=True, env="windows-sandbox", poll_s=0.01).arun("a")
    assert os.environ["OPERON_ENVIRONMENT"] == "live"
    monkeypatch.delenv("OPERON_ENVIRONMENT")
    await Agent(desktop=True, env="windows-sandbox", poll_s=0.01).arun("c")
    assert "OPERON_ENVIRONMENT" not in os.environ
    assert _EnvClient.seen[0] == "windows-sandbox" and _EnvClient.seen[-1] == "windows-sandbox"


class _WaitingClient(_EnvClient):
    def __init__(self, kind: str) -> None:
        self.kind, self.approved = kind, 0

    async def start(self, payload):
        return {"run_id": "r", "done": False, "waiting": {"kind": self.kind, "gate": None, "reason": "Confirm it"}}

    async def approve(self, run_id):
        self.approved += 1
        return {"run_id": run_id, "done": True, "waiting": None}

    async def stop(self, run_id):
        return {"run_id": run_id, "done": True, "waiting": None}

    async def report(self, run_id):
        return {"status": "cancelled", "steps": 1, "final_message": "paused"}


async def test_needs_you_goes_to_the_callback_and_takeover_is_stopped() -> None:
    asked: list[ApprovalRequest] = []
    agent = Agent(on_approval=lambda r: asked.append(r) or True, poll_s=0.01)
    client = _WaitingClient("needs_you")
    result = await agent._drive(client, {}, {})
    assert asked[0].kind == "needs_you" and client.approved == 1 and result.final_answer is None

    taken = await agent._drive(_WaitingClient("takeover"), {}, {})
    assert taken.status == "denied" and len(asked) == 1  # the callback is not asked about a takeover


def test_import_operon_is_lazy() -> None:
    code = (
        "import sys, operon;"
        "assert 'operon.agent.loop' not in sys.modules and 'playwright' not in sys.modules;"
        "assert operon.Agent.__name__ == 'Agent'"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
