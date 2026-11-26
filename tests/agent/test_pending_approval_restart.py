"""A paused run keeps what it waits for across a restart, and forgets it when it ends (KAN-226)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operon.agent.policy.trust_gate import TrustGate
from operon.api.frontdoor import _waiting
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.models.policy import ActionType
from operon.models.state import AgentState
from operon.store.approval_audit import read_events
from operon.store.background_writer import bg_writer
from operon.store.checkpoints import mark_interrupted_runs
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop

URL = "https://shop.example.com/checkout"
CONFIRM = TrustGate(enabled=True, confirm=("checkout",))


def _hints(loop) -> list[str]:
    hints: list[str] = []
    loop.perception_service.add_advisory_hints = lambda items, **_: hints.extend(items)
    return hints


async def _paused_on_start_url(monkeypatch: pytest.MonkeyPatch, root: Path) -> str:
    loop, executor = build_loop(monkeypatch, root, [CLICK_SAVE], gate=CONFIRM)
    response = await loop.start_run(RunTaskRequest(intent="Look around", start_url=URL))
    assert response.status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []
    await bg_writer.flush()
    return response.run_id


def _restart(monkeypatch: pytest.MonkeyPatch, root: Path, actions=(CLICK_SAVE,), gate=CONFIRM):
    """A fresh loop and run store over the same directory: the restarted process."""
    loop, executor = build_loop(monkeypatch, root, list(actions), gate=gate)
    return loop, executor, _hints(loop)


def _on_disk(root: Path, run_id: str) -> AgentState:
    return AgentState.model_validate_json((root / run_id / "state.json").read_text(encoding="utf-8"))


def _outcomes(root: Path, run_id: str) -> list[str]:
    return [e["outcome"] for e in read_events(root / run_id / "approval_audit.jsonl")]


async def test_approving_after_a_restart_opens_the_start_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = await _paused_on_start_url(monkeypatch, tmp_path)
    loop, executor, _ = _restart(monkeypatch, tmp_path)
    configured: list[str] = []
    executor.configure_run = lambda rid, **_: configured.append(rid)
    record = await loop.run_store.get_run(run_id)

    waiting = _waiting(loop, record)
    assert waiting is not None and waiting["kind"] == "approval" and waiting["gate"] == "trust"
    assert waiting["verdict"] == "confirm"
    assert URL in waiting["reason"]

    resumed = await loop.resume_run(run_id)

    assert resumed.status is RunStatus.RUNNING
    assert [a.url for a in executor.executed] == [URL]
    assert configured == [run_id]  # the new executor was told about the run first
    assert record.approval_count == 1
    assert loop.pending_gate(run_id) is None
    await bg_writer.flush()
    assert _outcomes(tmp_path, run_id) == ["requested", "approved"]
    assert _on_disk(tmp_path, run_id).pending_approval is None


async def test_declining_after_a_restart_leaves_the_page_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = await _paused_on_start_url(monkeypatch, tmp_path)
    loop, executor, hints = _restart(monkeypatch, tmp_path)

    await loop.resume_run(run_id, approve_gated_action=False)

    assert executor.executed == []
    assert any("did not approve opening the start page" in h for h in hints)
    await bg_writer.flush()
    assert _outcomes(tmp_path, run_id) == ["requested", "declined"]


async def test_an_interrupted_waiting_run_keeps_what_it_waits_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = await _paused_on_start_url(monkeypatch, tmp_path)
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    assert mark_interrupted_runs(tmp_path) == [run_id]
    loop, executor, _ = _restart(monkeypatch, tmp_path)

    kept = await loop.resume_interrupted(run_id)

    assert kept.status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []
    record = await loop.run_store.get_run(run_id)
    assert loop.pending_gate(run_id) == "trust"
    assert (record.pause_reason or "").startswith("Resumed after a restart.") and URL in record.pause_reason

    await loop.resume_run(run_id)

    assert [a.url for a in executor.executed] == [URL]
    assert _on_disk(tmp_path, run_id).pending_approval is None


async def test_a_trust_pause_on_a_step_keeps_its_verdict_across_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = TrustGate(enabled=True, deny=("purchase",))
    loop, _ = build_loop(monkeypatch, tmp_path, [PURCHASE], gate=gate)
    run_id = loop.run_store.create_run("Buy it").run_id
    paused = await loop.step_run(StepRequest(run_id=run_id))
    assert paused.status is RunStatus.WAITING_FOR_USER
    await bg_writer.flush()

    restarted, _, _ = _restart(monkeypatch, tmp_path, [PURCHASE], gate)
    waiting = _waiting(restarted, await restarted.run_store.get_run(run_id))

    assert waiting is not None
    assert (waiting["kind"], waiting["gate"], waiting["verdict"]) == ("approval", "trust", "deny")


async def test_an_approved_step_gate_lets_the_same_action_through_once_after_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [PURCHASE])
    run_id = loop.run_store.create_run("Buy it").run_id
    await loop.step_run(StepRequest(run_id=run_id))
    await bg_writer.flush()

    gate = TrustGate(enabled=True, confirm=("purchase",))
    restarted, executor, _ = _restart(monkeypatch, tmp_path, [PURCHASE, PURCHASE], gate)
    await restarted.resume_run(run_id)
    await restarted.step_run(StepRequest(run_id=run_id))

    assert [a.text for a in executor.executed if a.action_type is ActionType.TYPE] == ["purchase now"]


@pytest.mark.parametrize("end", ["succeeded", "cancelled"])
async def test_a_finished_run_forgets_what_it_waited_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, end: str
) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE], gate=CONFIRM)
    run_id = (await loop.start_run(RunTaskRequest(intent="Look", start_url=URL))).run_id
    loop._approved_gate_actions[run_id] = ("f", frozenset({"trust"}))
    loop._trust_approved[run_id] = True
    loop._pause_requests.add(run_id)
    status = RunStatus.SUCCEEDED if end == "succeeded" else RunStatus.CANCELLED
    await loop.run_store.set_status(run_id, status)

    await loop._cleanup_completed_run(run_id)

    for held in (
        loop._pending_gate_approvals,
        loop._pending_start_urls,
        loop._pending_audit,
        loop._approved_gate_actions,
        loop._trust_approved,
    ):
        assert run_id not in held
    assert run_id not in loop._pause_requests and run_id not in loop._created_runs
    await bg_writer.flush()
    assert _on_disk(tmp_path, run_id).pending_approval is None
    assert _outcomes(tmp_path, run_id)[-1] == "cancelled"


async def test_a_takeover_does_not_leave_an_approval_for_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = await _paused_on_start_url(monkeypatch, tmp_path)
    loop, _, _ = _restart(monkeypatch, tmp_path)
    await loop.take_over(run_id)
    await bg_writer.flush()

    assert _on_disk(tmp_path, run_id).pending_approval is None


def test_a_state_saved_before_the_field_still_loads() -> None:
    raw = AgentState(run_id="r", intent="x", status=RunStatus.RUNNING).model_dump(mode="json")
    del raw["pending_approval"]

    assert AgentState.model_validate(json.loads(json.dumps(raw))).pending_approval is None
