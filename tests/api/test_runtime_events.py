"""What a step event tells the task page (operon.api.runtime.events)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operon.agent import tools
from operon.agent.actions.translation import build_policy_decision
from operon.api import ws_stream
from operon.api.runtime import events as runtime_events
from operon.models.common import RunResponse, RunStatus


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(ws_stream, "publish_event", sent.append)
    return sent


def _write_step(root: Path, decision: dict | str, perception: dict | None = None) -> None:
    step_dir = root / "run-1" / "step_1"
    step_dir.mkdir(parents=True)
    text = decision if isinstance(decision, str) else json.dumps(decision)
    (step_dir / "policy_decision.json").write_text(text, encoding="utf-8")
    if perception is not None:
        (step_dir / "perception_parsed.json").write_text(json.dumps(perception), encoding="utf-8")


def _loop(root: Path, executor=None):
    return SimpleNamespace(run_store=SimpleNamespace(root_dir=root), executor=executor or SimpleNamespace())


def _response() -> RunResponse:
    return RunResponse(run_id="run-1", status=RunStatus.RUNNING, intent="x", step_count=1)


@pytest.mark.parametrize(
    ("action", "target", "value"),
    [
        ({"action_type": "navigate", "url": "https://example.com/products?page=2"}, "example.com/products", None),
        ({"action_type": "type", "target_element_id": "email", "text": "ada@example.com"}, "email", "ada@example.com"),
        ({"action_type": "hotkey", "key": "ctrl+a"}, "ctrl+a", None),
        ({"action_type": "launch_app", "text": "notepad"}, "notepad", None),
        ({"action_type": "scroll", "scroll_amount": 300}, "down", None),
        ({"action_type": "scroll", "scroll_amount": -300}, "up", None),
        ({"action_type": "click", "target_element_id": "submit"}, "submit", None),
    ],
)
def test_each_action_type_reports_its_target(action: dict, target: str, value: str | None) -> None:
    assert runtime_events._action_target(action) == (target, value)


def test_a_tool_step_shows_the_tool_call_as_its_target(tmp_path: Path, published) -> None:
    # Built the way the backend builds it, so the event sees the real serialized action.
    decision = build_policy_decision(
        {"action": {"action_type": "tool", "tool": tools.READ_FILE, "tool_args": {"path": "notes.txt"}}, "rationale": "read it", "confidence": 0.8}
    )
    _write_step(tmp_path, decision.model_dump(mode="json"))

    event = runtime_events._build_step_event(_loop(tmp_path), _response())

    assert event["action_type"] == "tool"
    assert event["action_target"] == tools.summary(tools.READ_FILE, {"path": "notes.txt"})
    assert "notes.txt" in event["action_target"]
    assert not [e for e in published if e["type"] == "action_intent"]  # a tool has no point on screen


def test_a_step_event_carries_the_decision_and_the_perception_summary(tmp_path: Path, published) -> None:
    _write_step(
        tmp_path,
        {
            "action": {"action_type": "type", "target_element_id": "email", "text": "ada@example.com"},
            "confidence": 0.9,
            "active_subgoal": "fill the email",
            "rule_name": None,
            "rationale": "the email box is empty",
            "expected_change": "the box shows the address",
        },
        perception={"page_hint": "form", "elements": [{}, {}], "focused_element": {"text": "Email"}},
    )

    event = runtime_events._build_step_event(_loop(tmp_path), _response())

    assert event == {
        "type": "step",
        "run_id": "run-1",
        "step": 1,
        "status": "running",
        "confidence": 0.9,
        "subgoal": "fill the email",
        "action_type": "type",
        "rule_name": None,
        "rationale": "the email box is empty",
        "action_expected_change": "the box shows the address",
        "action_target": "email",
        "action_value": "ada@example.com",
        "perception": "form . focus: Email",
    }


def test_the_perception_summary_counts_elements_when_nothing_has_focus(tmp_path: Path) -> None:
    path = tmp_path / "perception_parsed.json"
    path.write_text(json.dumps({"page_hint": "list", "elements": [{}, {}, {}]}), encoding="utf-8")
    assert runtime_events._read_perception_summary(path) == "list . 3 elements"

    path.write_text("{not json", encoding="utf-8")
    assert runtime_events._read_perception_summary(path) is None
    assert runtime_events._read_perception_summary(tmp_path / "missing.json") is None


@pytest.mark.parametrize("decision", [None, "{not json"])
def test_a_missing_or_broken_decision_still_sends_a_bare_step(tmp_path: Path, published, decision) -> None:
    if decision is None:
        (tmp_path / "run-1" / "step_1").mkdir(parents=True)
    else:
        _write_step(tmp_path, decision)

    event = runtime_events._build_step_event(_loop(tmp_path), _response())

    assert event == {"type": "step", "run_id": "run-1", "step": 1, "status": "running"}
    assert published == []


def test_an_executor_without_a_viewport_falls_back_to_1920x1080(tmp_path: Path, published) -> None:
    _write_step(tmp_path, {"action": {"action_type": "click", "x": 10, "y": 20}, "confidence": 0.7})

    runtime_events._build_step_event(_loop(tmp_path, executor=SimpleNamespace()), _response())

    [intent] = [e for e in published if e["type"] == "action_intent"]
    assert (intent["x"], intent["y"], intent["viewport_w"], intent["viewport_h"]) == (10, 20, 1920, 1080)


class _Store:
    def __init__(self, status: RunStatus | None, fail: bool = False) -> None:
        self.state = None if status is None else SimpleNamespace(run_id="run-1", status=status, intent="x", step_count=3)
        self.fail = fail

    async def get_run(self, run_id: str):
        if self.fail:
            raise OSError("disk gone")
        return self.state

    async def set_status(self, run_id: str, status: RunStatus):
        self.state.status = status
        return self.state


@pytest.mark.parametrize(
    "store",
    [_Store(RunStatus.SUCCEEDED), _Store(RunStatus.CANCELLED), _Store(None), _Store(RunStatus.RUNNING, fail=True)],
    ids=["already-succeeded", "already-cancelled", "gone", "store-error"],
)
async def test_mark_run_failed_leaves_finished_or_missing_runs_alone(published, store: _Store) -> None:
    before = None if store.state is None else store.state.status

    assert await runtime_events.mark_run_failed(SimpleNamespace(run_store=store), "run-1") is None
    assert published == []
    assert (None if store.state is None else store.state.status) == before


async def test_mark_run_failed_fails_a_live_run_and_announces_it(published) -> None:
    store = _Store(RunStatus.RUNNING)

    response = await runtime_events.mark_run_failed(SimpleNamespace(run_store=store), "run-1")

    assert response is not None and response.status is RunStatus.FAILED and response.step_count == 3
    assert published == [{"type": "run_completed", "run_id": "run-1", "status": "failed"}]
