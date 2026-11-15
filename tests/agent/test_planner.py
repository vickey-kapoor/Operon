"""One-shot planner (KAN-112): offline, scripted client."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from operon.agent.loop import AgentLoop
from operon.agent.planner import make_plan, parse_plan, plan_hint


class FakeClient:
    def __init__(self, raw=None, error=None):
        self.raw, self.error, self.prompts = raw, error, []

    async def generate_policy(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.raw

    def latest_usage(self):
        return None


def test_parse_plan_cleans_and_caps():
    raw = json.dumps({"steps": ["  Open   the app ", "", 7, "x" * 500] + ["s"] * 20})
    steps = parse_plan(raw)
    assert steps[0] == "Open the app"
    assert len(steps) == 8
    assert all(len(s) <= 160 for s in steps)


def test_parse_plan_rejects_garbage():
    assert parse_plan("not json") == []
    assert parse_plan('{"steps": "go"}') == []
    assert parse_plan("[1,2]") == []


def test_plan_hint_numbers_steps():
    hint = plan_hint(["Open it", "Type hello"])
    assert "1. Open it; 2. Type hello" in hint and "guide, not an order" in hint


def test_make_plan_happy_path_and_prompt_has_only_the_task():
    client = FakeClient(json.dumps({"steps": ["Open editor", "Type hello", "Check it shows hello"]}))
    steps, _ = asyncio.run(make_plan(client, "type hello in a text editor"))
    assert len(steps) == 3
    assert "type hello in a text editor" in client.prompts[0]


def test_make_plan_failures_give_no_plan():
    assert asyncio.run(make_plan(FakeClient(error=RuntimeError("boom")), "task"))[0] == []
    assert asyncio.run(make_plan(FakeClient('{"steps": ["only one"]}'), "task"))[0] == []
    assert asyncio.run(make_plan(None, "task"))[0] == []
    assert asyncio.run(make_plan(FakeClient("{}"), "  "))[0] == []


def _loop(tmp_path, client):
    loop = AgentLoop.__new__(AgentLoop)
    loop.planner_client = client
    loop.perception_service = MagicMock()
    loop.run_store = SimpleNamespace(root_dir=tmp_path)
    loop._recovered = MagicMock()
    return loop


def _flag(monkeypatch, on):
    monkeypatch.setattr("operon.agent.loop.get_settings", lambda: SimpleNamespace(planner=on, planner_model="m"))


def test_write_plan_off_does_nothing(monkeypatch, tmp_path):
    _flag(monkeypatch, False)
    client = FakeClient("{}")
    loop = _loop(tmp_path, client)
    asyncio.run(loop._write_plan(SimpleNamespace(run_id="r1", intent="t")))
    assert not client.prompts
    loop.perception_service.add_advisory_hints.assert_not_called()


def test_write_plan_on_adds_hint_and_saves_file(monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    loop = _loop(tmp_path, FakeClient(json.dumps({"steps": ["Open it", "Type hello"]})))
    asyncio.run(loop._write_plan(SimpleNamespace(run_id="r1", intent="t")))
    args, kwargs = loop.perception_service.add_advisory_hints.call_args
    assert "1. Open it" in args[0][0] and kwargs == {"source": "planner", "run_id": "r1"}
    saved = json.loads((tmp_path / "r1" / "plan.json").read_text())
    assert saved["steps"] == ["Open it", "Type hello"] and saved["model"] == "m"


def test_write_plan_failure_never_raises(monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    loop = _loop(tmp_path, FakeClient(error=RuntimeError("down")))
    asyncio.run(loop._write_plan(SimpleNamespace(run_id="r1", intent="t")))
    loop.perception_service.add_advisory_hints.assert_not_called()
