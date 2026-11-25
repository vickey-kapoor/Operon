"""Independent check of the model's "done" (KAN-208): offline, scripted model and scripted checker."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import evals.harness as harness
from evals.harness import Element, Scenario, Screen, cu_call, cu_text, run_scenario
from operon.agent import done_check
from operon.agent.done_check import (
    MAX_DONE_CHECKS,
    check_done,
    parse_verdict,
    rejection_note,
)
from operon.api.observer import summarize_run_usage
from operon.models.usage import ModelUsage

ACCEPT = json.dumps({"done": True, "reason": "The page shows the result."})
REJECT = json.dumps({"done": False, "reason": "The form was never submitted."})


def _usage() -> ModelUsage:
    return ModelUsage(
        provider="gemini",
        model="gemini-3.5-flash",
        request_kind="image",
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        estimated_cost_usd=0.001,
    )


class FakeChecker:
    """A scripted checker client: one reply (or exception) per call, the last one repeats."""

    def __init__(self, *replies, usage: ModelUsage | None = None):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self._usage = usage

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        self.calls.append((prompt, screenshot_path))
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def latest_usage(self):
        return self._usage


# ── parse_verdict ─────────────────────────────────────────────────────────


def test_parse_verdict_reads_the_fixed_json() -> None:
    assert parse_verdict('{"done": true, "reason": "  all   good "}') == (True, "all good")
    assert parse_verdict('{"done": false, "reason": "x"}') == (False, "x")
    assert parse_verdict('```json\n{"done": false, "reason": "no"}\n```') == (False, "no")
    assert parse_verdict('{"done": true}') == (True, "")
    long = parse_verdict(json.dumps({"done": False, "reason": "y" * 900}))
    assert long is not None and len(long[1]) == 200


@pytest.mark.parametrize(
    "raw", ["", "not json", "[1]", '{"done": "yes"}', '{"done": 1}', '{"reason": "x"}', "null"]
)
def test_parse_verdict_rejects_unusable_answers(raw) -> None:
    assert parse_verdict(raw) is None


def test_rejection_note_carries_the_reason() -> None:
    note = rejection_note("The form was never submitted.")
    assert "The form was never submitted." in note and "not done yet" in note


# ── check_done ────────────────────────────────────────────────────────────


def test_check_done_accepts_and_rejects(tmp_path: Path) -> None:
    shot = tmp_path / "s.png"
    ok = asyncio.run(check_done(FakeChecker(ACCEPT, usage=_usage()), "Open Products", "Done.", str(shot)))
    assert (ok.verdict, ok.reason) == ("accepted", "The page shows the result.")
    assert ok.usage is not None and ok.raw == ACCEPT
    client = FakeChecker(REJECT)
    no = asyncio.run(check_done(client, "Open Products", "All done", str(shot)))
    assert (no.verdict, no.reason) == ("rejected", "The form was never submitted.")
    prompt, path = client.calls[0]
    assert "Open Products" in prompt and "All done" in prompt and path == str(shot)


def test_check_done_failures_are_errors_never_raises() -> None:
    bad = asyncio.run(check_done(FakeChecker("nonsense"), "t", "d", "s.png"))
    assert bad.verdict == "error" and bad.error and bad.raw == "nonsense"
    boom = asyncio.run(check_done(FakeChecker(RuntimeError("boom")), "t", "d", "s.png"))
    assert boom.verdict == "error" and "boom" in (boom.error or "") and boom.raw is None


def test_check_done_times_out(monkeypatch) -> None:
    class Slow(FakeChecker):
        async def generate_perception(self, prompt, screenshot_path):
            await asyncio.sleep(5)
            return ACCEPT

    monkeypatch.setattr(done_check, "DONE_CHECK_DEADLINE_S", 0.05)
    result = asyncio.run(check_done(Slow(), "t", "d", "s.png"))
    assert result.verdict == "error" and "TimeoutError" in (result.error or "")


# ── The real loop, scripted ───────────────────────────────────────────────


SUBMIT = Element("submit", "Submit", 40, 80, width=160)


def _scenario(*, done_check_on: bool = True, max_steps: int = 8, says: int = 5) -> Scenario:
    return Scenario(
        name="done_check",
        description="The model says it is done again and again on a screen that never changes.",
        intent="Submit the form",
        backend="computer_use",
        screens=[Screen("form", "A form", (SUBMIT,), goal=True)],
        transitions=[],
        # A text-only reply is "done" only after the model has acted: one click, then it says done.
        model={
            "form": [cu_call("click_at", x=SUBMIT.center[0], y=SUBMIT.center[1])]
            + [cu_text(f"I submitted the form ({i}). The task is complete.") for i in range(1, says + 1)]
        },
        max_steps=max_steps,
        env={"OPERON_DONE_CHECK": "on" if done_check_on else "off"},
    )


def _run(monkeypatch, tmp_path: Path, scenario: Scenario, checker: FakeChecker | None):
    """Run the scenario through the real loop with ``checker`` as the done-check client."""
    hints: list[tuple[str, str]] = []
    original = harness._build_loop

    def build(scn, world, root):
        loop, executor, client = original(scn, world, root)
        loop.done_check_client = checker
        add = loop.perception_service.add_advisory_hints

        def spy(items, source="", run_id=""):
            hints.extend((h, source) for h in items)
            add(items, source=source, run_id=run_id)

        loop.perception_service.add_advisory_hints = spy
        return loop, executor, client

    monkeypatch.setattr(harness, "_build_loop", build)
    result = asyncio.run(run_scenario(scenario, root=tmp_path))
    state = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
    return result, state, hints


def _check_files(tmp_path: Path, name: str = "done_check.json") -> list[Path]:
    return sorted((tmp_path / "runs").glob(f"*/step_*/{name}"), key=lambda p: int(p.parent.name.split("_")[1]))


def test_flag_off_never_calls_the_checker(monkeypatch, tmp_path: Path) -> None:
    checker = FakeChecker(REJECT)
    result, state, hints = _run(monkeypatch, tmp_path, _scenario(done_check_on=False), checker)
    assert result.status == "succeeded" and result.steps == 2
    assert checker.calls == []
    assert state["done_checks"] == {}
    assert not [h for h in hints if h[1] == "done_check"]
    assert not list((tmp_path / "runs").glob("*/step_*/done_check*"))
    timing = json.loads(next((tmp_path / "runs").glob("*/step_1/step_timing.json")).read_text(encoding="utf-8"))
    assert "done_check" not in json.dumps(timing)


def test_accepted_done_ends_the_run_as_checked(monkeypatch, tmp_path: Path) -> None:
    checker = FakeChecker(ACCEPT, usage=_usage())
    result, state, hints = _run(monkeypatch, tmp_path, _scenario(), checker)
    assert (result.status, result.steps, len(checker.calls)) == ("succeeded", 2, 1)
    assert result.stop_reason == "task_completed"
    assert state["done_checks"] == {"checked": 1, "accepted": 1}
    assert state["verification_history"][-1]["unchecked"] is False
    assert state["verification_history"][-1]["critic_model_used"] is True
    assert not [h for h in hints if h[1] == "done_check"]
    (path,) = _check_files(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "accepted" and saved["done"] is True and saved["check_number"] == 1
    assert saved["usage"]["model"] == "gemini-3.5-flash" and saved["screenshot"]
    step = path.parent
    assert (step / "done_check_usage.json").exists() and (step / "done_check_raw.txt").exists()
    assert (step / "done_check_prompt.txt").exists()
    # The check's cost is part of the run's cost.
    from operon.api import observer

    monkeypatch.setattr(observer, "runs_root", lambda: tmp_path / "runs")
    assert summarize_run_usage(path.parents[1].name).request_count >= 1


def test_rejected_done_goes_back_to_the_model_as_a_hint(monkeypatch, tmp_path: Path) -> None:
    checker = FakeChecker(REJECT, ACCEPT)
    result, state, hints = _run(monkeypatch, tmp_path, _scenario(), checker)
    assert (result.status, result.steps, len(checker.calls)) == ("succeeded", 3, 2)
    assert state["done_checks"] == {"checked": 2, "rejected": 1, "accepted": 1}
    sent = [h for h, source in hints if source == "done_check"]
    assert len(sent) == 1 and "The form was never submitted." in sent[0]
    # The model saw the note on its next step.
    assert "The form was never submitted." in json.dumps(result.model_requests[2])
    first = state["verification_history"][1]
    assert first["status"] == "uncertain" and first["stop_condition_met"] is False
    assert first["failure_category"] is None and first["stop_reason"] is None
    assert state["failure_cause"] is None
    saved = json.loads(_check_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert saved["verdict"] == "rejected" and saved["done"] is False


def test_the_second_rejection_still_leaves_the_third_done_standing(monkeypatch, tmp_path: Path) -> None:
    checker = FakeChecker(REJECT)  # rejects every time it is asked
    result, state, hints = _run(monkeypatch, tmp_path, _scenario(), checker)
    assert MAX_DONE_CHECKS == 2
    assert (result.status, result.steps, len(checker.calls)) == ("succeeded", 4, 2)
    assert state["done_checks"] == {"checked": 2, "rejected": 2}
    assert len([h for h in hints if h[1] == "done_check"]) == 2
    last = state["verification_history"][-1]
    assert last["unchecked"] is True and last["critic_model_used"] is False
    assert len(_check_files(tmp_path)) == 2


def test_a_checker_failure_leaves_the_done_standing_and_unverified(monkeypatch, tmp_path: Path) -> None:
    checker = FakeChecker(RuntimeError("quota"))
    result, state, hints = _run(monkeypatch, tmp_path, _scenario(), checker)
    assert (result.status, result.steps, len(checker.calls)) == ("succeeded", 2, 1)
    assert state["done_checks"] == {"checked": 1, "error": 1}
    last = state["verification_history"][-1]
    assert last["unchecked"] is True and last["critic_model_used"] is False
    assert not [h for h in hints if h[1] == "done_check"]
    (path,) = _check_files(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "error" and saved["done"] is None and "quota" in saved["error"]
    # No reply arrived: no raw answer, no prompt file, nothing to flag as a cost gap.
    assert not (path.parent / "done_check_raw.txt").exists()


def test_an_unusable_answer_is_a_recorded_error(monkeypatch, tmp_path: Path) -> None:
    result, state, _ = _run(monkeypatch, tmp_path, _scenario(), FakeChecker("I think so", usage=_usage()))
    assert result.status == "succeeded"
    assert state["done_checks"] == {"checked": 1, "error": 1}
    assert state["verification_history"][-1]["unchecked"] is True


def test_a_done_on_the_last_step_is_not_checked(monkeypatch, tmp_path: Path) -> None:
    # A rejection there could only turn a finished run into a max-steps failure.
    checker = FakeChecker(REJECT)
    result, state, _ = _run(monkeypatch, tmp_path, _scenario(max_steps=2), checker)
    assert (result.status, checker.calls) == ("succeeded", [])
    assert state["done_checks"] == {}


def test_a_model_that_gave_up_is_not_checked(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from operon.agent.loop import AgentLoop
    from operon.agent.policy.thin import OwnDoneVerifier
    from operon.models.policy import ActionType, AgentAction, PolicyDecision

    monkeypatch.setenv("OPERON_DONE_CHECK", "on")
    checker = FakeChecker(ACCEPT)
    loop = AgentLoop.__new__(AgentLoop)
    loop.done_check_client = checker
    loop.perception_service = MagicMock()
    decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.STOP), rationale="Model gave up: cannot do it", confidence=0.5, active_subgoal="stop"
    )
    verification = asyncio.run(OwnDoneVerifier().verify(None, decision, SimpleNamespace()))
    ctx = SimpleNamespace(decision=decision, verification=verification, state=SimpleNamespace(done_checks={}))
    asyncio.run(loop._check_done_claim(ctx))
    assert checker.calls == [] and ctx.state.done_checks == {} and ctx.verification is verification


def test_a_missing_screenshot_is_a_recorded_error(monkeypatch, tmp_path: Path) -> None:
    from operon.agent.loop import AgentLoop

    checker = FakeChecker(ACCEPT)
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self.suffix == ".png" else original(self))
    _, state, _ = _run(monkeypatch, tmp_path, _scenario(), checker)
    assert AgentLoop is not None
    assert checker.calls == []
    assert state["done_checks"] == {"checked": 1, "error": 1}


def test_old_state_files_load_without_done_checks() -> None:
    from operon.models.state import AgentState

    state = AgentState(run_id="r", intent="i", status="running")
    assert state.done_checks == {}
    dumped = state.model_dump(mode="json")
    dumped.pop("done_checks")
    assert AgentState.model_validate(dumped).done_checks == {}


def test_a_checker_that_cannot_be_built_is_a_recorded_error(monkeypatch, tmp_path: Path) -> None:
    """Building the default client can fail (e.g. Vertex on without a project): the done stands, as an error."""
    import operon.clients.gemini as gemini

    def cannot_build(*args, **kwargs):
        raise ValueError("no project")

    monkeypatch.setattr(gemini, "GeminiHttpClient", cannot_build)
    result, state, _ = _run(monkeypatch, tmp_path, _scenario(), None)
    assert result.status == "succeeded"
    assert state["done_checks"] == {"checked": 1, "error": 1}
    assert state["verification_history"][-1]["unchecked"] is True
    (path,) = _check_files(tmp_path)
    assert "no project" in json.loads(path.read_text(encoding="utf-8"))["error"]
