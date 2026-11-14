"""KAN-167: record a run, replay it through the real loop, catch a change in behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evals import replay as rp

# Gemini computer-use answers, in the shape the HTTP client normalizes them to.
_PERCEPTION = {"summary": "screen", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5}


def call(name: str, **args) -> dict:
    fc = {"name": name, "args": args}
    return {"perception": _PERCEPTION, "function_call": fc, "function_calls": [fc], "rationale": name,
            "confidence": 0.7, "active_subgoal": name, "model_content": {"parts": [{"function_call": fc}]}}


def done() -> dict:
    return {"perception": _PERCEPTION, "action": {"action_type": "wait", "wait_ms": 1000},
            "rationale": "The task is complete.", "confidence": 0.5, "active_subgoal": "finish",
            "model_content": {"parts": [{"text": "The task is complete."}]}}


TURNS = [
    call("launch_app", app="notepad"),
    call("type_text_at", x=500, y=500, text="hello", press_enter=False),
    call("key_combination", keys="control+s"),
    done(),
]


def real_run(tmp_path: Path, turns: list[dict] = TURNS) -> dict:
    """A run as a benchmark would leave it (the real loop writing its run folder), recorded."""
    with rp.loop_environment({"OPERON_DESKTOP_PLATFORM": "windows"}):
        run = asyncio.run(rp.run_turns(turns, mode="desktop", intent="Type hello in Notepad and save",
                                       width=1280, height=800, root=tmp_path, max_steps=10))
    assert run.status == "succeeded"
    return rp.record_run(run.run_dir, task="notepad-hello", mode="desktop", outcome="success")


def test_a_recorded_run_replays_the_same(tmp_path):
    recording = real_run(tmp_path)
    assert recording["settings"]["OPERON_HARNESS_PROFILE"] == "thin"
    assert recording["screen"] == {"width": 1280, "height": 800}
    assert len(recording["turns"]) == 4
    assert [s["action"]["action_type"] for s in recording["steps"]] == ["launch_app", "type", "hotkey", "stop"]
    result = asyncio.run(rp.replay(json.loads(json.dumps(recording))))  # survives a JSON round trip
    assert result.ok, result.message


def test_a_different_action_for_the_same_answer_fails_with_the_step(tmp_path):
    recording = real_run(tmp_path)
    recording["steps"][1]["action"]["text"] = "goodbye"  # what the code "used to" type
    result = asyncio.run(rp.replay(recording))
    assert not result.ok
    assert "step 2" in result.message and "'goodbye'" in result.message and "'hello'" in result.message


def test_a_run_that_ends_early_or_late_fails(tmp_path):
    recording = real_run(tmp_path)
    shorter = {**recording, "steps": recording["steps"][:2]}
    result = asyncio.run(rp.replay(shorter))
    assert not result.ok and "no longer ends where the real run did" in result.message

    longer = {**recording, "steps": [*recording["steps"], {"step": 5, "action": {"action_type": "wait"}}]}
    result = asyncio.run(rp.replay(longer))
    assert not result.ok and "now ends after 4 step(s)" in result.message


def test_only_thin_harness_runs_record(tmp_path):
    recording_dir = real_run(tmp_path)  # noqa: F841 - leaves a run folder behind
    run_dir = next((tmp_path / "runs").iterdir())
    settings = json.loads((run_dir / "settings.json").read_text())
    settings["OPERON_HARNESS_PROFILE"] = "full"
    (run_dir / "settings.json").write_text(json.dumps(settings))
    with pytest.raises(rp.RecordingError, match="thin"):
        rp.record_run(run_dir, task="t", mode="desktop")


def test_a_copied_run_folder_still_records(tmp_path):
    """Benchmark artifacts copy run folders elsewhere; the step files are found again inside."""
    real_run(tmp_path)
    run_dir = next((tmp_path / "runs").iterdir())
    moved = tmp_path / "artifact" / "task"
    run_dir.rename(moved) if moved.parent.mkdir(parents=True) is None else None
    recording = rp.record_run(moved, task="t", mode="desktop")
    assert len(recording["turns"]) == 4


@pytest.mark.parametrize("path", rp.recording_paths(), ids=lambda p: p.stem)
def test_saved_recordings_replay_the_same(path):
    """Every recording in evals/recordings/ (real Gemini answers from the nightly run)."""
    result = asyncio.run(rp.replay(json.loads(path.read_text(encoding="utf-8"))))
    assert result.ok, result.message


def test_the_benchmark_saves_a_recording_per_thin_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evals import live_bench

    real_run(tmp_path / "loop")
    run_dir = next((tmp_path / "loop" / "runs").iterdir())
    monkeypatch.setattr(live_bench, "_run_dir", lambda run_id: run_dir if run_id == "r1" else tmp_path / "missing")
    results = [
        SimpleNamespace(task="notepad-hello", attempt=1, run_id="r1", outcome="success"),
        SimpleNamespace(task="never-started", attempt=1, run_id=None, outcome="error"),
        SimpleNamespace(task="gone", attempt=2, run_id="r2", outcome="failure"),
    ]
    target = tmp_path / "report" / "recordings"
    assert live_bench.save_recordings(results, target, mode="hybrid") == 1
    saved = json.loads((target / "notepad-hello.json").read_text())
    assert saved["mode"] == "desktop" and saved["outcome"] == "success"
    assert asyncio.run(rp.replay(saved)).ok


# ── Codex review cases ─────────────────────────────────────────────────────


def _record(tmp_path, turns, *, max_steps=10, outcomes=None, settings=None, expect=None, **run_args):
    with rp.loop_environment({"OPERON_DESKTOP_PLATFORM": "windows", **(settings or {})}):
        run = asyncio.run(rp.run_turns(turns, mode="desktop", intent="task", width=1280, height=800,
                                       root=tmp_path, max_steps=max_steps, outcomes=outcomes, **run_args))
    if expect:
        assert run.status == expect
    return rp.record_run(run.run_dir, task="t", mode="desktop")


def _replays(recording) -> None:
    result = asyncio.run(rp.replay(json.loads(json.dumps(recording))))
    assert result.ok, result.message


def test_repeated_actions_replay_under_the_thin_harness(tmp_path):
    """The full harness blocks a repeated action; the thin harness (what ran) does not."""
    click = call("click_at", x=300, y=300)
    recording = _record(tmp_path, [click, click, click, done()], expect="succeeded")
    assert [s["action"]["action_type"] for s in recording["steps"]][:3] == ["click"] * 3
    _replays(recording)


def test_a_failed_action_replays_as_failed(tmp_path):
    """After a failed action a text-only answer is not a 'done'; the replay keeps the failure."""
    turns = [call("click_at", x=300, y=300), done(), call("click_at", x=310, y=310), done()]
    recording = _record(tmp_path, turns, outcomes=[{"success": False, "detail": "missed"}])
    assert recording["steps"][0]["success"] is False
    _replays(recording)


def test_a_retried_answer_is_recorded_and_replayed(tmp_path):
    """A weak first answer gets a stronger retry in the same step: both answers are kept."""
    weak = {**done(), "confidence": 0.1, "model_content": {"parts": [{"text": "Looking."}]}, "rationale": "Looking."}
    recording = _record(tmp_path, [weak, call("launch_app", app="notepad"), done()], expect="succeeded")
    assert len(recording["turns"]) == 3
    _replays(recording)


def test_a_run_that_ran_out_of_steps_replays(tmp_path):
    turns = [call("click_at", x=100 + i, y=100) for i in range(3)]
    recording = _record(tmp_path, turns, max_steps=3)
    assert recording["max_steps"] == 3 and len(recording["steps"]) == 3
    _replays(recording)


def test_a_confirmation_pause_replays(tmp_path):
    """Gemini's require_confirmation pauses the run (a logged step the executor never sees)."""
    risky = call("click_at", x=300, y=300, safety_decision={"decision": "require_confirmation", "explanation": "buy"})
    recording = _record(tmp_path, [risky, done()])
    assert "wait_for_user" in [s["action"]["action_type"] for s in recording["steps"]]
    _replays(recording)


def test_the_run_settings_are_restored(tmp_path, monkeypatch):
    recording = _record(tmp_path, TURNS, settings={"OPERON_TRUST_GATE": "off"})
    assert recording["settings"]["OPERON_TRUST_GATE"] in ("off", "false")
    assert not any(k in recording["settings"] for k in ("GEMINI_API_KEY", "OPERON_RUNS_ROOT", "OPERON_ALLOWED_HOSTS"))
    monkeypatch.setenv("OPERON_TRUST_GATE", "on")
    with rp.loop_environment(recording["settings"]):
        from operon.core.settings import get_settings

        assert get_settings().effective()["OPERON_TRUST_GATE"] in ("off", False)


def test_backslash_paths_are_found_in_a_copied_folder(tmp_path):
    step = tmp_path / "run" / "step_1"
    step.mkdir(parents=True)
    (step / "computer_use_raw.json").write_text("{}")
    found = rp._local(r"C:\Users\runner\.operon\runs\abc\step_1\computer_use_raw.json", tmp_path / "run")
    assert found == step / "computer_use_raw.json"


def test_a_run_that_ended_waiting_at_a_pause_replays(tmp_path):
    """The benchmark's approval budget ran out: the real run stopped at the pause."""
    risky = call("click_at", x=300, y=300, safety_decision={"decision": "require_confirmation", "explanation": "buy"})
    recording = _record(tmp_path, [risky, done()], resumes_allowed=0, expect="waiting_for_user")
    assert rp.resumes_in(recording) == 0
    _replays(recording)


def test_a_run_folder_records_its_host(tmp_path):
    real_run(tmp_path)
    run_dir = next((tmp_path / "runs").iterdir())
    import os

    assert json.loads((run_dir / "host.json").read_text()) == {
        "os": os.name, "home": str(Path.home()), "cwd": os.getcwd()}


def test_file_roots_are_kept():
    kept = rp.recordable_settings({"OPERON_ALLOWED_FILE_ROOTS": ["C:/Users/runner/Documents"], "OPERON_RUNS_ROOT": "/x"})
    assert kept == {"OPERON_ALLOWED_FILE_ROOTS": "C:/Users/runner/Documents"}


def test_a_secondary_monitor_origin_replays(tmp_path):
    """Clicks on a monitor right of a 1920 px screen carry its offset; the replay must too."""
    on_second = [{"success": True, "monitor_origin": [1920, 0]}] * 3
    recording = _record(tmp_path, [call("click_at", x=500, y=500), call("click_at", x=100, y=100), done()],
                        outcomes=on_second, expect="succeeded")
    assert recording["steps"][0]["monitor_origin"] == [1920, 0]
    assert recording["steps"][0]["action"]["x"] >= 1920
    _replays(recording)


def test_the_run_permission_mode_is_restored(tmp_path):
    recording = _record(tmp_path, TURNS, permission_mode="ask")
    assert recording["permission_mode"] == "ask"
    _replays(recording)


def test_the_trust_gate_sees_the_pages_it_saw(tmp_path):
    """With a domain allowlist, an action on an off-list page pauses; the replay must show
    the gate the same page, or the pause turns into an executed click."""
    turns = [call("click_at", x=300, y=300), call("click_at", x=310, y=310), done()]
    recording = _record(tmp_path, turns, settings={"OPERON_TRUST_GATE": "on", "OPERON_TRUST_ALLOW_DOMAINS": "example.com"},
                        gate_urls=["https://example.com/", "https://evil.test/login"])
    assert recording["gate_urls"][:2] == ["https://example.com/", "https://evil.test/login"]
    assert [s["action"]["action_type"] for s in recording["steps"]][:2] == ["click", "wait_for_user"]
    _replays(recording)
    blind = asyncio.run(rp.replay({**recording, "gate_urls": []}))  # the gate never sees the page
    assert not blind.ok and "step 2" in blind.message


def test_a_run_from_another_os_judges_file_paths_its_way(tmp_path):
    """A Windows run's Documents root authorizes C:\\...\\report.txt; replayed on Linux too."""
    host = {"os": "nt", "home": r"C:\Users\alice"}
    turns = [call("read_file", path=r"C:\Users\alice\Documents\report.txt"),
             call("read_file", path=r"Documents\notes.txt"), done()]  # relative: under the home folder
    settings = {"OPERON_ALLOWED_FILE_ROOTS": r"C:\Users\alice\Documents"}
    with rp.recorded_path_rules(host), rp.loop_environment({"OPERON_DESKTOP_PLATFORM": "windows", **settings}):
        run = asyncio.run(rp.run_turns(turns, mode="desktop", intent="task", width=1280, height=800,
                                       root=tmp_path, max_steps=10))  # the guard as it ran on Windows
    (run.run_dir / "host.json").write_text(json.dumps(host))  # what the Windows machine wrote
    recording = rp.record_run(run.run_dir, task="t", mode="desktop")  # recorded after copying to Linux
    assert recording["host"] == host
    assert [s["action"]["action_type"] for s in recording["steps"]][:2] == ["tool", "tool"]
    _replays(recording)
    as_linux = asyncio.run(rp.replay({**recording, "host": {"os": "posix", "home": "/root"}}))
    assert not as_linux.ok and "step 1" in as_linux.message
    other_home = asyncio.run(rp.replay({**recording, "host": {"os": "nt", "home": None}}))
    assert not other_home.ok and "step 2" in other_home.message

    from operon.agent.policy.file_guard import blocked_file_path
    from operon.models.policy import ActionType, AgentAction

    outside = AgentAction(action_type=ActionType.TOOL, tool="read_file", tool_args={"path": r"C:\Windows\secret.txt"},
                          text="read_file: secret")
    with rp.loop_environment(settings), rp.recorded_path_rules(host):
        assert blocked_file_path(outside, intent="task", run_dir=tmp_path) == r"C:\Windows\secret.txt"
        inside = outside.model_copy(update={"tool_args": {"path": r"documents\report.txt"}})
        assert blocked_file_path(inside, intent="task", run_dir=tmp_path) is None


def test_relative_roots_follow_the_recorded_working_directory(tmp_path):
    """Roots resolve against the run's working directory; tool paths against its home."""
    from operon.agent.policy.file_guard import blocked_file_path
    from operon.models.policy import ActionType, AgentAction

    host = {"os": "nt", "home": r"C:\Users\alice", "cwd": r"C:\Operon"}
    read = AgentAction(action_type=ActionType.TOOL, tool="read_file", tool_args={"path": "report.txt"},
                       text="read_file: report.txt")
    with rp.loop_environment({"OPERON_ALLOWED_FILE_ROOTS": "."}), rp.recorded_path_rules(host):
        assert blocked_file_path(read, intent="task", run_dir=tmp_path) == "report.txt"  # C:\Users\alice\report.txt
        in_cwd = read.model_copy(update={"tool_args": {"path": r"C:\Operon\report.txt"}})
        assert blocked_file_path(in_cwd, intent="task", run_dir=tmp_path) is None
