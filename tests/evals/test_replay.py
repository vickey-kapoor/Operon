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
