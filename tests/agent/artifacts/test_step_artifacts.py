"""Per-step artifact paths and trace persistence (artifacts.step_artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

from operon.agent.artifacts.step_artifacts import StepArtifactsManager
from operon.models.execution import ExecutedAction, ExecutionTrace
from operon.models.policy import ActionType, AgentAction
from operon.models.progress import ProgressTrace

CLICK = AgentAction(action_type=ActionType.CLICK, x=1, y=1)


class _Store:
    """A run store that lays steps out under a temp root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def before_artifact_path(self, run_id: str, step_index: int) -> str:
        return str(self.root / run_id / f"step_{step_index}" / "before.png")

    def after_artifact_path(self, run_id: str, step_index: int) -> str:
        return str(self.root / run_id / f"step_{step_index}" / "after.png")

    def run_log_path(self, run_id: str) -> Path:
        return self.root / run_id / "run.jsonl"


def _manager(tmp_path: Path) -> StepArtifactsManager:
    return StepArtifactsManager(_Store(tmp_path))


def test_paths_come_from_the_run_store(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert manager.before_path("r", 2) == str(tmp_path / "r" / "step_2" / "before.png")
    assert manager.after_path("r", 2) == str(tmp_path / "r" / "step_2" / "after.png")
    assert manager.run_log_path("r") == str(tmp_path / "r" / "run.jsonl")


def test_paths_fall_back_to_the_plain_layout_without_store_methods() -> None:
    manager = StepArtifactsManager(object())
    assert Path(manager.before_path("r", 3)) == Path("runs/r/step_3/before.png")
    assert Path(manager.after_path("r", 3)) == Path("runs/r/step_3/after.png")
    assert Path(manager.run_log_path("r")) == Path("runs/r/run.jsonl")


def test_prepare_creates_step_dirs_and_touches_the_run_log(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    before, after = manager.before_path("r", 1), manager.after_path("r", 1)

    manager.prepare("r", 1, before, after)

    assert Path(before).parent.is_dir()
    assert (tmp_path / "r" / "run.jsonl").is_file()
    # Idempotent: a second prepare keeps the existing log.
    (tmp_path / "r" / "run.jsonl").write_text("line\n")
    manager.prepare("r", 1, before, after)
    assert (tmp_path / "r" / "run.jsonl").read_text() == "line\n"


def test_relocate_sets_the_planned_path_when_the_executor_saved_none(tmp_path: Path) -> None:
    planned = str(tmp_path / "r" / "step_1" / "after.png")
    executed = ExecutedAction(action=CLICK, success=True, detail="ok")

    moved = _manager(tmp_path).relocate_after_artifact(executed, planned)

    assert moved.artifact_path == planned
    assert executed.artifact_path is None  # the original is not mutated


def test_relocate_moves_the_executor_screenshot_into_the_step_dir(tmp_path: Path) -> None:
    source = tmp_path / "tmp" / "shot.png"
    source.parent.mkdir()
    source.write_bytes(b"png")
    planned = tmp_path / "r" / "step_1" / "after.png"
    executed = ExecutedAction(action=CLICK, success=True, detail="ok", artifact_path=str(source))

    moved = _manager(tmp_path).relocate_after_artifact(executed, str(planned))

    assert moved.artifact_path == str(planned)
    assert planned.read_bytes() == b"png"
    assert not source.exists()


def test_relocate_with_a_missing_source_still_points_at_the_planned_path(tmp_path: Path) -> None:
    planned = tmp_path / "r" / "step_1" / "after.png"
    executed = ExecutedAction(action=CLICK, success=True, detail="ok", artifact_path=str(tmp_path / "gone.png"))

    moved = _manager(tmp_path).relocate_after_artifact(executed, str(planned))

    assert moved.artifact_path == str(planned)
    assert not planned.exists()


def test_execution_trace_is_written_next_to_the_step(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.prepare("r", 1, manager.before_path("r", 1), manager.after_path("r", 1))
    trace = ExecutionTrace(final_outcome="success")
    executed = ExecutedAction(action=CLICK, success=True, detail="ok", execution_trace=trace)

    persisted = manager.persist_execution_trace("r", 1, executed)

    path = Path(persisted.execution_trace_artifact_path)
    assert path == (tmp_path / "r" / "step_1" / "execution_trace.json").resolve()
    assert json.loads(path.read_text())["final_outcome"] == "success"


def test_no_execution_trace_means_no_file(tmp_path: Path) -> None:
    executed = ExecutedAction(action=CLICK, success=True, detail="ok")

    assert _manager(tmp_path).persist_execution_trace("r", 1, executed) is executed
    assert not (tmp_path / "r").exists()


def test_progress_trace_is_written_and_its_path_returned(tmp_path: Path) -> None:
    trace = ProgressTrace(
        step_index=1, page_signature="page", action_signature="click", subgoal_signature="sub"
    )

    path = _manager(tmp_path).persist_progress_trace("r", 1, trace)

    assert json.loads(Path(path).read_text())["action_signature"] == "click"


def test_step_timing_records_rounded_stages_and_total(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    manager.persist_step_timing("r", 4, {"capture": 10.004, "decide": 20.0})

    payload = json.loads((tmp_path / "r" / "step_4" / "step_timing.json").read_text())
    assert payload == {
        "run_id": "r",
        "step_index": 4,
        "total_ms": 30.0,
        "stages_ms": {"capture": 10.0, "decide": 20.0},
    }


def test_empty_step_timing_writes_nothing(tmp_path: Path) -> None:
    _manager(tmp_path).persist_step_timing("r", 4, {})
    assert not (tmp_path / "r").exists()
