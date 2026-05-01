"""Per-step artifact paths and persistence.

Encapsulates the layout under `runs/<run_id>/step_<n>/...` and the persistence
of execution / progress / timing traces. Decouples filesystem concerns from
AgentLoop so the loop only holds a `StepArtifactsManager` and asks it for paths
and writes.

The manager prefers the run_store's path methods when available (lets tests
inject in-memory stores with custom layouts), falling back to a plain
`runs/<run_id>/step_<n>/...` shape otherwise.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.models.progress import ProgressTrace
from src.store.background_writer import bg_writer


class StepArtifactsManager:
    def __init__(self, run_store) -> None:
        self.run_store = run_store

    def before_path(self, run_id: str, step_index: int) -> str:
        if hasattr(self.run_store, "before_artifact_path"):
            return str(self.run_store.before_artifact_path(run_id, step_index))
        return str(Path("runs") / run_id / f"step_{step_index}" / "before.png")

    def after_path(self, run_id: str, step_index: int) -> str:
        if hasattr(self.run_store, "after_artifact_path"):
            return str(self.run_store.after_artifact_path(run_id, step_index))
        return str(Path("runs") / run_id / f"step_{step_index}" / "after.png")

    def run_log_path(self, run_id: str) -> str:
        if hasattr(self.run_store, "run_log_path"):
            return str(self.run_store.run_log_path(run_id))
        return str(Path("runs") / run_id / "run.jsonl")

    def prepare(self, run_id: str, step_index: int, before_path: str, after_path: str) -> None:
        Path(before_path).parent.mkdir(parents=True, exist_ok=True)
        Path(after_path).parent.mkdir(parents=True, exist_ok=True)
        run_log = self.run_log_path(run_id)
        Path(run_log).parent.mkdir(parents=True, exist_ok=True)
        Path(run_log).touch(exist_ok=True)

    def relocate_after_artifact(self, executed_action, planned_path: str):
        if executed_action.artifact_path is None:
            return executed_action.model_copy(update={"artifact_path": planned_path})

        current_path = Path(executed_action.artifact_path)
        target_path = Path(planned_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if current_path != target_path and current_path.exists():
            shutil.move(str(current_path), str(target_path))
        return executed_action.model_copy(update={"artifact_path": str(target_path)})

    def persist_execution_trace(self, run_id: str, step_index: int, executed_action):
        if executed_action.execution_trace is None:
            return executed_action
        # Step dir already created by prepare() — no mkdir needed.
        step_dir = Path(self.before_path(run_id, step_index)).resolve().parent
        trace_path = step_dir / "execution_trace.json"
        bg_writer.enqueue(trace_path, executed_action.execution_trace.model_dump_json())
        return executed_action.model_copy(update={"execution_trace_artifact_path": str(trace_path)})

    def persist_progress_trace(self, run_id: str, step_index: int, progress_trace: ProgressTrace) -> str:
        # Step dir already created by prepare() — no mkdir needed.
        step_dir = Path(self.before_path(run_id, step_index)).resolve().parent
        trace_path = step_dir / "progress_trace.json"
        bg_writer.enqueue(trace_path, progress_trace.model_dump_json())
        return str(trace_path)

    def persist_step_timing(self, run_id: str, step_index: int, stage_timings: dict[str, float]) -> None:
        """Persist per-stage durations as runs/<run_id>/step_<n>/step_timing.json.

        Enables tail-latency analysis and stage-level regression tracking — values
        that previously only appeared on stdout via _trace() when OPERON_TRACE=1.
        """
        if not stage_timings:
            return
        step_dir = Path(self.before_path(run_id, step_index)).resolve().parent
        timing_path = step_dir / "step_timing.json"
        total_ms = sum(stage_timings.values())
        payload = {
            "run_id": run_id,
            "step_index": step_index,
            "total_ms": round(total_ms, 2),
            "stages_ms": {k: round(v, 2) for k, v in stage_timings.items()},
        }
        bg_writer.enqueue(timing_path, json.dumps(payload, sort_keys=True))
