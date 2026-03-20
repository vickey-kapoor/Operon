"""Helpers for writing per-run JSONL step logs."""

from __future__ import annotations

from pathlib import Path

from src.models.logs import StepLog


def append_step_log(log_path: str | Path, entry: StepLog) -> None:
    """Append one strict `StepLog` entry to a run JSONL file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json())
        handle.write("\n")
