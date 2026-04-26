"""Helpers for writing per-run JSONL step logs."""

from __future__ import annotations

from pathlib import Path

from src.models.logs import RunLogEntry
from src.store.background_writer import bg_writer


def append_step_log(log_path: str | Path, entry: RunLogEntry) -> None:
    """Append one run-log entry to a JSONL file (non-blocking)."""
    path = Path(log_path)
    bg_writer.append(path, entry.model_dump_json() + "\n")
