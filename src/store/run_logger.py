"""Helpers for writing per-run JSONL step logs."""

from __future__ import annotations

from pathlib import Path

from src.models.logs import RunLogEntry
from src.store.background_writer import bg_writer


def append_step_log(log_path: str | Path, entry: RunLogEntry) -> None:
    """Append one run-log entry to a JSONL file (non-blocking)."""
    path = Path(log_path)
    bg_writer.append(path, entry.model_dump_json() + "\n")


def append_step_log_critical(log_path: str | Path, entry: RunLogEntry) -> None:
    """Append a critical log entry synchronously with immediate flush.

    Use for terminal_failure and perception_low_quality signals so external
    watchers see the entry in <1s without waiting for the thread pool.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = entry.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
