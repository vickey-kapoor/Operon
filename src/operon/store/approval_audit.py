"""Local approval events for the observer."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from operon.store.background_writer import bg_writer


def append_event(path: Path, event: dict[str, str | int]) -> None:
    bg_writer.append(path, json.dumps({**event, "timestamp": datetime.now(UTC).isoformat()}) + "\n")


def request_event(run_id: str, step_index: int, category: str) -> dict[str, str | int]:
    return {
        "request_id": str(uuid4()),
        "run_id": run_id,
        "step_index": step_index,
        "category": category,
        "outcome": "requested",
    }


def read_events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
