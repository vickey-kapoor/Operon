"""Shared helpers used across agent service modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.store.background_writer import bg_writer


def collect_latest_usage(client: Any, usage_artifact_path: Path):
    """Read the client's most-recent token-usage record and enqueue it for disk write."""
    if not hasattr(client, "latest_usage"):
        return None
    usage = client.latest_usage()
    if usage is not None:
        bg_writer.enqueue(usage_artifact_path, usage.model_dump_json(indent=2))
    return usage
