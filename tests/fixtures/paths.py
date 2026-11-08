"""Shared test path helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from operon.core.paths import test_artifacts_dir


def unique_artifact_dir(name: str) -> Path:
    path = test_artifacts_dir() / f"{name}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_artifact_path(run_id: str = "run-1", step: int = 1, filename: str = "before.png") -> str:
    return f"runs/{run_id}/step_{step}/{filename}"

