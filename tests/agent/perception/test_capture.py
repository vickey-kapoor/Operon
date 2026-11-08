"""Tests for ScreenCaptureService artifact placement."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.perception.capture import ScreenCaptureService
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.state import AgentState


def _executor(tmp_path: Path) -> SimpleNamespace:
    raw = tmp_path / "raw.png"
    raw.write_bytes(b"png")
    return SimpleNamespace(
        capture=AsyncMock(return_value=CaptureFrame(artifact_path=str(raw), width=10, height=10))
    )


@pytest.mark.asyncio
async def test_capture_defaults_to_run_store_root(tmp_path: Path, monkeypatch) -> None:
    # before.png must land in the same step dir as the run store's after.png,
    # otherwise the verifier and progress tracker silently compare nothing.
    runs_root = tmp_path / "runs-root"
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(runs_root))
    service = ScreenCaptureService(executor=_executor(tmp_path))
    state = AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING, step_count=1)

    frame = await service.capture(state)

    expected = runs_root / "run-1" / "step_2" / "before.png"
    assert Path(frame.artifact_path) == expected
    assert expected.exists()


@pytest.mark.asyncio
async def test_capture_honors_explicit_root(tmp_path: Path) -> None:
    service = ScreenCaptureService(executor=_executor(tmp_path), root_dir=tmp_path / "custom")
    state = AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING, step_count=1)

    frame = await service.capture(state)

    assert Path(frame.artifact_path) == tmp_path / "custom" / "run-1" / "step_2" / "before.png"
