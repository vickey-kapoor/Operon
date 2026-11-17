"""Resume keeps a desktop run in the environment provider it started in (KAN-80): a run
begun with --env windows-sandbox must not silently continue on the live host."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operon.models.common import RunStatus
from operon.store.checkpoints import mark_interrupted_runs
from tests.agent.test_checkpoints import STOP, _crash
from tests.fixtures.frontdoor import build_loop


def _interrupted(
    tmp_path: Path, *, environment: str | None, provider: str | None, record: bool = True
) -> str:
    run_id = _crash(tmp_path, RunStatus.RUNNING, environment=environment)
    settings = tmp_path / run_id / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    data.pop("OPERON_ENVIRONMENT", None)
    if record and provider is not None:
        data["OPERON_ENVIRONMENT"] = provider
    settings.write_text(json.dumps(data), encoding="utf-8")
    mark_interrupted_runs(tmp_path)
    return run_id


async def test_sandbox_run_is_not_resumed_on_the_live_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])

    with pytest.raises(ValueError, match=r"windows-sandbox.*live"):
        await loop.resume_interrupted(run_id)
    with pytest.raises(ValueError, match="windows-sandbox"):
        await loop.resume_run(run_id)  # the other way in refuses too

    state = await loop.run_store.get_run(run_id)
    assert state.status is RunStatus.WAITING_FOR_USER and state.interrupted_from is RunStatus.RUNNING
    assert executor.executed == []


async def test_same_provider_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "windows-sandbox")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])

    response = await loop.resume_interrupted(run_id)

    assert response.status is RunStatus.RUNNING


async def test_browser_run_is_unaffected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="browser", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])

    assert (await loop.resume_interrupted(run_id)).status is RunStatus.RUNNING


async def test_desktop_run_without_a_recorded_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider=None, record=False)
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])

    with pytest.raises(ValueError, match="no recorded environment provider"):
        await loop.resume_interrupted(run_id)
    assert executor.executed == []
