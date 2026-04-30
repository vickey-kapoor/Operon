"""Opt-in Browserbase integration smoke tests.

These tests validate the real Browserbase CDP path with live credentials.
They are skipped by default and are intended for manual runs or a future
secret-backed CI lane.

Enable with:

    OPERON_RUN_BROWSERBASE_INTEGRATION=true
    BROWSERBASE_API_KEY=...
    BROWSERBASE_PROJECT_ID=...

Optional:

    OPERON_BROWSERBASE_SMOKE_URL=https://example.com
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.executor.browserbase_native import BrowserbaseNativeBrowserExecutor
from src.models.policy import ActionType, AgentAction

_RUN_BROWSERBASE_INTEGRATION = (
    os.getenv("OPERON_RUN_BROWSERBASE_INTEGRATION", "false").lower() == "true"
)
_BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
_BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")
_SMOKE_URL = os.getenv("OPERON_BROWSERBASE_SMOKE_URL", "https://example.com")

pytestmark = [
    pytest.mark.skipif(
        not _RUN_BROWSERBASE_INTEGRATION,
        reason="Set OPERON_RUN_BROWSERBASE_INTEGRATION=true to run Browserbase smoke tests.",
    ),
]


@pytest.fixture
def require_browserbase_env() -> None:
    missing: list[str] = []
    if not _BROWSERBASE_API_KEY:
        missing.append("BROWSERBASE_API_KEY")
    if not _BROWSERBASE_PROJECT_ID:
        missing.append("BROWSERBASE_PROJECT_ID")
    if missing:
        pytest.skip(f"Missing Browserbase credentials: {', '.join(missing)}")


@pytest.mark.asyncio
async def test_browserbase_executor_can_navigate_and_capture(
    tmp_path: Path, require_browserbase_env: None
) -> None:
    """Smoke test the real Browserbase-backed navigation/capture path."""
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key=_BROWSERBASE_API_KEY,
        project_id=_BROWSERBASE_PROJECT_ID,
        headless=True,
    )
    run_id = "browserbase-smoke"
    executor.set_current_run_id(run_id)

    try:
        navigate = await executor.execute(
            AgentAction(action_type=ActionType.NAVIGATE, url=_SMOKE_URL)
        )
        assert navigate.success, navigate.detail

        current_url = await executor.current_url_for_run(run_id)
        assert current_url is not None
        assert current_url.startswith(("http://", "https://"))

        frame = await executor.capture()
        assert Path(frame.artifact_path).exists()
        assert frame.mime_type == "image/png"
    finally:
        await executor.aclose_run(run_id)


@pytest.mark.asyncio
async def test_browserbase_cleanup_releases_remote_session(
    tmp_path: Path, require_browserbase_env: None
) -> None:
    """Smoke test that Browserbase sessions are tracked and cleaned up."""
    executor = BrowserbaseNativeBrowserExecutor(
        artifact_dir=tmp_path,
        api_key=_BROWSERBASE_API_KEY,
        project_id=_BROWSERBASE_PROJECT_ID,
        headless=True,
    )
    run_id = "browserbase-cleanup"
    executor.set_current_run_id(run_id)

    await executor.execute(AgentAction(action_type=ActionType.NAVIGATE, url=_SMOKE_URL))
    assert run_id in executor._bb_session_ids

    closed = await executor.aclose_run(run_id)

    assert closed == 1
    assert run_id not in executor._bb_session_ids
