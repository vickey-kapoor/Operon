"""Real Chromium regression tests; no LLM credentials or external sites required.

Run after ``playwright install chromium`` with:
    OPERON_RUN_BROWSER_TESTS=true python -m pytest tests/browser/test_real_browser.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from operon.executor.browser_native import NativeBrowserExecutor
from operon.models.policy import AgentAction

pytestmark = pytest.mark.skipif(
    os.getenv("OPERON_RUN_BROWSER_TESTS", "").lower() != "true",
    reason="Opt in with OPERON_RUN_BROWSER_TESTS=true; requires installed Chromium",
)


@pytest.mark.asyncio
async def test_popup_becomes_active_page_for_next_action(tmp_path: Path) -> None:
    """A real target=_blank click must direct the next action into the new tab."""
    (tmp_path / "index.html").write_text(
        '<html><body style="margin:0"><a href="popup.html" target="_blank" '
        'style="display:block;width:200px;height:100px">Open tab</a></body></html>',
        encoding="utf-8",
    )
    popup_path = tmp_path / "popup.html"
    popup_path.write_text(
        '<html><body style="margin:0"><input id="entry" '
        'style="width:300px;height:100px"></body></html>',
        encoding="utf-8",
    )
    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path / "artifacts",
        headless=True,
        record_video=False,
    )
    run_id = "popup-regression"
    executor.configure_run(run_id, headless=True, mode="batch")
    executor.set_current_run_id(run_id)
    try:
        result = await executor.execute(
            AgentAction(action_type="navigate", url=(tmp_path / "index.html").as_uri())
        )
        assert result.success, result.detail
        original = executor._sessions[run_id].page
        async with original.expect_popup(timeout=5000) as popup_event:
            result = await executor.execute(AgentAction(action_type="click", x=50, y=50))
        assert result.success, result.detail
        popup = await popup_event.value
        await popup.wait_for_load_state("domcontentloaded")
        assert await executor.current_url_for_run(run_id) == popup_path.as_uri()
        result = await executor.execute(
            AgentAction(action_type="type", x=50, y=50, text="new tab input")
        )
        assert result.success, result.detail
        # DOM access is only a test oracle; Operon's actions use coordinates.
        assert await popup.input_value("#entry") == "new tab input"
    finally:
        await executor.aclose_run(run_id)
        await executor.aclose()
