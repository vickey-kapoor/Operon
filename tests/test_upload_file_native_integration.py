"""Integration tests for upload_file_native against a real headed browser + OS file picker.

These tests require a headed Windows desktop session and are skipped in CI.
Opt in by setting:

    OPERON_RUN_HEADED_INTEGRATION=true

All tests are also auto-skipped by conftest on non-Windows platforms (the word
"windows" appears in the module name pattern).

What these tests cover that unit tests cannot:
  - os_picker_macro against a real OS file dialog (no mock)
  - NativeBrowserExecutor full path: click → OS picker opens → macro types → picker closes → result
  - File actually reflected after the picker closes
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

_HEADED_INTEGRATION = os.getenv("OPERON_RUN_HEADED_INTEGRATION", "false").lower() == "true"
_IS_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
_IS_SAFE_MODE = os.getenv("OPERON_TEST_SAFE_MODE", "false").lower() == "true"

_SKIP_REASON = (
    "Headed integration tests require OPERON_RUN_HEADED_INTEGRATION=true "
    "and a headed Windows desktop session."
)

requires_headed_integration = pytest.mark.skipif(
    not _HEADED_INTEGRATION or _IS_HEADLESS or _IS_SAFE_MODE or os.name != "nt",
    reason=_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_temp_upload_file() -> Path:
    """Create a small temp file to upload during the test."""
    tmp = Path(tempfile.mktemp(suffix=".txt", prefix="operon_upload_test_"))
    tmp.write_text("operon upload integration test payload\n")
    return tmp


def _open_windows_file_dialog_async() -> None:
    """Open a native Windows file dialog in a background thread via PowerShell.

    The dialog stays open until dismissed — the macro will type the path and
    press Enter to close it.
    """
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dlg = New-Object System.Windows.Forms.OpenFileDialog; "
        "$dlg.Title = 'Open'; "
        "$dlg.ShowDialog() | Out-Null"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# 1. os_picker_macro against a real Windows file dialog
# ---------------------------------------------------------------------------

@requires_headed_integration
def test_os_picker_macro_against_real_windows_open_dialog() -> None:
    """Macro must type a path and close a real Windows Open dialog."""
    from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro

    upload_file = _create_temp_upload_file()
    try:
        # Open the dialog slightly before the macro polls for it
        threading.Thread(target=_open_windows_file_dialog_async, daemon=True).start()
        time.sleep(0.5)

        result = run_os_picker_macro(
            str(upload_file),
            appear_timeout_s=5.0,
            close_timeout_s=5.0,
        )

        assert result.outcome is PickerOutcome.SUCCESS, (
            f"Macro failed: {result.outcome} — {result.detail}"
        )
        assert str(upload_file) in result.detail
    finally:
        upload_file.unlink(missing_ok=True)


@requires_headed_integration
def test_os_picker_macro_detects_picker_not_detected_when_no_dialog() -> None:
    """Macro must return PICKER_NOT_DETECTED when no dialog opens within timeout."""
    from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro

    # Do NOT open any dialog — macro should time out
    result = run_os_picker_macro(
        r"C:\nonexistent\file.txt",
        appear_timeout_s=1.0,
        close_timeout_s=1.0,
    )

    assert result.outcome is PickerOutcome.PICKER_NOT_DETECTED, (
        f"Expected PICKER_NOT_DETECTED, got {result.outcome}: {result.detail}"
    )


# ---------------------------------------------------------------------------
# 2. NativeBrowserExecutor full path against a local HTML page
# ---------------------------------------------------------------------------

_UPLOAD_HTML = """<!DOCTYPE html>
<html>
<head><title>Upload Test</title></head>
<body>
  <input id="file-input" type="file" accept="*/*">
  <div id="filename"></div>
  <script>
    document.getElementById('file-input').addEventListener('change', function(e) {
      document.getElementById('filename').textContent = e.target.files[0].name;
    });
  </script>
</body>
</html>
"""


def _write_upload_html() -> Path:
    tmp = Path(tempfile.mktemp(suffix=".html", prefix="operon_upload_page_"))
    tmp.write_text(_UPLOAD_HTML, encoding="utf-8")
    return tmp


@requires_headed_integration
@pytest.mark.asyncio
async def test_native_browser_executor_upload_file_native_headed_end_to_end() -> None:
    """Full path: headed browser → click file input → OS picker → macro → file reflected."""
    from src.executor.browser_native import NativeBrowserExecutor
    from src.models.policy import ActionType, AgentAction

    upload_file = _create_temp_upload_file()
    html_page = _write_upload_html()
    run_id = "integration-test-upload"

    executor = NativeBrowserExecutor(
        artifact_dir=".browser-artifacts/integration-test",
        headless=False,
        record_video=False,
        post_action_delay=0.5,
    )
    executor.set_current_run_id(run_id)

    try:
        # Navigate to the test page
        nav_result = await executor.execute(AgentAction(
            action_type=ActionType.NAVIGATE,
            url=html_page.as_uri(),
        ))
        assert nav_result.success, f"Navigate failed: {nav_result.detail}"

        # Capture the file input coordinates via screenshot + known layout
        # The input is at the top of the page — use selector-based click
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            selector="#file-input",
            text=str(upload_file),
        )

        result = await executor.execute(action)

        assert result.success, (
            f"upload_file_native failed: {result.failure_category} — {result.detail}"
        )

        # Verify the filename appears in the page DOM
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        session = executor._sessions.get(run_id)
        assert session is not None
        filename_text = await session.page.locator("#filename").text_content(timeout=3000)
        assert filename_text == upload_file.name, (
            f"Expected filename '{upload_file.name}' in DOM, got '{filename_text}'"
        )

    finally:
        await executor.aclose_run(run_id)
        upload_file.unlink(missing_ok=True)
        html_page.unlink(missing_ok=True)
