"""Opt-in Google Drive integration smoke tests for FilePorter.

These tests use real network and Drive credentials. They are skipped by default
and are intended for manual runs or a future secret-backed CI lane.

Enable with:

    OPERON_RUN_DRIVE_INTEGRATION=true
    OPERON_DRIVE_TEST_FOLDER_ID=<target folder id>

Auth is read by FilePorter using either:
    GOOGLE_SERVICE_ACCOUNT_JSON
or application default credentials.

Optional:

    OPERON_DRIVE_TEST_URL=https://example.com/
"""

from __future__ import annotations

import os

import pytest

from src.tools.file_porter import run_porter

_RUN_DRIVE_INTEGRATION = os.getenv("OPERON_RUN_DRIVE_INTEGRATION", "false").lower() == "true"
_TEST_FOLDER_ID = os.getenv("OPERON_DRIVE_TEST_FOLDER_ID", "")
_TEST_URL = os.getenv("OPERON_DRIVE_TEST_URL", "https://example.com/")
_HAS_SERVICE_ACCOUNT = bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))

pytestmark = [
    pytest.mark.skipif(
        not _RUN_DRIVE_INTEGRATION,
        reason="Set OPERON_RUN_DRIVE_INTEGRATION=true to run Drive upload smoke tests.",
    ),
]


@pytest.fixture
def require_drive_env() -> None:
    if not _TEST_FOLDER_ID:
        pytest.skip("Missing OPERON_DRIVE_TEST_FOLDER_ID for Drive smoke test.")
    if not _HAS_SERVICE_ACCOUNT:
        # ADC may still work, but this keeps failures actionable in unattended runs.
        # Users who rely on ADC can remove or loosen this gate later.
        pytest.skip("Missing GOOGLE_SERVICE_ACCOUNT_JSON; Drive smoke test requires explicit credentials.")


def test_file_porter_can_upload_to_drive(require_drive_env: None) -> None:
    """Smoke test the real download + Drive upload path."""
    result = run_porter(_TEST_URL, _TEST_FOLDER_ID)

    assert result.success, result.detail
    assert result.drive_file_id
    assert result.local_bytes > 0
    assert "file id:" in result.detail.lower()
