"""Shared runtime validation helpers."""

from __future__ import annotations

import os
import re as _re

from fastapi import HTTPException, status

_MAX_RUN_ID_LENGTH = 64
_RUN_ID_RE = _re.compile(r"^[A-Za-z0-9_\-]+$")


def validate_run_id(run_id: str) -> None:
    """Reject run IDs that would cause filesystem issues or path traversal."""
    if len(run_id) > _MAX_RUN_ID_LENGTH or not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")


def test_safe_mode_enabled() -> bool:
    return os.getenv("OPERON_TEST_SAFE_MODE", "false").lower() == "true"

