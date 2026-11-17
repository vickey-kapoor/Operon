"""Shared fixtures: an empty vault under a temp config dir, with OPERON_SECRETS on."""

from __future__ import annotations

import pytest


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERON_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPERON_SECRETS", "on")
    from operon.core import vault

    return vault.vault_dir()
