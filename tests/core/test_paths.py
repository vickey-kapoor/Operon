"""Tests for shared filesystem path helpers."""

from __future__ import annotations

from pathlib import Path

from operon.core import paths


def test_default_runtime_paths_use_var_tree(monkeypatch) -> None:
    monkeypatch.delenv("OPERON_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("OPERON_RUNS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_BROWSER_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_DESKTOP_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_TEST_ARTIFACTS_ROOT", raising=False)

    root = paths.project_root()

    assert paths.runtime_dir() == root / ".var"
    assert paths.runs_dir() == root / ".var" / "runs"
    assert paths.browser_artifacts_dir() == root / ".var" / "browser-artifacts"
    assert paths.desktop_artifacts_dir() == root / ".var" / "desktop-artifacts"
    assert paths.test_artifacts_dir() == root / ".var" / "test-artifacts"


def test_runtime_root_override_feeds_default_children(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.delenv("OPERON_RUNS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_BROWSER_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_DESKTOP_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("OPERON_TEST_ARTIFACTS_ROOT", raising=False)

    assert paths.runtime_dir() == tmp_path / "runtime"
    assert paths.runs_dir() == tmp_path / "runtime" / "runs"
    assert paths.browser_artifacts_dir() == tmp_path / "runtime" / "browser-artifacts"
    assert paths.desktop_artifacts_dir() == tmp_path / "runtime" / "desktop-artifacts"
    assert paths.test_artifacts_dir() == tmp_path / "runtime" / "test-artifacts"


def test_specific_artifact_root_overrides_win_over_runtime_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path / "custom-runs"))
    monkeypatch.setenv("OPERON_BROWSER_ARTIFACTS_ROOT", str(tmp_path / "custom-browser"))
    monkeypatch.setenv("OPERON_DESKTOP_ARTIFACTS_ROOT", str(tmp_path / "custom-desktop"))
    monkeypatch.setenv("OPERON_TEST_ARTIFACTS_ROOT", str(tmp_path / "custom-tests"))

    assert paths.runs_dir() == tmp_path / "custom-runs"
    assert paths.browser_artifacts_dir() == tmp_path / "custom-browser"
    assert paths.desktop_artifacts_dir() == tmp_path / "custom-desktop"
    assert paths.test_artifacts_dir() == tmp_path / "custom-tests"

