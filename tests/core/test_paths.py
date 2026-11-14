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



def test_blank_override_falls_back_to_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("OPERON_RUNS_ROOT", "  ")

    assert paths.runs_dir() == tmp_path / "runtime" / "runs"


def test_source_checkout_is_this_repo() -> None:
    assert paths.source_checkout() == paths.project_root()


def _fake_install(monkeypatch, tmp_path: Path, package_parent: Path) -> None:
    fake = package_parent / "operon" / "core" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "__file__", str(fake))
    for name in ("OPERON_RUNTIME_ROOT", "OPERON_RUNS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPERON_CONFIG_DIR", str(tmp_path / "home" / ".operon"))


def test_pip_install_writes_runs_to_the_user_folder(monkeypatch, tmp_path: Path) -> None:
    # A venv inside someone else's project: their pyproject.toml must not pull runs there.
    project = tmp_path / "their-project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'theirs'\n", encoding="utf-8")
    _fake_install(monkeypatch, tmp_path, project / ".venv" / "lib" / "python3.14" / "site-packages")

    assert paths.source_checkout() is None
    assert paths.runtime_dir() == tmp_path / "home" / ".operon"
    assert paths.runs_dir() == tmp_path / "home" / ".operon" / "runs"


def test_pip_install_still_honours_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    _fake_install(monkeypatch, tmp_path, tmp_path / "site-packages")
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path / "custom"))

    assert paths.runs_dir() == tmp_path / "custom" / "runs"


def test_checkout_layout_keeps_the_var_folder(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "Operon"
    (repo / "src").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'operon-agent'\n", encoding="utf-8")
    _fake_install(monkeypatch, tmp_path, repo / "src")

    assert paths.source_checkout() == repo
    assert paths.runtime_dir() == repo / ".var"
