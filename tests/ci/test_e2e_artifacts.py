"""Offline tests for the opt-in e2e evidence copy (KAN-213)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests import e2e_artifacts
from tests.e2e_artifacts import ENV_VAR, safe_name, save_run_artifacts

TESTS = Path(__file__).resolve().parent.parent


def _make_run(src: Path) -> None:
    (src / "run" / "steps").mkdir(parents=True)
    (src / "run" / "steps" / "step_1_before.png").write_bytes(b"png")
    (src / "run" / "steps" / "step_1_after.JPG").write_bytes(b"jpg")
    (src / "run" / "run.jsonl").write_text("{}\n")
    (src / "run" / "state.json").write_text("{}")
    (src / "run" / "settings.json").write_text('{"key": "x"}')
    (src / "run" / "host.json").write_text("{}")
    (src / "run" / "memory").mkdir()
    (src / "run" / "memory" / "m.png").write_bytes(b"png")
    (src / "run" / "model_io.txt").write_text("secret")


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_env_copies_nothing(tmp_path, monkeypatch, value):
    """Regression for KAN-213: evidence is opt-in; without the env var nothing is touched."""
    src = tmp_path / "src"
    _make_run(src)
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)
    before = _files(tmp_path)
    assert save_run_artifacts(src, "a::b") is None
    assert _files(tmp_path) == before
    assert {p.name for p in tmp_path.iterdir()} == {"src"}


def test_copies_only_allowlisted_files(tmp_path):
    """Regression for KAN-213: only screenshots, run.jsonl and state.json leave the run dir."""
    src = tmp_path / "src"
    _make_run(src)
    out = tmp_path / "out"
    dest = save_run_artifacts(src, "tests/x.py::test_a[1]", str(out))
    assert dest == out / safe_name("tests/x.py::test_a[1]")
    assert _files(dest) == {
        "run/steps/step_1_before.png",
        "run/steps/step_1_after.JPG",
        "run/run.jsonl",
        "run/state.json",
        "nodeid.txt",
    }
    assert "tests/x.py::test_a[1]" in (dest / "nodeid.txt").read_text()


def test_env_var_is_used(tmp_path, monkeypatch):
    """Regression for KAN-213: the destination comes from OPERON_E2E_ARTIFACTS_DIR."""
    src = tmp_path / "src"
    _make_run(src)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env_out"))
    dest = save_run_artifacts(src, "t::x")
    assert dest is not None and (dest / "run" / "run.jsonl").is_file()


def test_safe_name_strips_and_bounds():
    """Regression for KAN-213: node ids become safe, bounded, distinct folder names."""
    name = safe_name("tests/browser_e2e/test_a.py::test_x[case/1:b]")
    assert not set(name) & set("/:[]")
    long_a = "tests/x.py::test_" + "a" * 300 + "1"
    long_b = "tests/x.py::test_" + "a" * 300 + "2"
    assert len(safe_name(long_a)) <= 130
    assert safe_name(long_a) != safe_name(long_b)


def test_oversize_file_skipped_and_cap_stops(tmp_path, monkeypatch):
    """Regression for KAN-213: per-file and per-test size caps bound the evidence."""
    monkeypatch.setattr(e2e_artifacts, "MAX_FILE_BYTES", 10)
    monkeypatch.setattr(e2e_artifacts, "MAX_TEST_BYTES", 15)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a_big.png").write_bytes(b"x" * 11)
    (src / "b.png").write_bytes(b"x" * 8)
    (src / "c.png").write_bytes(b"x" * 8)
    dest = save_run_artifacts(src, "t::cap", str(tmp_path / "out"))
    assert dest is not None
    assert _files(dest) == {"b.png", "nodeid.txt"}
    note = (dest / "nodeid.txt").read_text()
    assert "a_big.png" in note and "c.png" in note


def test_symlink_outside_src_not_copied(tmp_path):
    """Regression for KAN-213: a symlink out of the run dir is never followed."""
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    src = tmp_path / "src"
    src.mkdir()
    try:
        (src / "link.png").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")
    dest = save_run_artifacts(src, "t::link", str(tmp_path / "out"))
    assert dest is not None
    assert _files(dest) == {"nodeid.txt"}


def test_unwritable_destination_warns_not_raises(tmp_path):
    """Regression for KAN-213: a failing copy must never fail the test."""
    src = tmp_path / "src"
    _make_run(src)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert save_run_artifacts(src, "t::w", str(blocker / "sub")) is None
    assert any("could not save e2e artifacts" in str(w.message) for w in caught)


def test_conftests_wired_and_windows_untouched():
    """Regression for KAN-213: browser and desktop e2e save evidence; Windows Sandbox never does."""
    for folder in ("browser_e2e", "desktop_e2e"):
        assert "save_run_artifacts" in (TESTS / folder / "conftest.py").read_text()
    for path in (TESTS / "desktop_e2e_windows").rglob("*.py"):
        assert ENV_VAR not in path.read_text(encoding="utf-8")
