"""Run-directory retention sweep (store.cleanup)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from operon.store import cleanup
from operon.store.cleanup import _fmt_bytes, cleanup_old_runs

DAY = 24 * 60 * 60


def _run_dir(root: Path, name: str, *, age_days: float, with_state: bool = True, size: int = 10) -> Path:
    run = root / name
    (run / "step_1").mkdir(parents=True)
    (run / "step_1" / "after.png").write_bytes(b"x" * size)
    stamp = time.time() - age_days * DAY
    if with_state:
        state = run / "state.json"
        state.write_text("{}")
        os.utime(state, (stamp, stamp))
    os.utime(run, (stamp, stamp))
    return run


def test_missing_root_is_a_no_op(tmp_path: Path) -> None:
    assert cleanup_old_runs(tmp_path / "nope", keep_days=0, delete=True) == ([], 0)


def test_dry_run_lists_old_runs_and_keeps_them(tmp_path: Path) -> None:
    old = _run_dir(tmp_path, "old", age_days=10, size=100)
    _run_dir(tmp_path, "new", age_days=0)

    dirs, freed = cleanup_old_runs(tmp_path, keep_days=7)

    assert dirs == [old]
    assert freed == 100 + len("{}")
    assert old.exists()


def test_delete_removes_only_runs_older_than_the_cutoff(tmp_path: Path) -> None:
    old = _run_dir(tmp_path, "old", age_days=10)
    recent = _run_dir(tmp_path, "recent", age_days=3)

    dirs, _ = cleanup_old_runs(tmp_path, keep_days=7, delete=True)

    assert dirs == [old]
    assert not old.exists()
    assert recent.exists()


def test_state_json_mtime_wins_over_the_directory_mtime(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "run", age_days=10)
    # A fresh state.json (the run was resumed) keeps an old-looking directory.
    os.utime(run / "state.json", None)

    assert cleanup_old_runs(tmp_path, keep_days=7)[0] == []


def test_directory_mtime_is_used_without_state_json(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "run", age_days=10, with_state=False)

    assert cleanup_old_runs(tmp_path, keep_days=7)[0] == [run]


def test_files_in_the_root_are_ignored(tmp_path: Path) -> None:
    stray = tmp_path / "notes.txt"
    stray.write_text("keep me")
    old = time.time() - 30 * DAY
    os.utime(stray, (old, old))

    assert cleanup_old_runs(tmp_path, keep_days=0, delete=True)[0] == []
    assert stray.exists()


def test_fmt_bytes_picks_a_unit() -> None:
    assert _fmt_bytes(512) == "512.0 B"
    assert _fmt_bytes(2048) == "2.0 KB"
    assert _fmt_bytes(3 * 1024 * 1024) == "3.0 MB"


def test_cli_dry_run_reports_without_deleting(tmp_path: Path, monkeypatch, capsys) -> None:
    old = _run_dir(tmp_path, "old", age_days=10)
    monkeypatch.setattr("sys.argv", ["cleanup", "--runs-dir", str(tmp_path), "--days", "7"])

    cleanup.main()

    out = capsys.readouterr().out
    assert "Would delete 1 run directory" in out
    assert "--delete" in out
    assert old.exists()


def test_cli_delete_removes(tmp_path: Path, monkeypatch, capsys) -> None:
    old = _run_dir(tmp_path, "old", age_days=10)
    monkeypatch.setattr("sys.argv", ["cleanup", "--runs-dir", str(tmp_path), "--delete"])

    cleanup.main()

    assert "Deleted 1 run directory" in capsys.readouterr().out
    assert not old.exists()


def test_cli_with_nothing_to_clean(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["cleanup", "--runs-dir", str(tmp_path)])

    cleanup.main()

    assert capsys.readouterr().out.strip() == "Nothing to clean up."


def test_fmt_bytes_keeps_the_fraction() -> None:
    assert _fmt_bytes(1536) == "1.5 KB"


def test_cli_defaults_to_the_real_runs_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    seen: dict = {}
    monkeypatch.setattr(cleanup, "cleanup_old_runs", lambda **kw: seen.update(kw) or ([], 0))
    monkeypatch.setattr("sys.argv", ["cleanup"])

    cleanup.main()

    assert Path(seen["root_dir"]) == tmp_path
