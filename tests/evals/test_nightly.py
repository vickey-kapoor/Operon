"""The nightly paid run's planner (evals/nightly.py, KAN-165)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from evals import nightly
from evals.nightly import plan_night, read_queue

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TODAY = date(2026, 11, 15)


def _entries(text: str):
    entries, bad = read_queue(text)
    assert not bad, bad
    return entries


def test_queue_lines_need_a_date_and_a_valid_cell() -> None:
    entries, bad = read_queue(
        "# comment\n\n2026-11-14 desktop thin cap=2  # note\nbrowser thin\n2026-13-01 browser thin\n2026-11-14 turbo\n"
    )
    assert [e.cells_text for e in entries] == ["desktop thin cap=2"]
    assert len(bad) == 3


def test_cells_run_in_order_while_caps_fit_the_budget() -> None:
    entries = _entries("2026-11-14 desktop thin cap=2.5\n2026-11-14 hybrid full cap=9\n2026-11-14 browser thin cap=2\n")
    plan = plan_night(entries, set(), TODAY, month_spend=0)
    assert plan.budget == 10
    assert [e.cell.category for e in plan.tonight] == ["desktop", "browser"]  # hybrid ($9) waits, browser still fits
    assert [e.cell.category for e in plan.waiting] == ["hybrid"]
    assert plan.total_cap == 4.5


def test_the_month_cap_shrinks_the_budget_and_stops_the_run() -> None:
    entries = _entries("2026-11-14 desktop thin cap=2\n")
    assert plan_night(entries, set(), TODAY, month_spend=99).tonight == []
    assert plan_night(entries, set(), TODAY, month_spend=98).tonight[0].cell.cap == 2
    assert plan_night(entries, set(), TODAY, month_spend=120).budget == 0


def test_a_line_runs_once_and_old_lines_are_skipped() -> None:
    entries = _entries("2026-11-14 desktop thin\n2026-11-01 browser thin\n2026-11-14 desktop thin\n")
    plan = plan_night(entries, set(), TODAY, 0)
    assert len(plan.tonight) == 1 and len(plan.skipped) == 2
    assert plan_night(entries, {entries[0].key}, TODAY, 0).tonight == []


def test_cells_the_batch_would_refuse_wait_for_another_night() -> None:
    entries = _entries("2026-11-14 browser thin cap=1\n2026-11-13 browser thin cap=1  # again\n2026-11-14 desktop thin cap=1\n")
    plan = plan_night(entries, set(), TODAY, 0)
    assert [e.cell.category for e in plan.tonight] == ["browser", "desktop"]
    assert len(plan.waiting) == 1


def test_future_dated_lines_wait() -> None:
    plan = plan_night(_entries("2026-11-20 browser thin\n"), set(), TODAY, 0)
    assert plan.tonight == [] and len(plan.waiting) == 1


def test_main_fails_closed_without_the_database(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    queue = tmp_path / "q.txt"
    queue.write_text("2026-11-14 browser thin\n", encoding="utf-8")
    assert nightly.main(["plan", "--queue", str(queue), "--github-output", "", "--summary", ""]) == 1
    assert "nothing runs" in capsys.readouterr().out


def test_main_writes_outputs_and_marks_lines_as_run(tmp_path, monkeypatch) -> None:
    queue, done, out = tmp_path / "q.txt", tmp_path / "done.txt", tmp_path / "out"
    today = nightly.datetime.now(nightly.CHICAGO).date().isoformat()
    queue.write_text(f"{today} browser thin cap=2\n", encoding="utf-8")
    args = ["plan", "--queue", str(queue), "--done", str(done), "--month-spend", "10", "--github-output", str(out), "--summary", ""]
    assert nightly.main(args + ["--dry-run"]) == 0
    assert not done.exists()
    assert nightly.main(args) == 0
    text = out.read_text(encoding="utf-8")
    assert "run=true" in text and "budget=10" in text and "browser thin cap=2" in text
    assert len(done.read_text(encoding="utf-8").split()) == 1
    out.write_text("")
    nightly.main(args)  # second night: already run
    assert "run=false" in out.read_text(encoding="utf-8")


def test_the_real_queue_parses() -> None:
    _, bad = read_queue(nightly.QUEUE.read_text(encoding="utf-8"))
    assert not bad


def test_workflow_calls_the_batch_and_is_imported_by_the_dashboard() -> None:
    wf = yaml.safe_load((WORKFLOWS / "nightly.yml").read_text(encoding="utf-8"))
    assert wf[True]["schedule"][0]["cron"] == "23 8 * * *"
    batch_inputs = yaml.safe_load((WORKFLOWS / "bench-batch.yml").read_text(encoding="utf-8"))[True]["workflow_call"]["inputs"]
    assert set(wf["jobs"]["batch"]["with"]) <= set(batch_inputs)
    dash = yaml.safe_load((WORKFLOWS / "dashboard-data.yml").read_text(encoding="utf-8"))
    assert wf["name"] in dash[True]["workflow_run"]["workflows"]
