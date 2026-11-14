"""The benchmark batch reader (evals/bench_batch.py, KAN-105)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evals import bench_batch
from evals.bench_batch import (
    BatchError,
    matrices,
    parse_batch,
    parse_line,
    resolve_refs,
)

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def test_a_cell_reads_like_a_bench_comment() -> None:
    cell = parse_line("browser thin all x3 season=2 pr=242 cap=1.5", 1)
    assert (cell.category, cell.profile, cell.suite, cell.repeat, cell.season, cell.pr, cell.cap) == (
        "browser", "thin", "full", 3, "2", "242", 1.5
    )
    assert cell.extra == "--repeat 3 --season 2"
    assert cell.label == "Gemini · Browser · Thin · full · Season 2 · x3 · PR #242"


def test_defaults_match_a_plain_bench_comment() -> None:
    cell = parse_line("desktop azure gemini", 1)
    assert (cell.profile, cell.suite, cell.repeat, cell.critic, cell.ref, cell.cap) == ("full", "core", 1, "", "", 3.0)
    assert cell.extra == ""


def test_blank_and_comment_lines_are_skipped_and_semicolons_split_cells() -> None:
    cells = parse_batch("# today's queue\n\nbrowser thin; desktop thin  # KAN-104\n")
    assert [c.label for c in cells] == ["Gemini · Browser · Thin", "Gemini · Desktop · Thin"]


def test_tools_word_turns_every_tool_on_and_is_its_own_cell() -> None:
    cells = parse_batch("desktop thin season=2\ndesktop thin season=2 tools")
    assert [c.tools for c in cells] == ["off", "all"]
    assert cells[1].label == "Gemini · Desktop · Thin · Season 2 · tools"
    assert matrices(cells)["desktop"]["include"][1]["tools"] == "all"


def test_ref_keeps_its_case() -> None:
    assert parse_line("hybrid full ref=claude/Feature-X", 1).ref == "claude/Feature-X"


@pytest.mark.parametrize(
    "line, message",
    [
        ("thin season=2", "start with the category"),
        ("browser desktop", "one category"),
        ("browser grounding", "grounding is not part of a batch"),
        ("desktop apps", "browser only"),
        ("hybrid safety", "Hybrid has no safety suite"),
        ("browser thin same-critic", "drop thin"),
        ("browser season=x", "season="),
        ("browser pr=12a", "pr="),
        ("browser ref=-rf", "ref="),
        ("browser ref=a..b", "ref="),
        ("browser cap=0", "cap="),
        ("browser cap=lots", "cap="),
        ("browser pr=1 ref=main", "not both"),
        ("browser turbo", "unknown word"),
    ],
)
def test_bad_lines_are_refused_with_their_line(line: str, message: str) -> None:
    with pytest.raises(BatchError, match=message) as err:
        parse_batch(f"browser thin\n{line}")
    assert "line 2" in str(err.value) or "twice" in str(err.value)


def test_a_batch_needs_cells_and_no_repeats() -> None:
    with pytest.raises(BatchError, match="no cells"):
        parse_batch("# nothing queued\n")
    with pytest.raises(BatchError, match="twice"):
        parse_batch("browser thin pr=5\nbrowser thin pr=5")
    parse_batch("browser thin pr=5\nbrowser thin pr=6")  # same cell on two PRs is fine


def test_the_caps_must_fit_in_the_batch_cap() -> None:
    cells = "browser thin\ndesktop thin\nhybrid thin"  # 3 x $3 default
    assert bench_batch.total_cap(parse_batch(cells, 10)) == 9
    with pytest.raises(BatchError, match=r"\$9.00, above the batch cap of \$8.00"):
        parse_batch(cells, 8)
    assert len(parse_batch(cells + "\nbrowser full cap=50", 0)) == 4  # 0: no batch cap


def test_pr_cells_are_pinned_to_their_head_once_per_pr() -> None:
    cells = parse_batch("browser thin pr=242\ndesktop thin pr=242\nhybrid thin pr=7\nbrowser full ref=main", 0)
    calls = []
    resolve_refs(cells, lambda pr: calls.append(pr) or f"sha{pr}")
    assert calls == ["242", "7"]
    assert [c.ref for c in cells] == ["sha242", "sha242", "sha7", "main"]


def test_matrices_split_browser_from_the_vm_and_give_unique_artifact_names() -> None:
    out = matrices(parse_batch("desktop thin season=2 cap=1.5\nbrowser thin x2\nhybrid full"))
    assert out["has_browser"] and out["has_desktop"] and out["total_cap"] == "7.5"
    (browser,) = out["browser"]["include"]
    assert browser == {
        "label": "Gemini · Browser · Thin · x2", "category": "browser", "profile": "thin", "suite": "core",
        "critic": "", "tools": "off", "extra": "--repeat 2", "ref": "", "cap": "1.5", "suffix": "-cell2",
    }
    desktop = out["desktop"]["include"]
    assert [(d["category"], d["cap"], d["suffix"]) for d in desktop] == [("desktop", "1.5", "-cell1"), ("hybrid", "3", "-cell3")]


def test_main_writes_the_workflow_outputs(tmp_path, monkeypatch, capsys) -> None:
    out, summary = tmp_path / "out", tmp_path / "summary.md"
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("browser thin season=2"))
    code = bench_batch.main(["--github-output", str(out), "--summary", str(summary), "--max-total-usd", "5"])
    assert code == 0
    values = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    assert json.loads(values["browser"])["include"][0]["extra"] == "--season 2"
    assert values["has_browser"] == "true" and values["has_desktop"] == "false" and values["total_cap"] == "3"
    assert "Gemini · Browser · Thin · Season 2" in summary.read_text(encoding="utf-8")


def test_main_refuses_a_bad_batch(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("browser thin\ndesktop thin"))
    assert bench_batch.main(["--max-total-usd", "4", "--github-output", "", "--summary", ""]) == 1
    assert "::error::" in capsys.readouterr().out


def test_the_batch_workflow_passes_every_cell_field_to_the_called_workflows() -> None:
    batch = yaml.safe_load((WORKFLOWS / "bench-batch.yml").read_text(encoding="utf-8"))
    jobs = batch["jobs"]
    assert jobs["desktop"]["strategy"]["max-parallel"] == 1
    assert jobs["desktop"]["with"]["manage_vm"] == "false"
    for job, called in (("browser", "live-bench.yml"), ("desktop", "live-desktop-bench.yml")):
        inputs = yaml.safe_load((WORKFLOWS / called).read_text(encoding="utf-8"))[True]["workflow_call"]["inputs"]
        assert set(jobs[job]["with"]) <= set(inputs), f"{job} passes an input {called} does not take"
        assert jobs[job]["with"]["artifact_suffix"] == "${{ matrix.suffix }}"
    # The dashboard import listens for the batch by its name.
    dashboard = yaml.safe_load((WORKFLOWS / "dashboard-data.yml").read_text(encoding="utf-8"))
    assert batch["name"] in dashboard[True]["workflow_run"]["workflows"]


@pytest.mark.parametrize("line, run_cap", [("browser x5 cap=3", "0.6"), ("browser x3 cap=2", "0.6666"), ("desktop cap=1.5", "1.5")])
def test_cap_is_the_whole_cell_limit_with_repeats(line: str, run_cap: str) -> None:
    # live_bench multiplies max_cost_usd by --repeat, so the batch passes cap / repeat.
    cell = parse_line(line, 1)
    assert cell.run_cap == run_cap
    assert float(run_cap) * cell.repeat <= cell.cap
