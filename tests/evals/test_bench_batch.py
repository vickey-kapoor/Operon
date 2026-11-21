"""The benchmark batch reader (evals/bench_batch.py, KAN-105, KAN-184: public benchmarks only)."""

from __future__ import annotations

import io
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


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_a_grounding_cell_reads_like_a_bench_comment() -> None:
    cell = parse_line("grounding zoom=always per_set=50 pr=242 cap=1.5", 1)
    assert (cell.kind, cell.zoom, cell.per_set, cell.pr, cell.cap) == ("grounding", "always", 50, "242", 1.5)
    assert cell.label == "Gemini · Grounding · zoom always · 50 per set · PR #242"


def test_grounding_defaults() -> None:
    cell = parse_line("/bench grounding gemini thin", 1)
    assert (cell.zoom, cell.per_set, cell.ref, cell.cap) == ("off", 100, "", 3.0)
    assert cell.label == "Gemini · Grounding"
    assert parse_line("grounding zoom", 1).zoom == "dense"
    assert parse_line("grounding ref=main", 1).label == "Gemini · Grounding · main"


def test_an_osworld_cell_and_its_defaults() -> None:
    cell = parse_line("osworld", 1)
    assert (cell.limit, cell.only, cell.steps, cell.cap) == (25, "", 50, 3.0)
    assert cell.label == "Gemini · OSWorld · 25 tasks"
    assert parse_line("osworld limit=5 steps=30 cap=1", 1).label == "Gemini · OSWorld · 5 tasks · 30 steps"
    assert parse_line("osworld limit=0", 1).limit == 25  # 0 is all 25
    only = parse_line("osworld only=3E5a-9 limit=5", 1)
    assert only.only == "3e5a-9" and only.label == "Gemini · OSWorld · task 3e5a-9"


@pytest.mark.parametrize("word", ["browser", "desktop", "hybrid", "core", "smoke", "all", "safety", "apps", "hard", "long",
                                  "dynamic", "tools", "settle", "note", "planner", "expect", "x3", "season=2",
                                  "model=claude-sonnet-5-5"])
def test_own_suite_words_are_refused(word) -> None:
    for line in (f"{word} thin", f"grounding {word}", f"osworld {word}"):
        with pytest.raises(BatchError, match=r"own suites no longer run on paid runs \(KAN-184\); use osworld or grounding cells"):
            parse_line(line, 1)


@pytest.mark.parametrize("word", ["full", "same-critic"])
def test_the_removed_full_harness_words_are_errors(word) -> None:
    with pytest.raises(BatchError, match=r"Full harness was removed \(KAN-183\)"):
        parse_line(f"grounding {word}", 1)


def test_blank_and_comment_lines_are_skipped_and_semicolons_split_cells() -> None:
    cells = parse_batch("# today's queue\n\ngrounding; osworld limit=5  # KAN-104\n")
    assert [c.label for c in cells] == ["Gemini · Grounding", "Gemini · OSWorld · 5 tasks"]


def test_ref_keeps_its_case() -> None:
    assert parse_line("grounding ref=claude/Feature-X", 1).ref == "claude/Feature-X"


@pytest.mark.parametrize(
    "line, message",
    [
        ("thin cap=2", "start with the kind"),
        ("grounding osworld", "one kind"),
        ("grounding per_set=0", "per_set="),
        ("grounding per_set=501", "per_set="),
        ("grounding per_set=x", "per_set="),
        ("grounding limit=5", "unknown word"),
        ("osworld zoom", "unknown word"),
        ("osworld per_set=5", "unknown word"),
        ("osworld limit=26", "limit="),
        ("osworld limit=-1", "limit="),
        ("osworld only=xyz", "only="),
        ("osworld only=", "only="),
        ("osworld steps=0", "steps="),
        ("osworld steps=201", "steps="),
        ("osworld pr=5", "default branch on its own VM"),
        ("osworld ref=main", "default branch on its own VM"),
        ("grounding pr=12a", "pr="),
        ("grounding ref=-rf", "ref="),
        ("grounding ref=a..b", "ref="),
        ("grounding cap=0", "cap="),
        ("grounding cap=lots", "cap="),
        ("grounding pr=1 ref=main", "not both"),
        ("grounding turbo", "unknown word"),
    ],
)
def test_bad_lines_are_refused_with_their_line(line: str, message: str) -> None:
    with pytest.raises(BatchError, match=message) as err:
        parse_batch(f"grounding\n{line}")
    assert "line 2" in str(err.value) or "twice" in str(err.value)


def test_a_batch_needs_cells_and_no_repeats() -> None:
    with pytest.raises(BatchError, match="no cells"):
        parse_batch("# nothing queued\n")
    with pytest.raises(BatchError, match="twice"):
        parse_batch("grounding pr=5\ngrounding pr=5")
    with pytest.raises(BatchError, match="twice"):
        parse_batch("osworld limit=5\nosworld limit=5 cap=1")  # the cap is not part of the cell's identity
    # Cells that differ in any field are not repeats.
    parse_batch("grounding pr=5\ngrounding pr=6\ngrounding zoom pr=5\ngrounding per_set=10 pr=5", 0)
    parse_batch("osworld limit=5\nosworld limit=6\nosworld only=3e\nosworld steps=30", 0)


def test_only_overrides_limit_so_it_is_not_a_different_cell() -> None:
    with pytest.raises(BatchError, match="twice"):
        parse_batch("osworld only=3e5a\nosworld only=3e5a limit=5")


def test_ref_keys_are_case_insensitive_but_values_keep_their_case() -> None:
    assert parse_line("grounding REF=Main", 1).ref == "Main"
    assert parse_line("grounding Ref=Claude/X cap=1", 1).ref == "Claude/X"


def test_azure_is_refused_like_the_bench_comment_parser_does() -> None:
    with pytest.raises(BatchError, match=r"KAN-184"):
        parse_line("grounding azure", 1)


def test_osworld_checks_out_the_default_branch() -> None:
    steps = _workflow("osworld-bench.yml")["jobs"]["bench"]["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"


def test_a_batch_has_a_most_cells() -> None:
    text = "\n".join(f"grounding per_set={n}" for n in range(1, bench_batch.MAX_CELLS + 2))
    with pytest.raises(BatchError, match="the most is"):
        parse_batch(text, 0)


def test_the_caps_must_fit_in_the_batch_cap() -> None:
    cells = "grounding\nosworld limit=5\ngrounding zoom"  # 3 x $3 default
    assert bench_batch.total_cap(parse_batch(cells, 10)) == 9
    with pytest.raises(BatchError, match=r"\$9.00, above the batch cap of \$8.00"):
        parse_batch(cells, 8)
    assert len(parse_batch(cells + "\nosworld cap=50", 0)) == 4  # 0: no batch cap


def test_pr_cells_are_pinned_to_their_head_once_per_pr() -> None:
    cells = parse_batch("grounding pr=242\ngrounding zoom pr=242\ngrounding pr=7\ngrounding ref=main\nosworld", 0)
    calls = []
    resolve_refs(cells, lambda pr: calls.append(pr) or f"sha{pr}")
    assert calls == ["242", "7"]
    assert [c.ref for c in cells] == ["sha242", "sha242", "sha7", "main", ""]


def test_matrices_split_grounding_from_osworld_and_give_unique_artifact_names() -> None:
    out = matrices(parse_batch("osworld limit=5 only=3e5a steps=30 cap=1.5\ngrounding zoom=always per_set=20 ref=abc\nosworld limit=0"))
    assert out["has_grounding"] and out["has_osworld"] and out["total_cap"] == "7.5"
    assert out["grounding"]["include"] == [
        {"label": "Gemini · Grounding · zoom always · 20 per set · abc", "ref": "abc", "zoom": "always", "per_set": "20", "cap": "3", "suffix": "-cell2"}
    ]
    assert out["osworld"]["include"] == [
        {"label": "Gemini · OSWorld · task 3e5a · 30 steps", "limit": "5", "only": "3e5a", "steps": "30", "cap": "1.5", "suffix": "-cell1"},
        {"label": "Gemini · OSWorld · 25 tasks", "limit": "25", "only": "", "steps": "50", "cap": "3", "suffix": "-cell3"},
    ]
    only_osworld = matrices(parse_batch("osworld"))
    assert only_osworld["has_grounding"] is False and only_osworld["grounding"] == {"include": []}


def test_main_writes_the_workflow_outputs(tmp_path, monkeypatch) -> None:
    out, summary = tmp_path / "out", tmp_path / "summary.md"
    monkeypatch.setattr("sys.stdin", io.StringIO("grounding zoom\nosworld limit=5"))
    code = bench_batch.main(["--github-output", str(out), "--summary", str(summary), "--max-total-usd", "6"])
    assert code == 0
    values = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    assert json.loads(values["grounding"])["include"][0]["zoom"] == "dense"
    assert json.loads(values["osworld"])["include"][0]["limit"] == "5"
    assert values["has_grounding"] == "true" and values["has_osworld"] == "true" and values["total_cap"] == "6"
    assert set(values) == {"grounding", "osworld", "has_grounding", "has_osworld", "total_cap"}
    assert "Gemini · OSWorld · 5 tasks" in summary.read_text(encoding="utf-8")


def test_main_refuses_a_bad_batch(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("grounding\nosworld"))
    assert bench_batch.main(["--max-total-usd", "4", "--github-output", "", "--summary", ""]) == 1
    assert "::error::" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO("browser thin"))
    assert bench_batch.main(["--github-output", "", "--summary", ""]) == 1
    assert "KAN-184" in capsys.readouterr().out


def test_the_batch_workflow_passes_every_cell_field_to_the_called_workflows() -> None:
    batch = _workflow("benchmark.yml")
    jobs = batch["jobs"]
    assert jobs["batch-osworld"]["strategy"]["max-parallel"] == 1
    assert jobs["batch-osworld"]["permissions"] == {"contents": "write", "actions": "read", "id-token": "write"}
    for job, called in (("batch-grounding", "grounding.yml"), ("batch-osworld", "osworld-bench.yml")):
        inputs = _workflow(called)[True]["workflow_call"]["inputs"]
        assert set(jobs[job]["with"]) <= set(inputs), f"{job} passes an input {called} does not take"
        # Only the Gemini key reaches a paid run.
        assert set(jobs[job]["secrets"]) == {"GEMINI_API_KEY"}
        # Each cell's artifact gets its own name; a manual run (empty suffix) keeps today's name.
        assert jobs[job]["with"]["artifact_suffix"] == "${{ matrix.suffix }}"
        assert inputs["artifact_suffix"]["default"] == ""
    # Every matrix field the workflow reads is one the parser writes.
    fields = {"grounding": ("label", "ref", "zoom", "per_set", "cap", "suffix"), "osworld": ("label", "limit", "only", "steps", "cap", "suffix")}
    for job, kind in (("batch-grounding", "grounding"), ("batch-osworld", "osworld")):
        text = json.dumps(jobs[job])
        for used in {part.split("}}")[0].strip() for part in text.split("matrix.")[1:]}:
            assert used in fields[kind], f"{job} reads matrix.{used}"
        assert set(matrices(parse_batch(kind))[kind]["include"][0]) == set(fields[kind])
    # The dashboard import listens for the Benchmark workflow (manual and /bench runs) by its name.
    dashboard = _workflow("dashboard-data.yml")
    assert batch["name"] in dashboard[True]["workflow_run"]["workflows"]


@pytest.mark.parametrize("name, artifact", [("grounding.yml", "grounding-computer_use"), ("osworld-bench.yml", "osworld-report")])
def test_called_workflows_suffix_their_artifact_names(name, artifact) -> None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert f"name: {artifact}${{{{ inputs.artifact_suffix }}}}\n" in text
    # The dashboard import downloads grounding artifacts by pattern, so a suffix still matches.
    if artifact.startswith("grounding"):
        assert "--pattern 'grounding-*'" in (WORKFLOWS / "dashboard-data.yml").read_text(encoding="utf-8")


def test_run_videos_are_kept_for_batch_osworld_cells_only() -> None:
    jobs = _workflow("benchmark.yml")["jobs"]
    assert jobs["publish-videos"]["needs"] == ["batch-osworld"]
    assert jobs["batch-osworld"]["with"]["publish_videos"] == "false"
    # A caller must grant what the called workflow's jobs ask for.
    assert jobs["batch-osworld"]["permissions"] == {"contents": "write", "actions": "read", "id-token": "write"}
    osworld = _workflow("osworld-bench.yml")
    assert osworld["jobs"]["bench"]["permissions"] == {"contents": "read"}
    assert "publish_videos != 'false'" in osworld["jobs"]["publish-videos"]["if"]
    text = (WORKFLOWS / "osworld-bench.yml").read_text(encoding="utf-8")
    assert "name: run-videos-osworld${{ inputs.artifact_suffix }}\n" in text  # still matches 'run-videos-*'
    assert "--pattern 'run-videos-*'" in (WORKFLOWS / "benchmark.yml").read_text(encoding="utf-8")


def test_the_osworld_workflow_can_be_called_and_keeps_its_vm_lock() -> None:
    wf = _workflow("osworld-bench.yml")
    assert set(wf[True]) == {"workflow_dispatch", "workflow_call"}
    assert set(wf[True]["workflow_call"]["inputs"]) == {"limit", "only", "max_cost_usd", "max_steps", "artifact_suffix", "publish_videos"}
    assert wf["concurrency"]["group"] == "azure-osworld-vm"


@pytest.mark.parametrize("name", ["live-bench.yml", "live-desktop-bench.yml"])
def test_the_own_suite_workflows_are_not_startable_by_hand(name) -> None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert set(yaml.safe_load(text)[True]) == {"workflow_call"}
    assert "No longer started by any paid entry point (KAN-184)" in text
    assert "      OPERON_HARNESS_PROFILE: thin\n" in text


def test_benchmark_workflow_jobs_only_wait_on_jobs_that_exist() -> None:
    wf = _workflow("benchmark.yml")
    jobs = wf["jobs"]
    for name, job in jobs.items():
        needs = job.get("needs", [])
        for need in [needs] if isinstance(needs, str) else needs:
            assert need in jobs, f"{name} waits on {need}, which is not a job"
    # Every called engine input exists, for the single /bench comment job too.
    inputs = _workflow("grounding.yml")[True]["workflow_call"]["inputs"]
    assert set(jobs["grounding"]["with"]) <= set(inputs)
    # Paid runs are public benchmarks only: no own-suite job is left.
    assert set(jobs) == {"parse", "plan", "batch-grounding", "batch-osworld", "grounding", "publish-videos"}
    assert set(wf[True]) == {"issue_comment", "workflow_dispatch"}  # no schedule
    assert wf[True]["workflow_dispatch"]["inputs"]["cells"]["required"] is True


def test_the_bench_comment_parser_refuses_own_suites_and_a_bare_bench() -> None:
    text = (WORKFLOWS / "benchmark.yml").read_text(encoding="utf-8")
    assert "KAN-184" in text and "/bench grounding" in text
    assert "OSWorld runs the default branch on its own VM; start it as a batch cell" in text
    assert "live-bench.yml" not in text and "live-desktop-bench.yml" not in text and "start-vm" not in text


def test_the_old_benchmark_entry_points_are_gone() -> None:
    for old in ("bench-batch.yml", "bench-command.yml", "nightly.yml"):
        assert not (WORKFLOWS / old).exists(), f"{old} was merged into benchmark.yml"


def test_batch_jobs_run_past_the_skipped_comment_parser() -> None:
    # On a manual dispatch `parse` is skipped, so a job without a status function is skipped too.
    jobs = _workflow("benchmark.yml")["jobs"]
    for name in ("batch-grounding", "batch-osworld"):
        condition = jobs[name]["if"]
        assert "!cancelled()" in condition, name
        assert "needs.plan.result == 'success'" in condition, name
    assert "!cancelled() && needs.parse.result != 'failure'" in jobs["plan"]["if"]
