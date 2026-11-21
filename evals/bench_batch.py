"""Read a benchmark batch: many paid cells, one GitHub Actions run (KAN-105, KAN-184).

Paid runs are public benchmarks only (KAN-184): OSWorld and ScreenSpot grounding. Operon's own
suites no longer run on paid runs; they stay free ($0: offline evals, replay, the e2e tests).
Runs are queued during the day and run together once Vickey approves the batch. The batch
workflow (``.github/workflows/benchmark.yml``: a manual run or a ``/bench batch`` PR comment)
passes its cell list here; this module checks every line and turns it into the job matrices
the workflow runs: grounding cells side by side on hosted runners, OSWorld cells one after
another (one VM).

One cell per line (or separated by ``;``), plus where to run it and its cost cap::

    grounding zoom per_set=50 pr=242
    grounding ref=main cap=2
    osworld limit=5
    osworld only=3e5a cap=1
    osworld model=claude-sonnet-5-5 limit=25

``grounding`` words: ``zoom|zoom=dense|zoom=always`` (KAN-82: OPERON_ZOOM), ``per_set=<1..500>``
(samples per set, default 100), ``pr=<number>`` (that PR's head) or ``ref=<branch, tag or commit>``
(default: the default branch), ``cap=<USD>`` (default 3).

``osworld`` words: ``limit=<0..25>`` (the first N tasks; 0 or leaving it out runs all 25),
``only=<task id prefix>`` (one task; overrides limit), ``steps=<1..200>`` (per task, default 50),
``model=<profile>`` (``claude-sonnet-5-5`` or ``openai-computer-use`` through the model adapter, KAN-168;
default Gemini), ``cap=<USD>`` (default 3). OSWorld runs the default branch on its own VM, so ``pr=`` and ``ref=`` are refused.

``gemini`` and ``thin`` are accepted and change nothing (``full`` and ``same-critic`` are errors:
the Full harness was removed, KAN-183). A line for Operon's own suites (browser, desktop, hybrid, ...)
is refused, and so is ``model=`` on a grounding cell (grounding calls Gemini's API directly). Blank lines and ``#`` comments are skipped. The batch is refused when its cells' caps
add up to more than the batch cap, so the most it can spend is known before anything starts.

Run it as ``python -m evals.bench_batch --max-total-usd 10 < cells.txt``: it prints a
summary table, and with ``--github-output`` writes the matrices for the workflow.
Pull request numbers are resolved to their head commit with ``gh api`` (``$GH_TOKEN``);
a PR from a fork is refused, since the run has the repository's API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from evals.osworld import (
    MODEL_LABELS,  # the one allow-list of OSWorld models; the module imports no Operon code
)

KINDS = ("grounding", "osworld")
# Words that name one of Operon's own suites (or its per-suite switches): refused since KAN-184.
OWN_SUITE_WORDS = {
    "browser", "desktop", "hybrid", "core", "smoke", "all", "safety", "apps", "hard", "long", "dynamic",
    "tools", "settle", "note", "planner", "expect", "x2", "x3", "x5",
}
OWN_SUITE_MESSAGE = "Operon's own suites no longer run on paid runs (KAN-184); use osworld or grounding cells."
DEFAULT_CAP_USD = 3.0
DEFAULT_MAX_TOTAL_USD = 10.0
DEFAULT_PER_SET = 100
MAX_PER_SET = 500
OSWORLD_TASKS = 25
DEFAULT_STEPS = 50
MAX_STEPS = 200
MAX_CELLS = 20

_PR = re.compile(r"^[1-9][0-9]{0,6}$")
# Branch, tag or commit: git ref characters only, no leading dash (it would read as an option).
_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}$")
_CAP = re.compile(r"^[0-9]{1,3}(\.[0-9]{1,2})?$")
_NUMBER = re.compile(r"^[0-9]{1,4}$")
_TASK_PREFIX = re.compile(r"^[0-9a-f][0-9a-f-]{0,35}$")


class BatchError(ValueError):
    """A batch line that can't run; the message names the line."""


@dataclass
class Cell:
    kind: str  # "grounding" or "osworld"
    zoom: str = "off"  # grounding only
    per_set: int = DEFAULT_PER_SET  # grounding only
    limit: int = OSWORLD_TASKS  # osworld only; 0 is read as all
    only: str = ""  # osworld only: a task id prefix, overrides limit
    steps: int = DEFAULT_STEPS  # osworld only
    model: str = ""  # osworld only: a registry profile; empty is Gemini
    pr: str = ""
    ref: str = ""
    cap: float = DEFAULT_CAP_USD

    @property
    def label(self) -> str:
        """The cell's name the way runs are named in chat: "Gemini · OSWorld · 25 tasks"."""
        if self.kind == "osworld":
            label = f"{MODEL_LABELS.get(self.model, 'Gemini')} · OSWorld · " + (f"task {self.only}" if self.only else f"{self.limit} tasks")
            if self.steps != DEFAULT_STEPS:
                label += f" · {self.steps} steps"
            return label
        label = "Gemini · Grounding"
        if self.zoom != "off":
            label += f" · zoom {self.zoom}"
        if self.per_set != DEFAULT_PER_SET:
            label += f" · {self.per_set} per set"
        if self.pr:
            label += f" · PR #{self.pr}"
        elif self.ref:
            label += f" · {self.ref[:12]}"
        return label


def _number(word: str, name: str, low: int, high: int, where: str) -> int:
    value = word.removeprefix(f"{name}=")
    if not _NUMBER.match(value) or not low <= int(value) <= high:
        raise BatchError(f"{where}: {name}= needs a whole number from {low} to {high}.")
    return int(value)


def parse_line(line: str, number: int) -> Cell | None:
    """One batch line as a cell, or None for a blank or comment line."""
    text = line.split("#", 1)[0].strip().lower()
    if not text:
        return None
    where = f"line {number} ({text!r})"
    # gemini and thin change nothing; accepted so earlier batches still parse.
    words = [w for w in text.split() if w not in ("/bench", "gemini", "thin")]
    for word in words:
        if word in ("full", "same-critic"):
            raise BatchError(f"{where}: the Full harness was removed (KAN-183); {word!r} no longer exists. Runs are thin.")
        if word in OWN_SUITE_WORDS or word == "azure" or word.startswith("season="):
            raise BatchError(f"{where}: {OWN_SUITE_MESSAGE}")
    kinds = [w for w in words if w in KINDS]
    if not kinds:
        raise BatchError(f"{where}: start with the kind: osworld or grounding.")
    if len(kinds) > 1:
        raise BatchError(f"{where}: name one kind (osworld or grounding).")
    cell = Cell(kind=kinds[0])
    for word in words:
        if word in KINDS:
            continue
        if word.startswith("model="):
            if cell.kind != "osworld":
                raise BatchError(f"{where}: model= is for osworld cells; grounding calls Gemini's API directly.")
            cell.model = word.removeprefix("model=")
            if cell.model not in MODEL_LABELS:
                raise BatchError(f"{where}: model= must be one of {', '.join(MODEL_LABELS)}.")
            continue
        if cell.kind == "grounding" and word in ("zoom", "zoom=dense", "zoom=always"):
            cell.zoom = "always" if word == "zoom=always" else "dense"
        elif cell.kind == "grounding" and word.startswith("per_set="):
            cell.per_set = _number(word, "per_set", 1, MAX_PER_SET, where)
        elif cell.kind == "osworld" and word.startswith("limit="):
            cell.limit = _number(word, "limit", 0, OSWORLD_TASKS, where) or OSWORLD_TASKS
        elif cell.kind == "osworld" and word.startswith("only="):
            cell.only = word.removeprefix("only=")
            if not _TASK_PREFIX.match(cell.only):
                raise BatchError(f"{where}: only= needs the start of an OSWorld task id (hex digits and dashes).")
        elif cell.kind == "osworld" and word.startswith("steps="):
            cell.steps = _number(word, "steps", 1, MAX_STEPS, where)
        elif word.startswith("pr="):
            cell.pr = word.removeprefix("pr=")
            if not _PR.match(cell.pr):
                raise BatchError(f"{where}: pr= needs a pull request number such as 242.")
        elif word.startswith("ref="):
            raw = next(w for w in line.split("#", 1)[0].split() if w.lower().startswith("ref="))
            cell.ref = raw[len("ref="):]  # keep the ref's own case
            if not _REF.match(cell.ref) or ".." in cell.ref:
                raise BatchError(f"{where}: ref= needs a branch, tag or commit.")
        elif word.startswith("cap="):
            value = word.removeprefix("cap=").lstrip("$")
            if not _CAP.match(value) or float(value) <= 0:
                raise BatchError(f"{where}: cap= needs a cost in USD above 0, such as 1.5.")
            cell.cap = float(value)
        else:
            raise BatchError(f"{where}: unknown word {word!r} for a {cell.kind} cell.")
    if cell.pr and cell.ref:
        raise BatchError(f"{where}: give pr= or ref=, not both.")
    if cell.kind == "osworld" and (cell.pr or cell.ref):
        raise BatchError(f"{where}: OSWorld runs the default branch on its own VM; pr= and ref= are for grounding cells.")
    return cell


def parse_batch(text: str, max_total_usd: float = DEFAULT_MAX_TOTAL_USD) -> list[Cell]:
    """Every cell of a batch, checked as a whole: at least one, no repeats, caps within the batch cap."""
    # A ";" also ends a cell: a workflow_dispatch input is one line in the Actions form.
    lines = text.replace(";", "\n").splitlines()
    cells = [cell for n, line in enumerate(lines, 1) if (cell := parse_line(line, n))]
    if not cells:
        raise BatchError("The batch has no cells. Put one per line, e.g. 'grounding zoom pr=242' or 'osworld limit=5'.")
    if len(cells) > MAX_CELLS:
        raise BatchError(f"The batch has {len(cells)} cells; the most is {MAX_CELLS}.")
    seen: set[tuple] = set()
    for cell in cells:
        # only= overrides limit, so limit is not part of the cell's identity then.
        key = (cell.kind, cell.zoom, cell.per_set, cell.limit if not cell.only else 0, cell.only, cell.steps, cell.model, cell.pr, cell.ref)
        if key in seen:
            raise BatchError(f"{cell.label} is in the batch twice.")
        seen.add(key)
    total = total_cap(cells)
    if max_total_usd > 0 and total > max_total_usd + 1e-9:
        raise BatchError(
            f"The cells' cost caps add up to ${total:.2f}, above the batch cap of ${max_total_usd:.2f}. "
            "Lower some cap= values, drop cells, or raise the batch cap."
        )
    return cells


def total_cap(cells: list[Cell]) -> float:
    return round(sum(cell.cap for cell in cells), 2)


def gh_pr_head(pr: str, repository: str) -> str:
    """The head commit of a pull request in this repository (``gh api``); forks are refused."""
    out = subprocess.run(
        ["gh", "api", f"repos/{repository}/pulls/{pr}", "--jq", "[.head.repo.full_name, .head.sha] | @tsv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_repo, _, sha = out.partition("\t")
    if head_repo != repository:
        raise BatchError(f"PR #{pr} comes from {head_repo or 'a deleted fork'}; refusing, the run has this repository's API key.")
    return sha


def resolve_refs(cells: list[Cell], head_of: Callable[[str], str]) -> None:
    """Pin every pr= cell to its PR's head commit, so the whole batch measures what was approved."""
    heads: dict[str, str] = {}
    for cell in cells:
        if cell.pr:
            if cell.pr not in heads:
                heads[cell.pr] = head_of(cell.pr)
            cell.ref = heads[cell.pr]


def matrices(cells: list[Cell]) -> dict:
    """The workflow's job matrices; each cell gets its own artifact suffix so names never clash."""
    grounding: list[dict] = []
    osworld: list[dict] = []
    for index, cell in enumerate(cells, 1):
        cap, suffix = f"{cell.cap:g}", f"-cell{index}"
        if cell.kind == "grounding":
            grounding.append(
                {"label": cell.label, "ref": cell.ref, "zoom": cell.zoom, "per_set": str(cell.per_set), "cap": cap, "suffix": suffix}
            )
        else:
            osworld.append(
                {"label": cell.label, "limit": str(cell.limit), "only": cell.only, "steps": str(cell.steps), "model": cell.model, "cap": cap, "suffix": suffix}
            )
    return {
        "grounding": {"include": grounding},
        "osworld": {"include": osworld},
        "has_grounding": bool(grounding),
        "has_osworld": bool(osworld),
        "total_cap": f"{total_cap(cells):g}",
    }


def summary_markdown(cells: list[Cell], max_total_usd: float) -> str:
    lines = [
        "## Paid run list",
        "",
        f"{len(cells)} cells; cost caps add up to ${total_cap(cells):.2f}"
        + (f" (batch cap ${max_total_usd:.2f})." if max_total_usd > 0 else " (no batch cap)."),
        "Grounding cells run side by side; OSWorld cells run one after another on the OSWorld VM.",
        "",
        "| # | Cell | Code | Cap |",
        "|---|---|---|---|",
    ]
    for i, cell in enumerate(cells, 1):
        code = f"PR #{cell.pr} at `{cell.ref[:12]}`" if cell.pr else (f"`{cell.ref}`" if cell.ref else "default branch")
        lines.append(f"| {i} | {cell.label} | {code} | ${cell.cap:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-total-usd", type=float, default=DEFAULT_MAX_TOTAL_USD, help="batch cost cap (0: none)")
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY", ""), help="owner/name, for pr= cells")
    parser.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT"), help="write the matrices here")
    parser.add_argument("--summary", default=os.getenv("GITHUB_STEP_SUMMARY"), help="append the batch table here")
    args = parser.parse_args(argv)
    try:
        cells = parse_batch(sys.stdin.read(), args.max_total_usd)
        if any(cell.pr for cell in cells):
            if not args.repository:
                raise BatchError("pr= cells need --repository (or $GITHUB_REPOSITORY).")
            resolve_refs(cells, lambda pr: gh_pr_head(pr, args.repository))
    except (BatchError, subprocess.CalledProcessError) as exc:
        print(f"::error::{exc}")
        return 1
    table = summary_markdown(cells, args.max_total_usd)
    print(table)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(table)
    if args.github_output:
        out = matrices(cells)
        with open(args.github_output, "a", encoding="utf-8") as fh:
            for key in ("grounding", "osworld"):
                fh.write(f"{key}={json.dumps(out[key], separators=(',', ':'))}\n")
            for key in ("has_grounding", "has_osworld"):
                fh.write(f"{key}={str(out[key]).lower()}\n")
            fh.write(f"total_cap={out['total_cap']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
