"""Read a benchmark batch: many grid cells, one GitHub Actions run (KAN-105).

Paid runs are queued during the day and run together once Vickey approves the batch.
The batch workflow (``.github/workflows/bench-batch.yml``, or a ``/bench batch`` PR
comment) passes its cell list here; this module checks every line and turns it into
the job matrices the workflow runs: Browser cells side by side on hosted runners,
Desktop and Hybrid cells one after another on a single start of the Azure VM.

One cell per line (or separated by ``;``), in the words of a ``/bench`` comment, plus where to run it and its
cost cap::

    browser thin season=2 pr=242
    desktop thin season=2 pr=242 cap=1.5
    hybrid full ref=main

Words: ``browser|desktop|hybrid``, ``thin|full``, ``core|smoke|all|safety|apps|hard|long``,
``x2|x3|x5``, ``same-critic``, ``season=<n>``, ``pr=<number>`` (that PR's head) or
``ref=<branch, tag or commit>`` (default: the default branch), ``cap=<USD>`` (the cell's
cost cap, default 3). ``azure`` and ``gemini`` are accepted and change nothing: batch
Desktop and Hybrid cells always run on the Azure VM. Blank lines and ``#`` comments are
skipped. The batch is refused when its cells' caps add up to more than the batch cap,
so the most it can spend is known before anything starts.

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

CATEGORIES = ("browser", "desktop", "hybrid")
SUITES = {"core": "core", "smoke": "smoke", "all": "full", "safety": "safety", "apps": "apps", "hard": "hard", "long": "long"}
BROWSER_ONLY_SUITES = {"apps", "hard"}
HYBRID_SUITES = {"core", "smoke", "full"}
DEFAULT_CAP_USD = 3.0
DEFAULT_MAX_TOTAL_USD = 10.0
MAX_CELLS = 20

_SEASON = re.compile(r"^[1-9][0-9]{0,2}$")
_PR = re.compile(r"^[1-9][0-9]{0,6}$")
# Branch, tag or commit: git ref characters only, no leading dash (it would read as an option).
_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}$")
_CAP = re.compile(r"^[0-9]{1,3}(\.[0-9]{1,2})?$")


class BatchError(ValueError):
    """A batch line that can't run; the message names the line."""


@dataclass
class Cell:
    category: str
    profile: str = "full"
    suite: str = "core"
    repeat: int = 1
    critic: str = ""
    season: str = ""
    pr: str = ""
    ref: str = ""
    cap: float = DEFAULT_CAP_USD

    @property
    def extra(self) -> str:
        """The live_bench arguments the workflow passes on, as in a /bench run."""
        words = []
        if self.repeat != 1:
            words += ["--repeat", str(self.repeat)]
        if self.season:
            words += ["--season", self.season]
        return " ".join(words)

    @property
    def label(self) -> str:
        """The cell's name the way runs are named in chat: "Gemini · Browser · Thin"."""
        label = f"Gemini · {self.category.capitalize()} · {self.profile.capitalize()}"
        if self.suite != "core":
            label += f" · {self.suite}"
        if self.season:
            label += f" · Season {self.season}"
        if self.repeat != 1:
            label += f" · x{self.repeat}"
        if self.critic == "same":
            label += " · same critic"
        if self.pr:
            label += f" · PR #{self.pr}"
        elif self.ref:
            label += f" · {self.ref[:12]}"
        return label


def parse_line(line: str, number: int) -> Cell | None:
    """One batch line as a cell, or None for a blank or comment line."""
    text = line.split("#", 1)[0].strip().lower()
    if not text:
        return None
    where = f"line {number} ({text!r})"
    category = ""
    cell = Cell(category="browser")
    for word in text.split():
        if word in ("/bench", "gemini", "azure"):
            continue
        if word in CATEGORIES:
            if category:
                raise BatchError(f"{where}: name one category (browser, desktop or hybrid).")
            category = word
        elif word == "grounding":
            raise BatchError(f"{where}: grounding is not part of a batch; run /bench grounding on its own.")
        elif word in ("thin", "full"):
            cell.profile = word
        elif word in SUITES:
            cell.suite = SUITES[word]
        elif word in ("x2", "x3", "x5"):
            cell.repeat = int(word[1:])
        elif word == "same-critic":
            cell.critic = "same"
        elif word.startswith("season="):
            cell.season = word.removeprefix("season=")
            if not _SEASON.match(cell.season):
                raise BatchError(f"{where}: season= needs a season number such as 1.")
        elif word.startswith("pr="):
            cell.pr = word.removeprefix("pr=")
            if not _PR.match(cell.pr):
                raise BatchError(f"{where}: pr= needs a pull request number such as 242.")
        elif word.startswith("ref="):
            cell.ref = line.split("#", 1)[0].split("ref=", 1)[1].split()[0]  # keep the ref's own case
            if not _REF.match(cell.ref) or ".." in cell.ref:
                raise BatchError(f"{where}: ref= needs a branch, tag or commit.")
        elif word.startswith("cap="):
            value = word.removeprefix("cap=").lstrip("$")
            if not _CAP.match(value) or float(value) <= 0:
                raise BatchError(f"{where}: cap= needs a cost in USD above 0, such as 1.5.")
            cell.cap = float(value)
        else:
            raise BatchError(f"{where}: unknown word {word!r}.")
    if not category:
        raise BatchError(f"{where}: start with the category: browser, desktop or hybrid.")
    cell.category = category
    if cell.pr and cell.ref:
        raise BatchError(f"{where}: give pr= or ref=, not both.")
    if cell.suite in BROWSER_ONLY_SUITES and category != "browser":
        raise BatchError(f"{where}: the {cell.suite} suite is browser only.")
    if category == "hybrid" and cell.suite not in HYBRID_SUITES:
        raise BatchError(f"{where}: Hybrid has no {cell.suite} suite; use core, smoke or all.")
    if cell.critic == "same" and cell.profile != "full":
        raise BatchError(f"{where}: same-critic picks the full harness's step critic; drop thin.")
    return cell


def parse_batch(text: str, max_total_usd: float = DEFAULT_MAX_TOTAL_USD) -> list[Cell]:
    """Every cell of a batch, checked as a whole: at least one, no repeats, caps within the batch cap."""
    # A ";" also ends a cell: a workflow_dispatch input is one line in the Actions form.
    lines = text.replace(";", "\n").splitlines()
    cells = [cell for n, line in enumerate(lines, 1) if (cell := parse_line(line, n))]
    if not cells:
        raise BatchError("The batch has no cells. Put one per line, e.g. 'browser thin season=2 pr=242'.")
    if len(cells) > MAX_CELLS:
        raise BatchError(f"The batch has {len(cells)} cells; the most is {MAX_CELLS}.")
    seen: set[tuple] = set()
    for cell in cells:
        key = (cell.category, cell.profile, cell.suite, cell.repeat, cell.critic, cell.season, cell.pr, cell.ref)
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

    def entry(index: int, cell: Cell) -> dict:
        return {
            "label": cell.label,
            "category": cell.category,
            "profile": cell.profile,
            "suite": cell.suite,
            "critic": cell.critic,
            "extra": cell.extra,
            "ref": cell.ref,
            "cap": f"{cell.cap:g}",
            "suffix": f"-cell{index}",
        }

    numbered = list(enumerate(cells, 1))
    browser = [entry(i, c) for i, c in numbered if c.category == "browser"]
    desktop = [entry(i, c) for i, c in numbered if c.category != "browser"]
    return {
        "browser": {"include": browser},
        "desktop": {"include": desktop},
        "has_browser": bool(browser),
        "has_desktop": bool(desktop),
        "total_cap": f"{total_cap(cells):g}",
    }


def summary_markdown(cells: list[Cell], max_total_usd: float) -> str:
    lines = [
        "## Benchmark batch",
        "",
        f"{len(cells)} cells; cost caps add up to ${total_cap(cells):.2f}"
        + (f" (batch cap ${max_total_usd:.2f})." if max_total_usd > 0 else " (no batch cap)."),
        "Browser cells run side by side; Desktop and Hybrid cells run one after another on one start of the Azure VM.",
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
            for key in ("browser", "desktop"):
                fh.write(f"{key}={json.dumps(out[key], separators=(',', ':'))}\n")
            for key in ("has_browser", "has_desktop"):
                fh.write(f"{key}={str(out[key]).lower()}\n")
            fh.write(f"total_cap={out['total_cap']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
