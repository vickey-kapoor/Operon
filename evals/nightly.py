"""Plan the nightly paid run (KAN-165): which queued cells run tonight, within the budget.

Paid benchmark runs happen once a day, at 2 AM Chicago, in one scheduled run of
``.github/workflows/nightly.yml``. Threads queue what they need by adding a line to
``evals/nightly-queue.txt`` on main (any PR; it auto-merges once checks pass)::

    2026-11-14 desktop thin season=2 zoom=always cap=2.5   # KAN-82: zoom on whole tasks
    2026-11-14 hybrid full season=2 cap=2                  # KAN-161: first Full score

Each line is the date it was queued, then one batch cell in the words of
``evals/bench_batch.py`` (``ref=`` defaults to main), then an optional ``#`` note.
Every night this module picks the lines to run:

- lines already run on an earlier night are skipped (the workflow keeps the list of
  run lines between nights as an artifact), so a line runs once. To run the same cell
  again, add a new line (a new date or note makes it new);
- lines queued more than ``--max-age-days`` ago (default 7) are skipped, so a lost
  run list never re-runs old work;
- lines are taken in queue order while their caps fit tonight's budget: the nightly cap
  ($5, Vickey's choice) or what is left of the monthly budget ($100), whichever is less.
  A line that does not fit waits for the next night.

When the month's spend can't be read (no database), nothing runs: the budget check
fails closed. ``python -m evals.nightly plan`` prints the plan and, with
``--github-output``, writes it for the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from evals.bench_batch import BatchError, Cell, parse_line

CHICAGO = ZoneInfo("America/Chicago")
NIGHTLY_CAP_USD = 5.0
MONTH_CAP_USD = 100.0
MAX_AGE_DAYS = 7
QUEUE = Path(__file__).resolve().parent / "nightly-queue.txt"

_DATED = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(.+)$")


@dataclass
class Entry:
    line: str  # the queue line as written (stripped); its hash marks it as run
    queued: date
    cell: Cell
    cells_text: str  # the cell words alone, as passed to the batch

    @property
    def key(self) -> str:
        return hashlib.sha256(self.line.encode("utf-8")).hexdigest()[:16]


@dataclass
class Plan:
    tonight: list[Entry] = field(default_factory=list)
    waiting: list[Entry] = field(default_factory=list)  # did not fit tonight's budget
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (line, why)
    budget: float = 0.0

    @property
    def total_cap(self) -> float:
        return round(sum(e.cell.cap for e in self.tonight), 2)


def read_queue(text: str) -> tuple[list[Entry], list[tuple[str, str]]]:
    """The queue's dated cells, and the lines that can't run with the reason."""
    entries: list[Entry] = []
    bad: list[tuple[str, str]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _DATED.match(line)
        if not match:
            bad.append((line, "start the line with the date it was queued, e.g. 2026-11-14"))
            continue
        try:
            queued = date.fromisoformat(match.group(1))
        except ValueError:
            bad.append((line, "bad date"))
            continue
        words = match.group(2).split("#", 1)[0].strip()
        try:
            cell = parse_line(words, number)
        except BatchError as exc:
            bad.append((line, str(exc)))
            continue
        if cell is None:
            bad.append((line, "no cell after the date"))
            continue
        entries.append(Entry(line=line, queued=queued, cell=cell, cells_text=words))
    return entries, bad


def plan_night(
    entries: list[Entry],
    done: set[str],
    today: date,
    month_spend: float,
    nightly_cap: float = NIGHTLY_CAP_USD,
    month_cap: float = MONTH_CAP_USD,
    max_age_days: int = MAX_AGE_DAYS,
) -> Plan:
    """Tonight's cells: not run before, not too old, in queue order while the caps fit."""
    plan = Plan(budget=round(max(0.0, min(nightly_cap, month_cap - month_spend)), 2))
    seen: set[str] = set()
    for entry in entries:
        if entry.key in done:
            continue  # ran on an earlier night
        if entry.key in seen:
            plan.skipped.append((entry.line, "same line twice in the queue"))
            continue
        seen.add(entry.key)
        if (today - entry.queued).days > max_age_days:
            plan.skipped.append((entry.line, f"queued more than {max_age_days} days ago; add it again to run it"))
            continue
        if entry.queued > today:
            plan.waiting.append(entry)  # dated in the future: waits for its night
            continue
        if plan.total_cap + entry.cell.cap <= plan.budget + 1e-9:
            plan.tonight.append(entry)
        else:
            plan.waiting.append(entry)
    return plan


def month_spend_from_db(database_url: str, now: datetime) -> float:
    """This UTC month's benchmark spend, from the dashboard database."""
    from evals.dashboard import _MONTH_SQL, connect

    month_start = now.astimezone(ZoneInfo("UTC")).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with connect(database_url) as conn:
        return float(conn.execute(_MONTH_SQL, (month_start,)).fetchone()[0] or 0)


def summary_markdown(plan: Plan, month_spend: float, month_cap: float) -> str:
    lines = [
        "## Nightly paid run",
        "",
        f"Tonight's budget ${plan.budget:.2f} (nightly cap, or what is left of ${month_cap:.0f} this month: "
        f"${month_spend:.2f} spent). {len(plan.tonight)} cell(s), caps ${plan.total_cap:.2f}.",
        "",
    ]
    for title, rows in (
        ("Running tonight", [(e.line, f"cap ${e.cell.cap:.2f}") for e in plan.tonight]),
        ("Waiting for a later night", [(e.line, f"cap ${e.cell.cap:.2f}") for e in plan.waiting]),
        ("Skipped", plan.skipped),
    ):
        if rows:
            lines += [f"**{title}**", ""] + [f"- `{line}`: {why}" for line, why in rows] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="pick tonight's cells")
    p.add_argument("--queue", type=Path, default=QUEUE)
    p.add_argument("--done", type=Path, help="file of line keys already run (read, then rewritten with tonight's)")
    p.add_argument("--nightly-cap", type=float, default=NIGHTLY_CAP_USD)
    p.add_argument("--month-cap", type=float, default=MONTH_CAP_USD)
    p.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    p.add_argument("--month-spend", type=float, help="this month's spend (default: read from $DATABASE_URL)")
    p.add_argument("--dry-run", action="store_true", help="plan only: don't mark tonight's lines as run")
    p.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT"))
    p.add_argument("--summary", default=os.getenv("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args(argv)

    now = datetime.now(CHICAGO)
    if args.month_spend is not None:
        month_spend = args.month_spend
    else:
        url = os.getenv("DATABASE_URL")
        if not url:
            print("::error::Set DATABASE_URL (or pass --month-spend): without this month's spend nothing runs.")
            return 1
        try:
            month_spend = month_spend_from_db(url, now)
        except Exception as exc:  # fail closed: no budget check, no run
            print(f"::error::Could not read this month's spend ({exc}); nothing runs tonight.")
            return 1

    entries, bad = read_queue(args.queue.read_text(encoding="utf-8") if args.queue.exists() else "")
    done = set()
    if args.done and args.done.exists():
        done = {k.strip() for k in args.done.read_text(encoding="utf-8").split() if k.strip()}
    plan = plan_night(entries, done, now.date(), month_spend, args.nightly_cap, args.month_cap, args.max_age_days)
    plan.skipped += bad

    table = summary_markdown(plan, month_spend, args.month_cap)
    print(table)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(table)
    if args.done and not args.dry_run:
        # Marked before the run starts, so a cell that fails is not paid for again every night.
        keys = sorted(done | {e.key for e in plan.tonight})
        args.done.parent.mkdir(parents=True, exist_ok=True)
        args.done.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    if args.github_output:
        delim = "CELLS_" + hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"run={'true' if plan.tonight and not args.dry_run else 'false'}\n")
            fh.write(f"budget={plan.budget:g}\n")
            fh.write(f"cells<<{delim}\n" + "\n".join(e.cells_text for e in plan.tonight) + f"\n{delim}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
