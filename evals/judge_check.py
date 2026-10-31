"""Check the live benchmark's judge against human labels.

Every live benchmark number rests on the judge: a model that reads a run's final
screenshot and says whether the task's ``check`` is visible. This measures how
often it agrees with a person.

1. Make a labels file from a saved report (``live_bench --keep-screens`` keeps the
   screenshots next to it)::

       .venv/bin/python -m evals.judge_check template .var/bench/full-<stamp>.json >> labels.jsonl

   Each line names a task and a screenshot, with what the judge said at the time
   (``judge_then``) and ``"label": null``.

2. Look at each screenshot and set ``label`` to ``true`` (the task's check is
   visible) or ``false``. Lines left ``null`` are skipped. Add a ``note`` if useful.

3. Run the judge on every labelled line and compare (needs ``GEMINI_API_KEY``; one
   judge call per line and repeat, no agent runs)::

       .venv/bin/python -m evals.judge_check run labels.jsonl [--repeat 3] [--judge-model M]

The report gives precision (when the judge says "done", how often it is right),
recall (how many done tasks it sees), accuracy, the disagreements, and with
``--repeat`` how often the judge gives the same answer twice on one screenshot.
A judge false positive is the costly error: it turns a false success into a pass.

4. Add ``--save`` to record the result in the benchmark database (``DATABASE_URL``),
   where the dashboard's Judge accuracy card reads it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.desktop_tasks import DESKTOP_TASKS
from evals.live_tasks import TASKS, LiveTask

# What each outcome says the judge saw on the final screen.
JUDGED_VISIBLE = {"success": True, "missed": True, "false_success": False, "failure": False}


@dataclass(frozen=True)
class Label:
    task: str
    screenshot: str
    label: bool


@dataclass(frozen=True)
class Graded:
    label: Label
    verdicts: tuple[bool | None, ...]

    @property
    def verdict(self) -> bool | None:
        """The majority answer over repeats; a tie or no answer is None."""
        yes = sum(v is True for v in self.verdicts)
        no = sum(v is False for v in self.verdicts)
        return None if yes == no else yes > no


def template(report_path: Path, screens: Path | None = None) -> list[dict]:
    """One unlabelled line per graded task in a saved ``live_bench`` report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    screens = screens or report_path.with_suffix("")
    lines = []
    for result in report.get("results", []):
        if result.get("outcome") not in JUDGED_VISIBLE:
            continue  # errors and timeouts were never judged; safety tasks never are
        kept = screens / f"{result['task']}.png"
        shot = str(kept) if kept.is_file() else result.get("final_screenshot", "-")
        if shot == "-":
            continue
        lines.append({
            "task": result["task"],
            "screenshot": shot,
            "report": report_path.name,
            "judge_then": JUDGED_VISIBLE[result["outcome"]],
            "label": None,
            "note": "",
        })
    return lines


def read_labels(path: Path) -> tuple[list[Label], int]:
    """The labelled lines of a labels file, and how many were left unlabelled."""
    labels, unlabelled = [], 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("label") is None:
            unlabelled += 1
            continue
        if not isinstance(row["label"], bool):
            raise ValueError(f"{path}:{number}: label must be true, false or null")
        labels.append(Label(row["task"], row["screenshot"], row["label"]))
    return labels, unlabelled


def score(graded: list[Graded]) -> dict:
    """Agreement of the judge's majority verdict with the human labels (positive: goal visible)."""
    answered = [g for g in graded if g.verdict is not None]
    tp = sum(g.verdict and g.label.label for g in answered)
    fp = sum(g.verdict and not g.label.label for g in answered)
    tn = sum(not g.verdict and not g.label.label for g in answered)
    fn = sum(not g.verdict and g.label.label for g in answered)
    repeated = [g for g in graded if len(g.verdicts) > 1]
    return {
        "labelled": len(graded),
        "no_verdict": len(graded) - len(answered),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": round((tp + tn) / len(answered), 3) if answered else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "self_agreement": (
            round(sum(len(set(g.verdicts)) == 1 for g in repeated) / len(repeated), 3) if repeated else None
        ),
    }


def format_score(graded: list[Graded], summary: dict, unlabelled: int) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    lines = []
    for g in graded:
        if g.verdict is not g.label.label:
            said = {True: "done", False: "not done", None: "no verdict"}[g.verdict]
            lines.append(f"  {g.label.task:22} human: {'done' if g.label.label else 'not done':8} judge: {said:10} {g.label.screenshot}")
    head = [
        f"{summary['labelled']} labelled screenshots ({unlabelled} unlabelled skipped), {summary['no_verdict']} without a verdict",
        f"accuracy {pct(summary['accuracy'])} · precision {pct(summary['precision'])} · recall {pct(summary['recall'])}"
        + (f" · self-agreement {pct(summary['self_agreement'])}" if summary["self_agreement"] is not None else ""),
        f"judge said done: {summary['true_positive']} right, {summary['false_positive']} wrong (false passes) · "
        f"judge said not done: {summary['true_negative']} right, {summary['false_negative']} wrong (missed passes)",
    ]
    return "\n".join(head + (["", "disagreements:"] + lines if lines else []))


JUDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_checks (
    id            bigserial PRIMARY KEY,
    checked_at    timestamptz NOT NULL DEFAULT now(),
    judge_model   text,
    labels        integer NOT NULL,
    repeats       integer NOT NULL,
    accuracy      double precision,
    precision     double precision,
    recall        double precision,
    agreement     double precision,
    false_passes  integer NOT NULL,
    false_fails   integer NOT NULL,
    disagreements jsonb NOT NULL
);
"""


def check_row(graded: list[Graded], summary: dict, judge_model: str, repeat: int) -> dict:
    """One ``judge_checks`` row. Screenshots are named by file only: local paths stay local."""
    return {
        "judge_model": judge_model,
        "labels": summary["labelled"],
        "repeats": repeat,
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "agreement": summary["self_agreement"],
        "false_passes": summary["false_positive"],
        "false_fails": summary["false_negative"],
        "disagreements": [
            {"task": g.label.task, "screenshot": Path(g.label.screenshot).name, "label": g.label.label, "judge": g.verdict}
            for g in graded
            if g.verdict is not g.label.label
        ],
    }


def save_check(conn, row: dict) -> None:
    from psycopg.types.json import Jsonb

    with conn.transaction():
        conn.execute(JUDGE_SCHEMA)
        values = {**row, "disagreements": Jsonb(row["disagreements"])}
        conn.execute(
            f"INSERT INTO judge_checks ({', '.join(values)}) VALUES ({', '.join('%(' + c + ')s' for c in values)})",
            values,
        )


async def judge_labels(labels: list[Label], judge, repeat: int = 1, concurrency: int = 4) -> list[Graded]:
    tasks: dict[str, LiveTask] = {t.name: t for t in [*TASKS, *DESKTOP_TASKS]}
    gate = asyncio.Semaphore(concurrency)

    async def one(label: Label) -> Graded:
        task = tasks.get(label.task)
        if task is None or not Path(label.screenshot).is_file():
            return Graded(label, (None,))
        async with gate:
            verdicts = [(await judge(task, label.screenshot)).goal_visible for _ in range(repeat)]
        return Graded(label, tuple(verdicts))

    return list(await asyncio.gather(*(one(label) for label in labels)))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("template", help="print unlabelled lines for a saved report")
    make.add_argument("report", type=Path)
    make.add_argument("--screens", type=Path, help="screenshot folder (default: next to the report)")
    run = commands.add_parser("run", help="judge every labelled line and compare")
    run.add_argument("labels", type=Path)
    run.add_argument("--repeat", type=int, default=1, help="judge each screenshot this many times")
    run.add_argument("--judge-model", default=os.getenv("OPERON_BENCH_JUDGE_MODEL", "gemini-3.5-flash"))
    run.add_argument("--json", action="store_true")
    run.add_argument("--save", action="store_true", help="record the result in the benchmark database (DATABASE_URL)")
    args = parser.parse_args(argv)

    if args.command == "template":
        for line in template(args.report, args.screens):
            print(json.dumps(line))
        return 0

    from evals.live_bench import gemini_judge

    database_url = os.getenv("DATABASE_URL")
    if args.save and not database_url:
        print("--save needs DATABASE_URL.", file=sys.stderr)
        return 2
    labels, unlabelled = read_labels(args.labels)
    if not labels:
        print("No labelled lines: set \"label\" to true or false first.", file=sys.stderr)
        return 2
    graded = asyncio.run(judge_labels(labels, gemini_judge(args.judge_model), repeat=max(1, args.repeat)))
    summary = score(graded)
    if args.json:
        print(json.dumps({"judge_model": args.judge_model, **summary}, indent=2))
    else:
        print(f"judge {args.judge_model}")
        print(format_score(graded, summary, unlabelled))
    if args.save:
        from evals.dashboard import connect

        with connect(database_url) as conn:
            save_check(conn, check_row(graded, summary, args.judge_model, max(1, args.repeat)))
        print("saved to judge_checks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
