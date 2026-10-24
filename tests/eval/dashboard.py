"""Store live benchmark reports in the eval dashboard's Postgres database.

The dashboard (``dashboard/``, deployed on Vercel) reads benchmark runs from a
Neon Postgres database through its API routes. This script writes reports from
``live_bench`` into that database: one ``bench_runs`` row per run and one
``bench_task_results`` row per task. Importing the same report again replaces it.

    DATABASE_URL=postgres://... .venv/bin/python -m tests.eval.dashboard import .var/bench/full-20261024T012945Z.json
    DATABASE_URL=postgres://... .venv/bin/python -m tests.eval.dashboard import downloaded-artifacts/ --backend json

It needs ``psycopg`` (``pip install "psycopg[binary]"``), which is not a project
dependency: only the "Dashboard data" workflow and a person backfilling by hand
run it.

The dashboard groups runs by ``config`` (backend, harness, model, commit, workflow
run), filled from the report's own ``backend``/``claude_model``/``harness_profile``
fields and from the options below. Older reports don't record the backend or
harness; pass ``--backend``/``--harness``/``--model`` for those, and
``--run-url``/``--git-sha`` to link a run to its workflow and commit. ``--flag
REASON`` marks a run whose numbers are not comparable (a broken harness, a bad
runner), which the dashboard leaves out of its trends by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Per-task fields that only make sense on the machine that ran the benchmark.
DROPPED_RESULT_FIELDS = ("final_screenshot",)
TASK_COLUMNS = (
    "task", "category", "outcome", "steps", "seconds", "cost_usd", "input_tokens",
    "output_tokens", "cache_read_tokens", "cache_write_tokens", "judge_cost_usd",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bench_runs (
    key             text PRIMARY KEY,
    mode            text NOT NULL,
    suite           text NOT NULL,
    backend         text,
    harness_profile text,
    model           text,
    started_at      timestamptz NOT NULL,
    judge_model     text,
    git_sha         text,
    github_run_url  text,
    flagged         text,
    config          jsonb NOT NULL,
    summary         jsonb NOT NULL,
    imported_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS bench_runs_started_at ON bench_runs (started_at);
CREATE TABLE IF NOT EXISTS bench_task_results (
    run_key            text NOT NULL REFERENCES bench_runs (key) ON DELETE CASCADE,
    position           integer NOT NULL,
    task               text NOT NULL,
    category           text,
    outcome            text NOT NULL,
    steps              integer,
    seconds            integer,
    cost_usd           double precision,
    input_tokens       integer,
    output_tokens      integer,
    cache_read_tokens  integer,
    cache_write_tokens integer,
    judge_cost_usd     double precision,
    result             jsonb NOT NULL,
    PRIMARY KEY (run_key, task)
);
"""


def is_report(data: object) -> bool:
    return isinstance(data, dict) and isinstance(data.get("summary"), dict) and isinstance(data.get("results"), list)


def find_reports(paths: list[Path], include_partial: bool = False) -> list[Path]:
    """Every report JSON under ``paths`` (files or folders), oldest name first."""
    found: list[Path] = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*.json"))
        for candidate in candidates:
            if candidate.name.startswith("partial-") and not include_partial:
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if is_report(data):
                found.append(candidate)
    return found


def run_key(report: dict) -> str:
    config = report.get("config") or {}
    parts = [
        report.get("mode", "browser"),
        report["suite"].removeprefix("desktop-"),
        config.get("backend") or "unknown",
        config.get("harness_profile") or "full",
        report["started_at"],
    ]
    return "-".join(str(part).replace("/", "_") for part in parts)


def prepare(report: dict, overrides: dict[str, str | None]) -> dict:
    """The report as the dashboard stores it: overrides applied, runner-only fields dropped."""
    report = dict(report)
    report.setdefault("mode", "desktop" if report["suite"].startswith("desktop-") else "browser")
    config = dict(report.get("config") or {})
    # live_bench records these at the top level; "server default" means the env var was unset.
    known = {k: v for k in ("backend", "harness_profile") if (v := report.get(k)) and v != "server default"}
    for field, value in known.items():
        config.setdefault(field, value)
    model = report.get("claude_model") if config.get("backend") == "claude_computer_use" else report.get(f"{report['mode']}_model")
    if model and model != "server default":
        config.setdefault("model", model)
    for field, value in overrides.items():
        if value is not None:
            config[field] = value
    config.setdefault("harness_profile", "full")
    report["config"] = config
    report["results"] = [
        {k: v for k, v in result.items() if k not in DROPPED_RESULT_FIELDS} for result in report["results"]
    ]
    return report


def parse_started_at(stamp: str) -> datetime:
    """live_bench stamps look like ``20261024T012945Z`` (UTC)."""
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def to_rows(report: dict) -> tuple[dict, list[dict]]:
    """A prepared report as one ``bench_runs`` row and its ``bench_task_results`` rows."""
    config = report["config"]
    run = {
        "key": run_key(report),
        "mode": report["mode"],
        "suite": report["suite"],
        "backend": config.get("backend"),
        "harness_profile": config.get("harness_profile"),
        "model": config.get("model"),
        "started_at": parse_started_at(report["started_at"]),
        "judge_model": report.get("judge_model"),
        "git_sha": config.get("git_sha"),
        "github_run_url": config.get("github_run_url"),
        "flagged": config.get("flagged"),
        "config": config,
        "summary": report["summary"],
    }
    tasks = []
    seen = set()
    for position, result in enumerate(report["results"]):
        if result["task"] in seen:  # a task listed twice keeps its first result
            continue
        seen.add(result["task"])
        row = {column: result.get(column) for column in TASK_COLUMNS}
        row.update(run_key=run["key"], position=position, result=result)
        tasks.append(row)
    return run, tasks


def connect(database_url: str):
    try:
        import psycopg
    except ImportError:
        sys.exit('This needs psycopg: pip install "psycopg[binary]"')
    return psycopg.connect(database_url)


def write_report(conn, report: dict) -> str:
    """Insert or replace one prepared report; returns its key."""
    from psycopg.types.json import Jsonb

    run, tasks = to_rows(report)
    run_values = {**run, "config": Jsonb(run["config"]), "summary": Jsonb(run["summary"])}
    columns = list(run_values)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != "key")
    with conn.transaction():
        conn.execute(
            f"INSERT INTO bench_runs ({', '.join(columns)}, imported_at) "
            f"VALUES ({', '.join('%(' + c + ')s' for c in columns)}, now()) "
            f"ON CONFLICT (key) DO UPDATE SET {updates}, imported_at = now()",
            run_values,
        )
        conn.execute("DELETE FROM bench_task_results WHERE run_key = %s", (run["key"],))
        if tasks:
            task_columns = ["run_key", "position", *TASK_COLUMNS, "result"]
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO bench_task_results ({', '.join(task_columns)}) "
                    f"VALUES ({', '.join('%(' + c + ')s' for c in task_columns)})",
                    [{**t, "result": Jsonb(t["result"])} for t in tasks],
                )
    return run["key"]


def import_reports(paths: list[Path], conn, overrides: dict[str, str | None], include_partial: bool = False) -> list[str]:
    conn.execute(SCHEMA)
    conn.commit()
    keys = []
    for path in find_reports(paths, include_partial):
        report = prepare(json.loads(path.read_text(encoding="utf-8")), overrides)
        keys.append(write_report(conn, report))
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("import", help="write reports to the dashboard database")
    add.add_argument("paths", nargs="+", type=Path, help="report files or folders to search")
    add.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="Postgres URL (default: $DATABASE_URL)")
    add.add_argument("--backend", help="config.backend for reports that lack it")
    add.add_argument("--harness", help="config.harness_profile (full or thin)")
    add.add_argument("--model", help="config.model")
    add.add_argument("--run-url", help="config.github_run_url")
    add.add_argument("--git-sha", help="config.git_sha")
    add.add_argument("--flag", help="why this run is left out of trends")
    add.add_argument("--include-partial", action="store_true", help="also import partial-*.json (--only/--category runs)")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        print("Set DATABASE_URL (or pass --database-url) to the dashboard's Postgres database.", file=sys.stderr)
        return 2
    overrides = {
        "backend": args.backend,
        "harness_profile": args.harness,
        "model": args.model,
        "github_run_url": args.run_url or None,
        "git_sha": args.git_sha or None,
        "flagged": args.flag,
    }
    with connect(args.database_url) as conn:
        keys = import_reports(args.paths, conn, overrides, args.include_partial)
    for key in keys:
        print(f"imported {key}")
    if not keys:
        print("No benchmark reports found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
