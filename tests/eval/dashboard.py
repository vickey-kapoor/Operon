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
runner), which the dashboard leaves out of its trends by default. A run stopped at
its cost cap (``budget_stop`` in the report) is flagged on its own.

``--blob`` uploads each task's screenshots to Vercel Blob (``BLOB_READ_WRITE_TOKEN``)
from the folder ``live_bench --keep-screens`` writes next to the report: the graded
final screen for every task and, for tasks that did not pass, every step's screen
and a zip of the whole run folder. The links go into the task's stored result
(``screenshot_url``, ``step_screenshots``, ``trace_url``). The upload itself runs
``dashboard/scripts/blob-upload.mjs`` (Vercel's ``@vercel/blob`` client), so it needs
``npm ci`` in ``dashboard/`` first. The store is private; the dashboard reads the
files through its ``/api/blob`` route.

``--alert`` posts to a Slack incoming webhook (``SLACK_WEBHOOK_URL``) when a run's
pass rate is 5 or more points below the previous comparable run (same mode, suite,
backend, harness and model, not flagged), when its cost per task is 25% or more
above it, or when it stopped at its cost cap.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
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


DASHBOARD_URL = "https://operon-evals.vercel.app"
BLOB_SCRIPT = Path(__file__).resolve().parents[2] / "dashboard" / "scripts" / "blob-upload.mjs"
# A task's step screens beyond this many are left out of the dashboard (the zip has them all).
MAX_STEP_SCREENS = 40
PASS_DROP_POINTS = 5.0
COST_RISE_PCT = 25.0

# Takes (local file, blob pathname) pairs; returns each file's public URL, in order.
Uploader = Callable[[list[tuple[Path, str]]], list[str]]


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
    stop = report.get("budget_stop")
    if stop and not config.get("flagged"):
        config["flagged"] = (
            f"stopped at the ${stop['limit_usd']:g} cost cap after {stop['tasks_run']} of {stop['tasks_planned']} tasks"
        )
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


# ---------- screenshots and traces (Vercel Blob) ----------


def _step_number(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.parent.name)
    return int(match.group(1)) if match else 0


# Task names become file names under the screens folder. The report comes from the
# benchmarked commit, which may be a PR branch, so anything else (a path separator,
# "..", an absolute path) is refused rather than followed out of that folder.
_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def _inside(path: Path, root: Path) -> bool:
    resolved, base = path.resolve(), root.resolve()
    return resolved == base or base in resolved.parents


def trace_files(screens: Path, task: str) -> tuple[Path | None, list[tuple[int, Path]], Path | None]:
    """A task's graded screen, its step screens (step number, file) and its run folder.

    ``live_bench --keep-screens`` writes ``<task>.png`` for every task and, for a task
    that did not pass, its run folder at ``<task>/`` with ``step_<n>/before.png``.
    A task name that is not a plain file name gets nothing.
    """
    if not _TASK_NAME_RE.match(task) or ".." in task:
        return None, [], None
    final = screens / f"{task}.png"
    run_dir = screens / task
    if not (_inside(final, screens) and _inside(run_dir, screens)):
        return None, [], None
    steps = sorted(
        ((_step_number(p), p) for p in run_dir.glob("step_*/before.png") if _inside(p, run_dir)),
        key=lambda s: s[0],
    )
    final_ok = final.is_file() and not final.is_symlink()
    run_dir_ok = run_dir.is_dir() and not run_dir.is_symlink()
    return (final if final_ok else None), steps[:MAX_STEP_SCREENS], (run_dir if run_dir_ok else None)


def attach_traces(report: dict, screens: Path, uploader: Uploader, work_dir: Path) -> int:
    """Upload a prepared report's screens and record their links in its results; returns the file count."""
    key = run_key(report)
    uploads: list[tuple[Path, str]] = []
    slots: list[tuple[dict, str, int | None]] = []  # (result, field, step) for each upload, in order
    for result in report["results"]:
        task = str(result["task"])
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task)
        final, steps, run_dir = trace_files(screens, task)
        if final:
            uploads.append((final, f"bench/{key}/{safe}/final.png"))
            slots.append((result, "screenshot_url", None))
        for number, path in steps:
            uploads.append((path, f"bench/{key}/{safe}/step_{number:03d}.png"))
            slots.append((result, "step_screenshots", number))
        if run_dir:
            archive = shutil.make_archive(str(work_dir / f"{key}-{safe}"), "zip", run_dir)
            uploads.append((Path(archive), f"bench/{key}/{safe}/trace.zip"))
            slots.append((result, "trace_url", None))
    if not uploads:
        return 0
    urls = uploader(uploads)
    if len(urls) != len(uploads):
        raise RuntimeError(f"uploaded {len(urls)} of {len(uploads)} files")
    for (result, field, step), url in zip(slots, urls):
        if field == "step_screenshots":
            result.setdefault(field, []).append({"step": step, "url": url})
        else:
            result[field] = url
    return len(uploads)


def blob_uploader(token: str) -> Uploader:
    """Upload through ``dashboard/scripts/blob-upload.mjs``, which uses Vercel's own client."""

    def upload(files: list[tuple[Path, str]]) -> list[str]:
        plan = [{"file": str(path), "pathname": pathname} for path, pathname in files]
        done = subprocess.run(
            ["node", str(BLOB_SCRIPT)],
            input=json.dumps(plan), capture_output=True, text=True, check=False,
            env={**os.environ, "BLOB_READ_WRITE_TOKEN": token},
        )
        if done.returncode != 0:
            raise RuntimeError(f"blob upload failed: {done.stderr.strip()[-500:]}")
        return json.loads(done.stdout)

    return upload


# ---------- alerts ----------


def previous_comparable(conn, run: dict) -> dict | None:
    """The latest unflagged run before ``run`` with the same mode, suite, backend, harness and model."""
    row = conn.execute(
        """SELECT key, summary, github_run_url FROM bench_runs
            WHERE mode = %s AND suite = %s AND backend IS NOT DISTINCT FROM %s
              AND harness_profile IS NOT DISTINCT FROM %s AND model IS NOT DISTINCT FROM %s
              AND flagged IS NULL AND started_at < %s AND key <> %s
            ORDER BY started_at DESC LIMIT 1""",
        (run["mode"], run["suite"], run["backend"], run["harness_profile"], run["model"], run["started_at"], run["key"]),
    ).fetchone()
    return {"key": row[0], "summary": row[1], "github_run_url": row[2]} if row else None


def regressions(
    summary: dict, previous: dict | None, *, pass_drop: float = PASS_DROP_POINTS, cost_rise: float = COST_RISE_PCT
) -> list[str]:
    """Why a run's numbers are worse than the previous comparable run's, one line each."""
    if not previous:
        return []
    found = []
    rate, before = summary.get("success_rate"), previous.get("success_rate")
    if rate is not None and before is not None and (before - rate) * 100 >= pass_drop - 1e-9:
        found.append(
            f"Pass rate fell {(before - rate) * 100:.0f} points: {rate:.0%} "
            f"({summary.get('success')}/{summary.get('tasks')}), was {before:.0%} ({previous.get('success')}/{previous.get('tasks')})."
        )
    cost, cost_before = summary.get("cost_per_task_usd"), previous.get("cost_per_task_usd")
    if cost is not None and cost_before and (cost / cost_before - 1) * 100 >= cost_rise - 1e-9:
        found.append(
            f"Cost per task rose {(cost / cost_before - 1) * 100:.0f}%: ${cost:.4f}, was ${cost_before:.4f}."
        )
    return found


def alert_text(report: dict, problems: list[str]) -> str:
    config = report["config"]
    label = " · ".join(str(v) for v in (report["mode"], report["suite"], config.get("backend"), config.get("harness_profile"), config.get("model")) if v)
    lines = [f":warning: *Operon benchmark* ({label}, {report['started_at']})", *(f"• {p}" for p in problems)]
    links = [f"<{DASHBOARD_URL}|Dashboard>"]
    if config.get("github_run_url"):
        links.append(f"<{config['github_run_url']}|Workflow run>")
    lines.append(" · ".join(links))
    return "\n".join(lines)


def post_slack(webhook_url: str, text: str) -> None:
    request = urllib.request.Request(
        webhook_url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - the webhook URL is ours
        response.read()


def check_alerts(conn, report: dict, notify: Callable[[str], None]) -> list[str]:
    """Compare a just-written report with the previous comparable run; notify if it got worse."""
    run, _ = to_rows(report)
    problems = []
    stop = report.get("budget_stop")
    if stop:
        problems.append(
            f"Stopped at the ${stop['limit_usd']:g} cost cap after {stop['tasks_run']} of "
            f"{stop['tasks_planned']} tasks (spent ${stop['spent_usd']:.2f})."
        )
    elif not run["flagged"]:
        previous = previous_comparable(conn, run)
        problems += regressions(report["summary"], previous["summary"] if previous else None)
    if problems:
        notify(alert_text(report, problems))
    return problems


def import_reports(
    paths: list[Path],
    conn,
    overrides: dict[str, str | None],
    include_partial: bool = False,
    *,
    uploader: Uploader | None = None,
    notify: Callable[[str], None] | None = None,
) -> list[str]:
    conn.execute(SCHEMA)
    conn.commit()
    keys = []
    with tempfile.TemporaryDirectory() as work_dir:
        for path in find_reports(paths, include_partial):
            report = prepare(json.loads(path.read_text(encoding="utf-8")), overrides)
            screens = path.with_suffix("")
            if uploader and screens.is_dir():
                try:
                    count = attach_traces(report, screens, uploader, Path(work_dir))
                    print(f"uploaded {count} screenshot and trace files for {path.name}")
                except Exception as exc:  # the results matter more than their screenshots
                    print(f"::warning::Could not upload the screenshots for {path.name}: {exc}", file=sys.stderr)
            keys.append(write_report(conn, report))
            if notify:
                try:
                    check_alerts(conn, report, notify)
                except Exception as exc:  # an alert failing must not lose the import
                    print(f"::warning::Could not check or send the alert for {path.name}: {exc}", file=sys.stderr)
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
    add.add_argument("--blob", action="store_true", help="upload screenshots and traces to Vercel Blob ($BLOB_READ_WRITE_TOKEN)")
    add.add_argument("--alert", action="store_true", help="post regressions to Slack ($SLACK_WEBHOOK_URL)")
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
    uploader = notify = None
    if args.blob:
        token = os.getenv("BLOB_READ_WRITE_TOKEN")
        if token:
            uploader = blob_uploader(token)
        else:
            print("::warning::BLOB_READ_WRITE_TOKEN is not set; importing without screenshots.", file=sys.stderr)
    if args.alert:
        webhook = os.getenv("SLACK_WEBHOOK_URL")
        if webhook:
            notify = lambda text: post_slack(webhook, text)  # noqa: E731
        else:
            print("::warning::SLACK_WEBHOOK_URL is not set; no alerts.", file=sys.stderr)
    with connect(args.database_url) as conn:
        keys = import_reports(args.paths, conn, overrides, args.include_partial, uploader=uploader, notify=notify)
    for key in keys:
        print(f"imported {key}")
    if not keys:
        print("No benchmark reports found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
