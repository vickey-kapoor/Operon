"""Feed live benchmark reports to the eval dashboard in ``dashboard/``.

The dashboard is a static site: ``dashboard/data/index.json`` lists every run
with its summary, and ``dashboard/data/reports/<key>.json`` holds each run's
per-task results. This script copies reports written by ``live_bench`` into that
folder and rebuilds the index:

    .venv/bin/python -m tests.eval.dashboard import .var/bench/full-20261024T012945Z.json
    .venv/bin/python -m tests.eval.dashboard import downloaded-artifacts/ --backend json
    .venv/bin/python -m tests.eval.dashboard index

The dashboard groups runs by ``config`` (backend, harness, model, commit, workflow
run), filled from the report's own ``backend``/``claude_model``/``harness_profile``
fields and from the options below. Older reports don't record the backend or
harness; pass ``--backend``/``--harness``/``--model`` for those, and
``--run-url``/``--git-sha`` to link a run to its workflow and commit. ``--flag REASON`` marks a run whose numbers are not comparable (a broken
harness, a bad runner), which the dashboard leaves out of its trends by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "dashboard" / "data"
# Per-task fields that only make sense on the machine that ran the benchmark.
DROPPED_RESULT_FIELDS = ("final_screenshot",)
INDEX_FIELDS = ("suite", "mode", "started_at", "judge_model", "summary")


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


def import_reports(paths: list[Path], data_dir: Path, overrides: dict[str, str | None], include_partial: bool = False) -> list[Path]:
    target = data_dir / "reports"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for path in find_reports(paths, include_partial):
        report = prepare(json.loads(path.read_text(encoding="utf-8")), overrides)
        out = target / f"{run_key(report)}.json"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        written.append(out)
    return written


def build_index(data_dir: Path) -> dict:
    runs = []
    for path in sorted((data_dir / "reports").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        entry = {"key": path.stem, "file": f"reports/{path.name}"}
        entry.update({field: report.get(field) for field in INDEX_FIELDS})
        entry["config"] = report.get("config") or {}
        runs.append(entry)
    runs.sort(key=lambda run: run["started_at"])
    index = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "runs": runs}
    (data_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("import", help="copy reports into the dashboard and rebuild the index")
    add.add_argument("paths", nargs="+", type=Path, help="report files or folders to search")
    add.add_argument("--backend", help="config.backend for reports that lack it")
    add.add_argument("--harness", help="config.harness_profile (full or thin)")
    add.add_argument("--model", help="config.model")
    add.add_argument("--run-url", help="config.github_run_url")
    add.add_argument("--git-sha", help="config.git_sha")
    add.add_argument("--flag", help="why this run is left out of trends")
    add.add_argument("--include-partial", action="store_true", help="also import partial-*.json (--only/--category runs)")
    commands.add_parser("index", help="rebuild dashboard/data/index.json from the stored reports")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import":
        overrides = {
            "backend": args.backend,
            "harness_profile": args.harness,
            "model": args.model,
            "github_run_url": args.run_url or None,
            "git_sha": args.git_sha or None,
            "flagged": args.flag,
        }
        written = import_reports(args.paths, args.data_dir, overrides, args.include_partial)
        for path in written:
            print(f"imported {path}")
        if not written:
            print("No benchmark reports found.", file=sys.stderr)
    index = build_index(args.data_dir)
    print(f"index: {len(index['runs'])} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
