"""Run-over-run score changes for one grid cell (KAN-60).

A cell is one model, category and harness, for example "Gemini · Desktop · Full". For
each run of a cell this gives:

- ``score_range``: the pass rate with a bootstrapped 95% range over the cell's tasks.
- ``run_change``: the paired comparison with the cell's previous run on the tasks both
  ran, labelled Improved / Regressed / No clear change by ``stats.compare``, plus the
  tasks that flipped from pass to fail or back.
- ``change_markdown``: the table a /bench PR comment shows.

A task counts as passed when its outcome is ``success``; a ``--repeat`` run's task
scores the share of its attempts that passed. Tasks, not attempts, are the samples.

With about 10 tasks and one pass each, the ranges are wide: a cell shows Improved or
Regressed only when about 4 of 10 tasks flipped the same way and none flipped back.
Smaller moves read "No clear change" rather than overclaiming.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from evals.leaderboard.stats import (
    IMPROVED,
    NO_CLEAR_CHANGE,
    REGRESSED,
    compare,
    summarize,
)

FIRST_RUN = "First run"
ALL = "all"


def task_pass_rates(results: Iterable[Mapping]) -> dict[str, float]:
    """Each task's share of passed attempts (1.0 or 0.0 for a single-pass run)."""
    attempts: dict[str, list[bool]] = {}
    for result in results:
        attempts.setdefault(str(result["task"]), []).append(result.get("outcome") == "success")
    return {task: sum(passed) / len(passed) for task, passed in attempts.items()}


def score_range(results: Iterable[Mapping]) -> dict | None:
    """The pass rate with its bootstrapped 95% range, 0-1 scale, or None with no tasks."""
    rates = task_pass_rates(results)
    if not rates:
        return None
    stats = summarize(rates, dict.fromkeys(rates, ALL), seed="score-range")[ALL]
    return {"score": round(stats["score"], 4), "low": round(stats["low"], 4), "high": round(stats["high"], 4), "tasks": stats["tasks"]}


def flipped_tasks(new: Mapping[str, float], old: Mapping[str, float]) -> tuple[list[str], list[str]]:
    """Tasks both runs have that went fail -> pass (fixed) and pass -> fail (broke).

    For a ``--repeat`` run a task flips when its pass share moved; a task only one run
    has is not a flip."""
    shared = sorted(set(new) & set(old))
    fixed = [t for t in shared if new[t] > old[t]]
    broke = [t for t in shared if new[t] < old[t]]
    return fixed, broke


def run_change(results: Iterable[Mapping], previous_results: Iterable[Mapping] | None, previous: Mapping | None = None) -> dict:
    """This run against the cell's previous run: label, paired difference with its 95% range, flips.

    ``previous`` describes the earlier run for the reader (key, started_at, success,
    tasks, github_run_url). With no previous run the label is "First run"."""
    if previous_results is None:
        return {"label": FIRST_RUN, "previous": None}
    new, old = task_pass_rates(results), task_pass_rates(previous_results)
    fixed, broke = flipped_tasks(new, old)
    info = {k: previous.get(k) for k in ("key", "started_at", "success", "tasks", "github_run_url")} if previous else {}
    shared = set(new) & set(old)
    if not shared:
        return {"label": NO_CLEAR_CHANGE, "previous": info, "shared_tasks": 0, "fixed": [], "broke": []}
    stats = compare(new, old, {task: ALL for task in shared}, seed="run-change")[ALL]
    return {
        "label": stats["label"],
        "difference": round(stats["difference"], 4),
        "low": round(stats["corrected_low"], 4),
        "high": round(stats["corrected_high"], 4),
        "shared_tasks": stats["tasks"],
        "fixed": fixed,
        "broke": broke,
        "previous": info,
    }


def fmt_range(rng: Mapping | None) -> str:
    if not rng:
        return "–"
    return f"{rng['score']:.0%} (range {rng['low']:.0%} to {rng['high']:.0%})"


def fmt_change(change: Mapping | None) -> str:
    if not change:
        return "–"
    label = change["label"]
    if "difference" not in change:
        return label
    icon = {IMPROVED: "🟢", REGRESSED: "🔴"}.get(label, "⚪")
    pts = round(change["difference"] * 100)
    return f"{icon} {label} ({'+' if pts > 0 else ''}{pts} pts)"


def fmt_flips(change: Mapping | None) -> str:
    if not change or "fixed" not in change:
        return "–"
    parts = [f"now pass: {', '.join(change['fixed'])}" if change["fixed"] else "",
             f"now fail: {', '.join(change['broke'])}" if change["broke"] else ""]
    return "; ".join(p for p in parts if p) or "none"


def cell_name(report: Mapping) -> str:
    """"computer_use · desktop · full" style name of a prepared report's cell."""
    config = report.get("config") or {}
    model = config.get("model_profile") or config.get("model") or config.get("backend") or "unknown model"
    category = str(report.get("mode") or "browser").capitalize()
    return f"{model} · {category} · {(config.get('harness_profile') or 'full').capitalize()} ({report.get('suite')})"


def change_markdown(rows: Iterable[Mapping], dashboard_url: str | None = None) -> str:
    """The /bench PR comment: one table row per imported report.

    Each row: ``{"cell", "summary", "run_url"}``, with ``summary`` holding
    ``success``, ``tasks``, ``score_range`` and ``change``."""
    rows = list(rows)
    if not rows:
        return ""
    lines = [
        "### Benchmark results vs the previous run of each cell",
        "",
        "| Cell | Passed | Score (95% range) | Change vs previous run | Flipped tasks |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        s = row["summary"]
        cell = f"[{row['cell']}]({row['run_url']})" if row.get("run_url") else row["cell"]
        change = s.get("change")
        flips = fmt_flips(change)
        lines.append(f"| {cell} | {s.get('success')}/{s.get('tasks')} | {fmt_range(s.get('score_range'))} | {fmt_change(change)} | {flips} |")
    lines += [
        "",
        "Improved or Regressed shows only when the 95% range of the paired difference excludes zero; with about "
        "10 tasks that takes about 4 tasks flipping the same way. Smaller moves say No clear change.",
    ]
    if dashboard_url:
        lines.append(f"Dashboard: {dashboard_url} (Leaderboard tab).")
    return "\n".join(lines) + "\n"
