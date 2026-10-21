"""Live benchmark: real web tasks against a running Operon API, graded from the final screen.

The offline eval (``python -m tests.eval``) scripts the model. This runs the
real agent on real pages and grades each run the same way: a run's own claim
("succeeded") is compared with what a judge model sees on the final screenshot.
It needs a server started with a real ``GEMINI_API_KEY``::

    .venv/bin/python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080
    .venv/bin/python -m tests.eval.live_bench [--suite smoke|full] [--only NAME] [--category CAT]

Outcomes, per task:

- ``success``: the run claimed success and the screen shows the goal.
- ``false_success``: the run claimed success but the screen does not show it.
- ``missed``: the run ended without claiming success, but the goal is on screen.
- ``failure``: no success claim, and the goal is not on screen.
- ``timeout``: the run was still going when the per-task timeout hit.
- ``error``: the API or the run itself broke.

Pauses for confirmation are approved automatically (``--no-approve`` turns
that off) and counted. Every run's results are saved under
``.var/bench/`` and the report compares them with the previous saved run.
The judge model is ``OPERON_BENCH_JUDGE_MODEL`` (default ``gemini-3.5-flash``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from operon.core.paths import runs_dir
from tests.eval.live_tasks import TASKS, LiveTask

TERMINAL = {"succeeded", "failed", "cancelled"}
OUTCOMES = ("success", "false_success", "missed", "failure", "timeout", "error")
BENCH_DIR = Path(os.getenv("OPERON_BENCH_DIR", ".var/bench"))

Judge = Callable[[LiveTask, str], Awaitable["Verdict"]]


@dataclass(frozen=True)
class Verdict:
    goal_visible: bool | None
    reason: str


@dataclass
class TaskResult:
    task: str
    category: str
    run_id: str | None
    status: str
    outcome: str
    stop_reason: str
    steps: int
    pauses: int
    seconds: int
    cu_steps: int
    other_steps: int
    judge_reason: str
    final_screenshot: str


def classify(status: str, goal_visible: bool | None) -> str:
    """Grade a run: its claim (``status``) against what the screen shows."""
    if status == "error":
        return "error"
    if status not in TERMINAL:
        return "timeout"
    if goal_visible is None:
        return "error"
    claimed = status == "succeeded"
    if claimed:
        return "success" if goal_visible else "false_success"
    return "missed" if goal_visible else "failure"


JUDGE_PROMPT = """You grade a browser agent. Look only at the screenshot.

The user asked the agent: {intent}
The task is complete only if: {check}

Does the screenshot show the task complete? Judge strictly from what is visible;
do not assume anything happened off screen.
Reply with JSON only: {{"goal_visible": true or false, "reason": "<one sentence>"}}"""


def parse_verdict(text: str) -> Verdict:
    """Read the judge's JSON reply; anything unreadable is no verdict."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        return Verdict(None, f"unreadable judge reply: {text[:120]!r}")
    visible = data.get("goal_visible") if isinstance(data, dict) else None
    if not isinstance(visible, bool):
        return Verdict(None, f"judge reply had no goal_visible: {text[:120]!r}")
    return Verdict(visible, str(data.get("reason", "")).strip())


def gemini_judge(model: str) -> Judge:
    from operon.clients.gemini import GeminiHttpClient

    client = GeminiHttpClient(api_key=os.getenv("GEMINI_API_KEY"), model=model, timeout_seconds=60.0)

    async def judge(task: LiveTask, screenshot: str) -> Verdict:
        prompt = JUDGE_PROMPT.format(intent=task.intent, check=task.check)
        try:
            return parse_verdict(await client.generate_perception(prompt, screenshot))
        except Exception as exc:  # the judge failing must not stop the benchmark
            return Verdict(None, f"judge call failed: {exc}")

    return judge


def _run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def _backends_used(run_id: str) -> tuple[int, int]:
    steps = sorted(_run_dir(run_id).glob("step_*"))
    cu = sum((step / "computer_use_raw.json").exists() for step in steps)
    return cu, len(steps) - cu


def _final_screenshot(run_id: str) -> str | None:
    frames = sorted(_run_dir(run_id).glob("step_*/*.png"), key=lambda p: p.stat().st_mtime)
    return str(frames[-1]) if frames else None


def _stop_reason(run_id: str) -> str:
    try:
        state = json.loads((_run_dir(run_id) / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "-"
    return state.get("stop_reason") or "-"


async def run_task(
    client: httpx.AsyncClient,
    task: LiveTask,
    judge: Judge,
    *,
    timeout_s: float,
    approve_pauses: int,
) -> TaskResult:
    started = time.monotonic()
    run_id, status, steps, pauses = None, "error", 0, 0
    try:
        created = await client.post(
            "/run-task",
            json={"intent": task.intent, "start_url": task.start_url, "max_steps": task.max_steps},
        )
        created.raise_for_status()
        run_id, status = created.json()["run_id"], created.json()["status"]
        while time.monotonic() - started < timeout_s:
            await asyncio.sleep(2)
            response = await client.get(f"/run/{run_id}")
            response.raise_for_status()
            status, steps = response.json()["status"], response.json()["step_count"]
            if status == "waiting_for_user" and pauses < approve_pauses:
                pauses += 1
                (await client.post("/resume", json={"run_id": run_id})).raise_for_status()
                continue
            if status in TERMINAL or status == "waiting_for_user":
                break
        if status not in TERMINAL and status != "waiting_for_user":
            await client.post("/stop", json={"run_id": run_id})
    except httpx.HTTPError as exc:
        status = "error"
        print(f"  {task.name}: API error: {exc}", file=sys.stderr)

    screenshot = _final_screenshot(run_id) if run_id else None
    verdict = await judge(task, screenshot) if screenshot else Verdict(None, "no screenshot")
    # A run paused for the user did not claim success; grade it as ended.
    graded_status = "failed" if status == "waiting_for_user" else status
    cu, other = _backends_used(run_id) if run_id else (0, 0)
    return TaskResult(
        task=task.name,
        category=task.category,
        run_id=run_id,
        status=status,
        outcome=classify(graded_status, verdict.goal_visible),
        stop_reason=_stop_reason(run_id) if run_id else "-",
        steps=steps,
        pauses=pauses,
        seconds=round(time.monotonic() - started),
        cu_steps=cu,
        other_steps=other,
        judge_reason=verdict.reason,
        final_screenshot=screenshot or "-",
    )


def summarize(results: list[TaskResult]) -> dict:
    counts = {outcome: sum(r.outcome == outcome for r in results) for outcome in OUTCOMES}
    total = len(results)
    steps = sum(r.steps for r in results)
    return {
        "tasks": total,
        **counts,
        "success_rate": round(counts["success"] / total, 3) if total else 0.0,
        "mean_steps": round(steps / total, 1) if total else 0.0,
        "mean_seconds": round(sum(r.seconds for r in results) / total, 1) if total else 0.0,
        "seconds_per_step": round(sum(r.seconds for r in results) / steps, 1) if steps else 0.0,
        "fallback_steps": sum(r.other_steps for r in results),
    }


def _previous_report(bench_dir: Path, suite: str) -> dict | None:
    reports = sorted(bench_dir.glob(f"{suite}-*.json"))
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def format_report(results: list[TaskResult], summary: dict, previous: dict | None) -> str:
    before = {r["task"]: r["outcome"] for r in previous["results"]} if previous else {}
    lines = [f"{'task':22} {'category':11} {'outcome':14} {'steps':>5} {'secs':>5}  note"]
    for r in results:
        change = ""
        if r.task in before and before[r.task] != r.outcome:
            change = f"(was {before[r.task]}) "
        note = change + (f"paused x{r.pauses} " if r.pauses else "") + (f"fallback x{r.other_steps} " if r.other_steps else "")
        if r.outcome != "success":
            note += r.judge_reason
        lines.append(f"{r.task:22} {r.category:11} {r.outcome:14} {r.steps:>5} {r.seconds:>5}  {note.strip()}")
    lines.append("")
    lines.append(
        f"{summary['success']}/{summary['tasks']} success ({summary['success_rate']:.0%}) · "
        f"{summary['false_success']} false success · {summary['missed']} missed · "
        f"{summary['failure']} failure · {summary['timeout']} timeout · {summary['error']} error"
    )
    lines.append(
        f"mean {summary['mean_steps']} steps, {summary['mean_seconds']} s per task, "
        f"{summary['seconds_per_step']} s per step · {summary['fallback_steps']} fallback steps"
    )
    if previous:
        prev = previous["summary"]
        lines.append(
            f"previous ({previous['started_at']}): {prev['success']}/{prev['tasks']} success "
            f"({prev['success_rate']:.0%}), {prev['false_success']} false success"
        )
    return "\n".join(lines)


def select_tasks(suite: str, only: list[str] | None, category: list[str] | None) -> list[LiveTask]:
    tasks = [t for t in TASKS if suite == "full" or suite in t.tags]
    if only:
        tasks = [t for t in tasks if t.name in only]
    if category:
        tasks = [t for t in tasks if t.category in category]
    return tasks


async def run_bench(args: argparse.Namespace, judge: Judge) -> int:
    tasks = select_tasks(args.suite, args.only, args.category)
    if not tasks:
        print("No tasks match.", file=sys.stderr)
        return 2
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[TaskResult] = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=300.0) as client:
        (await client.get("/health")).raise_for_status()
        for index, task in enumerate(tasks, 1):
            result = await run_task(
                client, task, judge,
                timeout_s=args.timeout,
                approve_pauses=0 if args.no_approve else 2,
            )
            results.append(result)
            if not args.json:
                print(f"[{index}/{len(tasks)}] {result.task}: {result.outcome} ({result.steps} steps, {result.seconds}s)", flush=True)

    summary = summarize(results)
    bench_dir = Path(args.out_dir)
    previous = _previous_report(bench_dir, args.suite) if not (args.only or args.category) else None
    report = {
        "suite": args.suite,
        "started_at": started_at,
        "browser_model": os.getenv("OPERON_BROWSER_MODEL", "server default"),
        "judge_model": args.judge_model,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    if args.only or args.category:
        name = f"partial-{started_at}.json"
    else:
        name = f"{args.suite}-{started_at}.json"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print()
        print(format_report(results, summary, previous))
        print(f"\nsaved {bench_dir / name}")
    return 0


def build_parser(default_suite: str = "full") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=("smoke", "full"), default=default_suite)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--only", action="append", help="run only the named task(s)")
    parser.add_argument("--category", action="append", help="run only tasks in these categories")
    parser.add_argument("--timeout", type=float, default=420.0, help="seconds per task")
    parser.add_argument("--no-approve", action="store_true", help="do not approve pauses for confirmation")
    parser.add_argument("--judge-model", default=os.getenv("OPERON_BENCH_JUDGE_MODEL", "gemini-3.5-flash"))
    parser.add_argument("--out-dir", default=str(BENCH_DIR))
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str], default_suite: str = "full") -> int:
    args = build_parser(default_suite).parse_args(argv)
    if args.list:
        for task in select_tasks(args.suite, args.only, args.category):
            print(f"{task.name:22} {task.category:11} {task.intent}")
        return 0
    return asyncio.run(run_bench(args, gemini_judge(args.judge_model)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
