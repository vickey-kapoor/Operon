"""WebArena benchmark runner.

Usage:
    python -m src.benchmarks.webarena [options]

Options:
    --tasks PATH      Path to WebArena tasks JSON (default: benchmarks/webarena_tasks.json)
    --site SITE       Filter by site field (optional)
    --diff DIFF       Filter by difficulty field (optional)
    --steps N         Max steps per task (default: 15)
    --output DIR      Output directory (default: results/webarena_<timestamp>)
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.benchmarks.webarena_eval import evaluate_task
from src.benchmarks.webarena_models import (  # noqa: E501
    WebArenaSummary,
    WebArenaTask,
    WebArenaTaskResult,
)


def _load_tasks(path: Path) -> list[WebArenaTask]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [WebArenaTask.model_validate(t) for t in raw]


def _filter_tasks(
    tasks: list[WebArenaTask],
    site: str | None,
    difficulty: str | None,
) -> list[WebArenaTask]:
    if site:
        tasks = [t for t in tasks if t.site == site]
    if difficulty:
        tasks = [t for t in tasks if t.difficulty == difficulty]
    return tasks


def _extract_final_text(agent_state) -> str:
    """Build the answer-search corpus that the string_match evaluator greps.

    Priority order:
    1. The text payload of the final stop action, when present. This is the
       agent's explicit answer (per the ANSWER ON STOP prompt directive).
    2. The last perception's summary plus all visible-element labels/text,
       as a fallback for tasks where the answer is read off the screen
       rather than committed via stop.text.

    Both are concatenated so a stop.text answer is never lost even if the
    final perception happens to lack the same string.
    """
    if not agent_state:
        return ""
    parts: list[str] = []

    # Pull the last stop action's text first — this is the agent's explicit answer.
    from src.models.policy import ActionType
    if agent_state.action_history:
        last_action = agent_state.action_history[-1].action
        if last_action.action_type is ActionType.STOP and last_action.text:
            parts.append(last_action.text)

    # Then perception fallback.
    if agent_state.observation_history:
        last = agent_state.observation_history[-1]
        if last.summary:
            parts.append(last.summary)
        for el in last.visible_elements:
            for field in (el.label, el.text):
                if field:
                    parts.append(field)

    return " ".join(parts)


def _extract_final_url(loop, run_id: str, agent_state) -> str:
    executor = loop.executor
    if hasattr(executor, "current_url_for_run"):
        try:
            url = asyncio.get_event_loop().run_until_complete(
                executor.current_url_for_run(run_id)
            )
            if url:
                return url
        except Exception:
            pass
    return agent_state.start_url or "" if agent_state else ""


async def _run_all_tasks(
    tasks: list[WebArenaTask],
    max_steps: int,
    output_dir: Path,
    loop,
) -> list[WebArenaTaskResult]:
    results: list[WebArenaTaskResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"  [{task.task_id}] {task.intent[:80]}...")
        started_at = time.perf_counter()

        response = await loop.run_live_benchmark(
            task.intent,
            benchmark_url=task.start_url,
            max_steps=max_steps,
        )

        duration = time.perf_counter() - started_at
        run_id = response.run_id

        agent_state = await loop.run_store.get_run(run_id)

        final_text = _extract_final_text(agent_state)

        # URL extraction — prefer live page URL, fall back to start_url
        executor = loop.executor
        final_url = task.start_url
        if hasattr(executor, "current_url_for_run"):
            try:
                live_url = await executor.current_url_for_run(run_id)
                if live_url:
                    final_url = live_url
            except Exception:
                pass
        elif agent_state and agent_state.start_url:
            final_url = agent_state.start_url

        eval_types_result = evaluate_task(task, final_url, final_text)
        passed = bool(eval_types_result) and all(eval_types_result.values())

        stop_reason = (
            agent_state.stop_reason.value
            if agent_state and agent_state.stop_reason
            else None
        )

        result = WebArenaTaskResult(
            task_id=task.task_id,
            run_id=run_id,
            passed=passed,
            eval_types_result=eval_types_result,
            extracted_url=final_url,
            extracted_text_excerpt=final_text[:500] if final_text else None,
            duration_seconds=round(duration, 3),
            stop_reason=stop_reason,
            site=task.site,
            category=task.category,
            difficulty=task.difficulty,
        )
        results.append(result)

        status_label = "PASS" if passed else "FAIL"
        print(f"    -> {status_label}  ({duration:.1f}s)  {eval_types_result}")

        task_path = output_dir / f"task_{task.task_id}.json"
        task_path.write_text(
            json.dumps(result.model_dump(), indent=2),
            encoding="utf-8",
        )

    return results


def _build_summary(results: list[WebArenaTaskResult]) -> WebArenaSummary:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0.0

    def _tally(group_key: str) -> dict[str, dict[str, int]]:
        tally: dict[str, dict[str, int]] = {}
        for r in results:
            bucket = getattr(r, group_key)
            if bucket not in tally:
                tally[bucket] = {"total": 0, "passed": 0}
            tally[bucket]["total"] += 1
            if r.passed:
                tally[bucket]["passed"] += 1
        return tally

    return WebArenaSummary(
        total=total,
        passed=passed,
        pass_rate=round(pass_rate, 4),
        by_category=_tally("category"),
        by_difficulty=_tally("difficulty"),
        by_site=_tally("site"),
        tasks=results,
    )


def _print_summary(summary: WebArenaSummary) -> None:
    bar = "=" * 52
    print(f"\n{bar}")
    print("WebArena Benchmark Results")
    print(bar)
    print(f"  Tasks run    : {summary.total}")
    print(f"  Passed       : {summary.passed}")
    print(f"  Pass rate    : {summary.pass_rate:.1%}")

    for label, breakdown in (
        ("By difficulty", summary.by_difficulty),
        ("By category", summary.by_category),
        ("By site", summary.by_site),
    ):
        if breakdown:
            print(f"  {label}:")
            for key, counts in sorted(breakdown.items()):
                pct = counts["passed"] / counts["total"] if counts["total"] else 0.0
                print(f"    {key:<22} {counts['passed']}/{counts['total']}  ({pct:.0%})")
    print(bar + "\n")


async def main(args) -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)

    from src.api.routes import get_agent_loop

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    all_tasks = _load_tasks(tasks_path)
    tasks = _filter_tasks(all_tasks, site=args.site, difficulty=args.diff)

    if not tasks:
        print("No tasks match the given filters.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else Path(f"results/webarena_{timestamp}")

    print(f"Running {len(tasks)} WebArena task(s) -> {output_dir}")

    loop = get_agent_loop()
    results = await _run_all_tasks(tasks, max_steps=args.steps, output_dir=output_dir, loop=loop)

    summary = _build_summary(results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary.model_dump(), indent=2),
        encoding="utf-8",
    )

    _print_summary(summary)
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebArena benchmark runner")
    parser.add_argument(
        "--tasks",
        default="benchmarks/webarena_tasks.json",
        help="Path to WebArena tasks JSON",
    )
    parser.add_argument("--site", default=None, help="Filter by site")
    parser.add_argument("--diff", default=None, help="Filter by difficulty")
    parser.add_argument("--steps", type=int, default=15, help="Max steps per task")
    parser.add_argument("--output", default=None, help="Output directory")
    _args = parser.parse_args()
    asyncio.run(main(_args))
