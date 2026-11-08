"""Print the offline eval report: ``python -m evals [--json]``."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict

from evals.harness import RunResult, run_scenario
from evals.scenarios import SCENARIOS


async def _run_all() -> list[tuple[object, RunResult]]:
    results = []
    for factory in SCENARIOS:
        scenario = factory()
        results.append((scenario, await run_scenario(scenario)))
    return results


def main(argv: list[str]) -> int:
    os.environ.setdefault("GEMINI_API_KEY", "offline-eval")
    logging.disable(logging.WARNING)
    results = asyncio.run(_run_all())
    if "--json" in argv:
        rows = [
            {**{k: v for k, v in asdict(r).items() if k != "model_requests"}, "ok": s.accepts(r.outcome), "known_issue": s.known_issue}
            for s, r in results
        ]
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'scenario':30} {'outcome':14} {'ok':4} {'steps':>5} {'pauses':>6} {'model':>5}  note")
    for scenario, result in results:
        ok = "yes" if scenario.accepts(result.outcome) else "NO"
        note = f"known: {scenario.known_issue[:60]}..." if scenario.known_issue else ""
        print(
            f"{result.scenario:30} {result.outcome:14} {ok:4} {result.steps:>5} {result.pauses:>6} "
            f"{result.model_calls:>5}  {note}"
        )
    total = len(results)
    correct = sum(s.accepts(r.outcome) for s, r in results)
    false_success = sum(r.outcome == "false_success" for _, r in results)
    crashes = sum(r.outcome == "crash" for _, r in results)
    print(f"\n{correct}/{total} correct · {false_success} false success · {crashes} crash")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
