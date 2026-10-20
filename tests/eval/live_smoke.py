"""Live smoke run: a few real browser tasks against a running Operon API.

The offline eval (``python -m tests.eval``) scripts the model; this checks that
the same behaviors hold with the real Gemini Computer Use model on real pages.
It needs a server started with a real ``GEMINI_API_KEY``::

    .venv/bin/python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080
    .venv/bin/python -m tests.eval.live_smoke [--only NAME] [--base-url URL]

Each row shows the run's claim (status, stop reason), its cost (steps, seconds),
which backend served each step (``other`` > 0 means it fell back from Computer
Use), and the final screenshot, so the claim can be checked against the screen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from operon.core.paths import runs_dir

TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class LiveTask:
    name: str
    intent: str
    start_url: str
    covers: str
    max_steps: int = 15


TASKS = [
    LiveTask(
        "click_link",
        "Click the 'Learn more' link on this page",
        "https://example.com",
        "single click, then a text-only 'done' reply",
    ),
    LiveTask(
        "search_enter",
        "Search Wikipedia for Alan Turing and open his article",
        "https://en.wikipedia.org/wiki/Main_Page",
        "type_text_at with Enter",
    ),
    LiveTask(
        "multi_field_form",
        "Fill in the customer name Ada Lovelace and the email ada@example.com, then submit the order",
        "https://httpbin.org/forms/post",
        "two fields typed in a row, then submit",
    ),
    LiveTask(
        "scroll_down",
        "Scroll down to the References section of this article",
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "scroll_document / scroll_at direction",
    ),
    LiveTask(
        "go_back",
        "Open the 'Talk' tab, then go back to the article",
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "go_back hotkey",
    ),
]


def _run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def _backends_used(run_id: str) -> str:
    steps = sorted(_run_dir(run_id).glob("step_*"))
    cu = sum((step / "computer_use_raw.json").exists() for step in steps)
    other = len(steps) - cu
    return f"cu={cu} other={other}" if steps else "no steps"


def _final_screenshot(run_id: str) -> str:
    frames = sorted(_run_dir(run_id).glob("step_*/*.png"), key=lambda p: p.stat().st_mtime)
    return str(frames[-1]) if frames else "-"


def _stop_reason(run_id: str) -> str:
    state = _run_dir(run_id) / "state.json"
    try:
        return json.loads(state.read_text(encoding="utf-8")).get("stop_reason") or "-"
    except (OSError, ValueError):
        return "-"


def run_task(client: httpx.Client, task: LiveTask, timeout_s: float) -> dict:
    started = time.monotonic()
    created = client.post(
        "/run-task",
        json={"intent": task.intent, "start_url": task.start_url, "max_steps": task.max_steps},
    )
    created.raise_for_status()
    run_id = created.json()["run_id"]
    status, steps = created.json()["status"], 0
    while time.monotonic() - started < timeout_s:
        time.sleep(2)
        response = client.get(f"/run/{run_id}")
        response.raise_for_status()
        status, steps = response.json()["status"], response.json()["step_count"]
        if status in TERMINAL or status == "waiting_for_user":
            break
    return {
        "task": task.name,
        "covers": task.covers,
        "run_id": run_id,
        "status": status,
        "stop_reason": _stop_reason(run_id),
        "steps": steps,
        "seconds": round(time.monotonic() - started),
        "backends": _backends_used(run_id),
        "final_screenshot": _final_screenshot(run_id),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--only", action="append", help="run only the named task(s)")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per task")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    tasks = [t for t in TASKS if not args.only or t.name in args.only]
    results = []
    with httpx.Client(base_url=args.base_url, timeout=300.0) as client:
        client.get("/health").raise_for_status()
        for task in tasks:
            result = run_task(client, task, args.timeout)
            results.append(result)
            if not args.json:
                print(
                    f"{result['task']:18} {result['status']:17} {result['stop_reason']:32} "
                    f"steps={result['steps']:<3} {result['seconds']:>4}s  {result['backends']}"
                )
                print(f"{'':18} screenshot: {result['final_screenshot']}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
