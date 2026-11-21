"""OSWorld-Verified subset (KAN-169): Operon on OSWorld's own tasks, scored by OSWorld.

    python -m evals.osworld --list
    python -m evals.osworld --osworld-repo ~/OSWorld [--limit 5] [--max-cost-usd 8]

Per task: OSWorld's ``DesktopEnv`` boots the Ubuntu task VM and sets the task up; Operon
(``operon run --desktop --env osworld``, thin harness) drives that VM from screenshots;
then OSWorld's own evaluator scores the end state. Operon's done-check is not the score.
The task list is fixed in ``evals/osworld_subset.json`` (chosen before any run).
Costs real credits: it is started by hand and needs Vickey's OK.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SUBSET_FILE = Path(__file__).with_name("osworld_subset.json")
# OSWorld's own runner waits 20 s before evaluating (lib_run_single.py): Chrome saves preference
# changes to disk with a delay, and file-based checkers read the stale value if we score at once.
EVALUATE_SETTLE_S = 20.0
# The registry profiles (KAN-168, models.yaml) an OSWorld run may use besides Gemini, with the names runs
# carry in chat. The one allow-list: bench_batch.py reads it from here. This file imports no Operon code.
MODEL_LABELS = {"claude-sonnet-5-5": "Claude Sonnet 5.5", "openai-computer-use": "GPT computer use"}
RUN_ID_RE = re.compile(r"Run ([0-9a-f]{8}-[0-9a-f-]{27}|[0-9a-f]{8,}) started")


def load_subset(path: Path = SUBSET_FILE) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def parse_run_id(output: str) -> str | None:
    match = RUN_ID_RE.search(output)
    return match.group(1) if match else None


def task_config(repo: Path, task: dict[str, str]) -> dict[str, Any]:
    path = repo / "evaluation_examples" / "examples" / task["domain"] / f"{task['id']}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def operon_command(instruction: str, max_steps: int, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "operon.cli", "run", "--desktop", "--env", "osworld",
            "--mode", "auto", "--max-steps", str(max_steps), instruction]


def operon_env(server_url: str, runs_root: str | None = None, model: str = "") -> dict[str, str]:
    """The environment of one ``operon run``. ``model`` (a registry profile; empty is Gemini) selects the
    model backend (KAN-168) for the desktop; Gemini runs leave the backend settings as they are."""
    env = {**os.environ, **({"OPERON_RUNS_ROOT": runs_root} if runs_root else {}), "OPERON_ENVIRONMENT": "osworld", "OPERON_OSWORLD_URL": server_url,
           "OPERON_AUTO_CONFIRM": "on", "OPERON_TOOLS": "off",
           "PYTHONUNBUFFERED": "1", "OPERON_DESKTOP_PLATFORM": "linux"}
    if model:
        env["OPERON_DESKTOP_BACKEND"] = "model"
        env["OPERON_MODEL_PROFILE"] = model
    return env


def run_cost(run_id: str | None, runs_root: str | Path | None = None) -> float:
    """USD of the model calls a run saved (``step_*/*_usage.json``, ``planner_usage.json``).

    Reads the files directly so this runner needs no Operon import: OSWorld's own
    dependencies (torch pins) and Operon (Python 3.14) may live in different Pythons."""
    if not run_id or runs_root is None:
        return 0.0
    run_dir = Path(runs_root) / run_id
    files = sorted(run_dir.glob("step_*/*_usage.json")) + [run_dir / "planner_usage.json"]
    total = 0.0
    for path in files:
        try:
            total += float(json.loads(path.read_text(encoding="utf-8")).get("estimated_cost_usd") or 0.0)
        except (OSError, ValueError, AttributeError):
            continue
    return total


def find_run_since(runs_root: str | Path | None, since: float) -> str | None:
    """The newest run folder created at or after ``since`` (epoch seconds): the run whose id
    never reached us, for example because the task timed out before the start line was read."""
    if runs_root is None or not Path(runs_root).is_dir():
        return None
    runs = [p for p in Path(runs_root).iterdir() if p.is_dir() and p.stat().st_mtime >= since]
    return max(runs, key=lambda p: p.stat().st_mtime).name if runs else None


def wait_for_desktop(url: str, *, min_bytes: int = 200_000, timeout_s: float = 150.0) -> bool:
    """Block until the VM's screenshot is a drawn desktop: the first frames after boot are a
    ~9 KB blank screen. Returns False on timeout (the task still runs)."""
    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/screenshot", timeout=15, trust_env=False)
            if response.status_code == 200 and len(response.content) >= min_bytes:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def write_task_log(logs_dir: str | None, task_id: str, header: str, stdout: str, stderr: str) -> str | None:
    """Save one task's Operon output to ``<logs_dir>/<task id>.log``; None if unset or unwritable."""
    if not logs_dir:
        return None
    path = Path(logs_dir) / f"{task_id}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = f"{header}\n\n=== stdout ===\n{stdout}\n\n=== stderr ===\n{stderr}\n"
        path.write_text(body, encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: could not write {path}: {exc}", file=sys.stderr)
        return None
    return str(path)


def run_one(env: Any, task: dict[str, str], config: dict[str, Any], *, max_steps: int, timeout_s: float,
            python: str | None = None, runs_root: str | None = None, settle_s: float = EVALUATE_SETTLE_S,
            logs_dir: str | None = None, model: str = "",
            run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    started = time.monotonic()
    wall_started = time.time()
    env.reset(task_config=config)
    url = f"http://{env.vm_ip}:{env.server_port}"
    wait_for_desktop(url)
    run_id, note = None, ""
    out_text, err_text = "", ""
    try:
        done = run(operon_command(task["instruction"], max_steps, python), env=operon_env(url, runs_root, model), capture_output=True,
                   text=True, timeout=timeout_s, check=False)
        out_text, err_text = done.stdout or "", done.stderr or ""
        run_id = parse_run_id(out_text + err_text)
        note = f"operon exit {done.returncode}"
    except subprocess.TimeoutExpired as exc:
        partial = [(x.decode("utf-8", "replace") if isinstance(x, bytes) else x or "") for x in (exc.stdout, exc.stderr)]
        out_text, err_text = partial
        run_id = parse_run_id("".join(partial))
        note = f"operon timed out after {timeout_s:.0f}s"
    log = write_task_log(logs_dir, task["id"], note, out_text, err_text)
    run_id = run_id or find_run_since(runs_root, wall_started)
    time.sleep(settle_s)  # let the apps flush their state to disk before the checker reads it
    score = float(env.evaluate())  # OSWorld's evaluator decides
    return {"domain": task["domain"], "id": task["id"], "score": score, "passed": score >= 1.0,
            "run_id": run_id, "cost_usd": round(run_cost(run_id, runs_root), 4), "seconds": round(time.monotonic() - started),
            "note": note, "log": log}


def save_task_videos(results: list[dict[str, Any]], out: Path, python: str,
                     run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
    """One captioned MP4 per finished task (KAN-185): ``<out>/videos/<id prefix>-<pass|fail>.mp4``.

    Rendered by ``python -m evals.run_video`` under ``python`` (the venv that runs ``operon run``,
    which has OpenCV), since this script's own venv may not. Never raises."""
    runs_root = out / "osworld-runs"
    repo_root = Path(__file__).resolve().parent.parent
    made = 0
    for r in results:
        short = str(r.get("id", ""))[:8]
        try:
            run_dir = runs_root / str(r.get("run_id") or "")
            if not r.get("run_id") or not run_dir.is_dir():
                continue
            verdict = "pass" if r.get("passed") else "fail"
            (out / "videos").mkdir(parents=True, exist_ok=True)
            target = out / "videos" / f"{short}-{verdict}.mp4"
            cmd = [python, "-m", "evals.run_video", "--run", str(run_dir.resolve()), "--out", str(target.resolve()),
                   "--task", f"{r.get('domain', '')}: {short}", "--verdict", verdict]
            done = run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=300)
            if done.returncode == 0:
                made += 1
                continue
            why = (done.stderr or done.stdout or "").strip()[-200:]
        except Exception as exc:  # noqa: BLE001
            why = str(exc)
        try:
            target.unlink(missing_ok=True)  # never leave a truncated MP4 to be uploaded
        except Exception:  # noqa: BLE001
            pass
        print(f"::warning::no video for {short}: {why}", flush=True)
    return made


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_domain: dict[str, list[int]] = {}
    for r in results:
        row = by_domain.setdefault(r["domain"], [0, 0])
        row[0] += int(r["passed"])
        row[1] += 1
    return {"tasks": n, "passed": passed, "success_rate": round(passed / n, 3) if n else 0.0,
            "cost_usd": round(sum(r["cost_usd"] for r in results), 4),
            "by_domain": {d: f"{p}/{t}" for d, (p, t) in sorted(by_domain.items())}}


def make_env(repo: Path):  # pragma: no cover - needs OSWorld and a VM
    sys.path.insert(0, str(repo))
    from desktop_env.desktop_env import DesktopEnv

    return DesktopEnv(provider_name="docker", action_space="pyautogui", screen_size=(1920, 1080),
                      headless=True, os_type="Ubuntu", require_a11y_tree=False)


def main(argv: list[str] | None = None, *, env_factory: Callable[[Path], Any] = make_env,
         run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
    parser = argparse.ArgumentParser(prog="evals.osworld", description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="print the fixed task list and exit (free)")
    parser.add_argument("--osworld-repo", default=os.environ.get("OSWORLD_REPO", ""), help="path to an OSWorld checkout")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks (the pilot uses 5)")
    parser.add_argument("--only", default="", help="run only the task whose id starts with this")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--task-timeout", type=float, default=1500.0, help="seconds per task")
    parser.add_argument("--max-cost-usd", type=float, default=8.0, help="stop starting tasks past this spend")
    parser.add_argument("--operon-python", default=sys.executable,
                        help="Python that has Operon installed (3.14); default: this one")
    parser.add_argument("--model", default="", help="registry model profile (" + ", ".join(MODEL_LABELS)
                        + "); empty = Gemini. Spends that provider's credits: needs Vickey's OK")
    parser.add_argument("--videos", action="store_true", help="write one MP4 per task under <out>/videos")
    parser.add_argument("--out", default=".var/bench")
    args = parser.parse_args(argv)

    tasks = load_subset()
    if args.only:
        tasks = [t for t in tasks if t["id"].startswith(args.only)]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.list:
        for t in tasks:
            print(f"{t['domain']:<20} {t['id']}  {t['instruction'][:80]}")
        return 0
    if args.model and args.model not in MODEL_LABELS:
        parser.error(f"--model must be one of: {', '.join(MODEL_LABELS)} (or empty for Gemini)")
    if not args.osworld_repo:
        parser.error("--osworld-repo (or $OSWORLD_REPO) is required to run")
    repo = Path(args.osworld_repo).expanduser()
    env = env_factory(repo)
    results: list[dict[str, Any]] = []
    spent = 0.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"osworld-{time.strftime('%Y%m%dT%H%M%S')}.json"

    def save() -> dict[str, Any]:
        report = {"benchmark": "osworld-verified-subset", "harness": "thin", "auto_confirm": True,
                  "model": args.model, "label": f"{MODEL_LABELS.get(args.model, 'Gemini')} · OSWorld",
                  "summary": summarize(results), "results": results}
        target.write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report

    try:
        for task in tasks:
            if args.max_cost_usd and spent >= args.max_cost_usd:
                print(f"cost cap ${args.max_cost_usd} reached; stopping before {task['id']}")
                break
            result = run_one(env, task, task_config(repo, task), max_steps=args.max_steps,
                             timeout_s=args.task_timeout, python=args.operon_python,
                             runs_root=str(out / "osworld-runs"),
                             logs_dir=str(out / "osworld-logs"), model=args.model, run=run)
            spent += result["cost_usd"]
            results.append(result)
            save()  # after every task: a later failure or Ctrl+C must not lose finished scores
            print(f"{'PASS' if result['passed'] else 'fail'} {task['domain']} {task['id'][:8]} "
                  f"${result['cost_usd']:.3f} {result['seconds']}s")
    finally:
        save()
        if args.videos:
            save_task_videos(results, out, args.operon_python)
        close = getattr(env, "close", None)
        if close:
            close()
    print(json.dumps(summarize(results), indent=1), f"\nreport: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
