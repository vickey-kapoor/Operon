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
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.failure_cause import bench_cause, by_cause

SUBSET_FILE = Path(__file__).with_name("osworld_subset.json")
# OSWorld's own runner waits 20 s before evaluating (lib_run_single.py): Chrome saves preference
# changes to disk with a delay, and file-based checkers read the stale value if we score at once.
EVALUATE_SETTLE_S = 20.0
# The registry profiles (KAN-168, models.yaml) an OSWorld run may use besides Gemini, with the names runs
# carry in chat. The one allow-list: bench_batch.py reads it from here. This file imports no Operon code.
MODEL_LABELS = {"claude-sonnet-5-5": "Claude Sonnet 5.5", "openai-computer-use": "GPT computer use"}
RUN_ID_RE = re.compile(r"Run ([0-9a-f]{8}-[0-9a-f-]{27}|[0-9a-f]{8,}) started")


# The first words of a step prompt file when the backend made no model call (replayed approved calls).
NO_MODEL_CALL = "No model call"

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
    model backend (KAN-168) for the desktop; a Gemini run forces computer_use and an empty profile."""
    env = {**os.environ, **({"OPERON_RUNS_ROOT": runs_root} if runs_root else {}), "OPERON_ENVIRONMENT": "osworld", "OPERON_OSWORLD_URL": server_url,
           "OPERON_AUTO_CONFIRM": "on", "OPERON_TOOLS": "off",
           "PYTHONUNBUFFERED": "1", "OPERON_DESKTOP_PLATFORM": "linux"}
    # Always set both, so a run never inherits the runner's backend settings.
    env["OPERON_DESKTOP_BACKEND"] = "model" if model else "computer_use"
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


def _no_model_call(step: Path, name: str) -> bool:
    """The step's ``<name>_prompt.txt`` says no model call was made (approved calls replayed), so its
    saved answer has, rightly, no usage file."""
    try:
        with (step / f"{name}_prompt.txt").open(encoding="utf-8") as handle:
            return handle.read(len(NO_MODEL_CALL)) == NO_MODEL_CALL
    except OSError:
        return False


def run_usage_gaps(run_id: str | None, runs_root: str | Path | None = None) -> dict[str, int]:
    """Steps with a saved model response (``*_raw.*``) but no ``*_usage.json``, and usage files that
    cannot be read: calls ``run_cost`` could not count (KAN-177). Import-free, like ``run_cost``."""
    gaps = {"steps_without_usage": 0, "unreadable_usage_files": 0}
    if not run_id or runs_root is None:
        return gaps
    run_dir = Path(runs_root) / run_id
    for step in sorted(run_dir.glob("step_*")):
        names = {raw.name.split("_raw.", 1)[0] for raw in step.glob("*_raw.*")}
        if any(not (step / f"{n}_usage.json").exists() and not _no_model_call(step, n) for n in names):
            gaps["steps_without_usage"] += 1
    for path in sorted(run_dir.glob("step_*/*_usage.json")) + [run_dir / "planner_usage.json"]:
        if not path.exists():
            continue
        try:
            ok = isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        except (OSError, ValueError):
            ok = False
        gaps["unreadable_usage_files"] += not ok
    return gaps


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


def run_state(run_id: str | None, runs_root: str | Path | None = None) -> tuple[str | None, str | None]:
    """``(status, stop_reason)`` from ``<runs_root>/<run_id>/state.json``; (None, None) if unreadable.

    Reads the file directly, like ``run_cost``, so this runner needs no Operon import."""
    if not run_id or runs_root is None:
        return None, None
    try:
        data = json.loads((Path(runs_root) / run_id / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    status, reason = data.get("status"), data.get("stop_reason")
    return (str(status) if status else None), (str(reason) if reason else None)


def run_failure(run_id: str | None, runs_root: str | Path | None = None) -> tuple[str | None, str | None]:
    """``(failure_cause, failure_detail)`` saved in the run's state.json (KAN-175); (None, None) if unreadable."""
    if not run_id or runs_root is None:
        return None, None
    try:
        data = json.loads((Path(runs_root) / run_id / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    cause, detail = data.get("failure_cause"), data.get("failure_detail")
    return (str(cause) if cause else None), (str(detail) if detail else None)


def failure_cause(*, passed: bool, bench_error: bool, timed_out: bool, run_status: str | None,
                  stop_reason: str | None, state_cause: str | None) -> str | None:
    """The one cause for a non-passing task (evals.failure_cause.bench_cause, which needs no Operon install)."""
    return bench_cause(passed=passed, bench_error=bench_error, run_status=run_status, stop_reason=stop_reason,
                       state_cause=state_cause, timed_out=timed_out)


def error_result(task: dict[str, str], note: str, error: str, *, seconds: int = 0, run_id: str | None = None,
                 cost_usd: float = 0.0, log: str | None = None) -> dict[str, Any]:
    """The result of a task that raised before it could be scored (score 0)."""
    return {"domain": task["domain"], "id": task["id"], "score": 0.0, "passed": False, "run_id": run_id,
            "cost_usd": round(cost_usd, 4), "seconds": seconds, "note": note, "log": log, "status": "error",
            "stop_reason": None, "run_status": None, "error": error, "failure_cause": "bench_error"}


def run_one(env: Any, task: dict[str, str], config: dict[str, Any], *, max_steps: int, timeout_s: float,
            python: str | None = None, runs_root: str | None = None, settle_s: float = EVALUATE_SETTLE_S,
            logs_dir: str | None = None, model: str = "",
            run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    """One task. An ``Exception`` in reset, Operon or evaluate is recorded as an error result (score 0, the
    cost and run id so far kept); ``BaseException`` (Ctrl+C) still propagates."""
    started = time.monotonic()
    wall_started = time.time()
    phase = "reset"
    run_id, note, error = None, "", None
    out_text, err_text = "", ""
    timed_out, ran, score = False, False, 0.0
    log = None
    try:
        env.reset(task_config=config)
        url = f"http://{env.vm_ip}:{env.server_port}"
        wait_for_desktop(url)
        phase = "operon"
        try:
            ran = True
            done = run(operon_command(task["instruction"], max_steps, python), env=operon_env(url, runs_root, model),
                       capture_output=True, text=True, timeout=timeout_s, check=False)
            out_text, err_text = done.stdout or "", done.stderr or ""
            run_id = parse_run_id(out_text + err_text)
            note = f"operon exit {done.returncode}"
        except subprocess.TimeoutExpired as exc:
            partial = [(x.decode("utf-8", "replace") if isinstance(x, bytes) else x or "") for x in (exc.stdout, exc.stderr)]
            out_text, err_text = partial
            run_id = parse_run_id("".join(partial))
            timed_out = True
            note = f"operon timed out after {timeout_s:.0f}s"
        log = write_task_log(logs_dir, task["id"], note, out_text, err_text)
        run_id = run_id or find_run_since(runs_root, wall_started)
        phase = "evaluate"
        time.sleep(settle_s)  # let the apps flush their state to disk before the checker reads it
        score = float(env.evaluate())  # OSWorld's evaluator decides
    except Exception as exc:  # noqa: BLE001 - one task's failure must not stop the others
        error = f"{type(exc).__name__}: {exc}"[:300]
        note = f"{phase} error: {error}"
        score = 0.0
        if ran and not log:
            log = write_task_log(logs_dir, task["id"], note, out_text, err_text)
    if ran:
        run_id = run_id or find_run_since(runs_root, wall_started)
    run_status, stop_reason = run_state(run_id, runs_root)
    state_cause, state_detail = run_failure(run_id, runs_root)
    passed = score >= 1.0 and not error
    # A bench error gives bench_error, unless the run itself recorded a cause (the model API died inside Operon).
    cause = failure_cause(passed=passed, bench_error=bool(error) and not state_cause, timed_out=timed_out,
                          run_status=run_status, stop_reason=stop_reason, state_cause=state_cause)
    status = "error" if error else "passed" if score >= 1.0 else "timed_out" if timed_out else "failed"
    return {"domain": task["domain"], "id": task["id"], "score": score, "passed": score >= 1.0 and not error,
            "run_id": run_id, "cost_usd": round(run_cost(run_id, runs_root), 4), "seconds": round(time.monotonic() - started),
            "note": note, "log": log, "status": status, "stop_reason": stop_reason, "run_status": run_status,
            "error": error or (None if passed else state_detail), "failure_cause": cause,
            **run_usage_gaps(run_id, runs_root)}


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
    errors = sum(1 for r in results if r.get("status") == "error")
    timed_out = sum(1 for r in results if r.get("status") == "timed_out")
    return {"tasks": n, "passed": passed, "errors": errors, "timed_out": timed_out,
            "by_cause": by_cause(r.get("failure_cause") for r in results if not r["passed"]),
            "success_rate": round(passed / n, 3) if n else 0.0,
            "cost_usd": round(sum(r["cost_usd"] for r in results), 4),
            # KAN-177: calls run_cost could not count; evals/dashboard.py flags a run with any.
            "steps_without_usage": sum(r.get("steps_without_usage", 0) for r in results),
            "unreadable_usage_files": sum(r.get("unreadable_usage_files", 0) for r in results),
            "by_domain": {d: f"{p}/{t}" for d, (p, t) in sorted(by_domain.items())}}


def job_minutes(n_tasks: int, task_timeout_s: float = 1500.0) -> int:
    """Minutes the CI job needs for ``n_tasks``: 30 for setup.sh plus, per task, its timeout and 600 s for
    the VM reset and boot, wait_for_desktop (150 s), the 20 s settle, evaluate and the video render."""
    if n_tasks <= 0:
        return 30
    return 30 + n_tasks * math.ceil((task_timeout_s + 600) / 60)


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
    parser.add_argument("--job-minutes", action="store_true",
                        help="print the CI job timeout in minutes for the chosen tasks and exit (free)")
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
    if args.job_minutes:
        print(job_minutes(len(tasks), args.task_timeout))
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
            started = time.monotonic()
            try:
                result = run_one(env, task, task_config(repo, task), max_steps=args.max_steps,
                                 timeout_s=args.task_timeout, python=args.operon_python,
                                 runs_root=str(out / "osworld-runs"),
                                 logs_dir=str(out / "osworld-logs"), model=args.model, run=run)
            except Exception as exc:  # noqa: BLE001 - keep going with the next task
                error = f"{type(exc).__name__}: {exc}"[:300]
                result = error_result(task, f"setup error: {error}", error, seconds=round(time.monotonic() - started))
            if result.get("error"):
                print(f"::warning::task {task['id']} errored: {result['error']}", flush=True)
            spent += result["cost_usd"]
            results.append(result)
            save()  # after every task: a later failure or Ctrl+C must not lose finished scores
            label = ("ERROR" if result.get("status") == "error" else "TIMEOUT" if result.get("status") == "timed_out"
                     else "PASS" if result["passed"] else "fail")
            print(f"{label} {task['domain']} {task['id'][:8]} ${result['cost_usd']:.3f} {result['seconds']}s")
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
