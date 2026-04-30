"""In-memory benchmark suite runner — runs WebArena/Mind2Web/Operon tasks sequentially."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.models.common import RunStatus, RunTaskRequest, StepRequest

logger = logging.getLogger(__name__)

_SUITES: dict[str, "SuiteState"] = {}

# Global lock: only one browser task may execute at a time across all suites.
# Concurrent suites share the same NativeBrowserExecutor; _close_other_sessions
# would kill competing runs' browsers without this serialisation.
_BROWSER_TASK_LOCK: asyncio.Lock | None = None


def _get_browser_lock() -> asyncio.Lock:
    global _BROWSER_TASK_LOCK
    if _BROWSER_TASK_LOCK is None:
        _BROWSER_TASK_LOCK = asyncio.Lock()
    return _BROWSER_TASK_LOCK

_DATASET_PATHS = [
    Path("benchmarks/datasets/remaining_tasks.json"),  # temp: m2w_008+ and WebArena only
]

_SUCCESS_STOP_REASONS = {"form_submitted_success", "task_completed", "stop_before_send"}

# Map stop_reason values to infrastructure failure tags
_INFRA_FAILURE_MAP: dict[str, str] = {
    "max_retries_exceeded": "coordinate_drift",
    "repeated_loop_detected": "perception_blindness",
    "target_reresolution_failed": "coordinate_drift",
    "target_reresolution_ambiguous": "coordinate_drift",
    "stale_target_before_action": "viewport_clipping",
    "target_shifted_before_action": "viewport_clipping",
    "target_lost_before_action": "viewport_clipping",
    "perception_failure": "perception_blindness",
    "no_progress": "perception_blindness",
}


def _load_all_tasks() -> list[dict[str, Any]]:
    all_tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _DATASET_PATHS:
        if not path.exists():
            continue
        for t in json.loads(path.read_text(encoding="utf-8")):
            if t["task_id"] not in seen:
                seen.add(t["task_id"])
                all_tasks.append(t)
    return all_tasks


def _load_tasks_filtered(difficulty: str, source: str = "all") -> list[dict[str, Any]]:
    tasks = _load_all_tasks()
    if difficulty != "all":
        tasks = [t for t in tasks if t["difficulty"] == difficulty]
    if source != "all":
        tasks = [t for t in tasks if t.get("source", "operon") == source]
    return tasks


def _classify_infra_tag(stop_reason: str | None, failure_category: str | None) -> str | None:
    if stop_reason:
        for key, tag in _INFRA_FAILURE_MAP.items():
            if key in stop_reason.lower():
                return tag
    if failure_category and "perception" in failure_category.lower():
        return "perception_blindness"
    return None


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    category: str
    site: str
    intent: str
    start_url: str
    source: str = "operon"
    optimal_steps: int = 5
    status: str = "pending"
    run_id: str | None = None
    stop_reason: str | None = None
    step_count: int = 0
    step_efficiency: float | None = None
    infra_tag: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class SuiteState:
    suite_id: str
    difficulty: str
    tasks: list[TaskResult]
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cancelled: bool = False

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status in ("passed", "failed"))

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == "passed")

    @property
    def pass_rate(self) -> float:
        done = self.completed_count
        return round(self.passed_count / done, 3) if done > 0 else 0.0

    @property
    def avg_step_efficiency(self) -> float | None:
        efficiencies = [t.step_efficiency for t in self.tasks if t.step_efficiency is not None]
        if not efficiencies:
            return None
        return round(sum(efficiencies) / len(efficiencies), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "difficulty": self.difficulty,
            "status": self.status,
            "total": self.total,
            "completed": self.completed_count,
            "passed": self.passed_count,
            "pass_rate": self.pass_rate,
            "avg_step_efficiency": self.avg_step_efficiency,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "source": t.source,
                    "difficulty": t.difficulty,
                    "category": t.category,
                    "site": t.site,
                    "intent": t.intent,
                    "start_url": t.start_url,
                    "optimal_steps": t.optimal_steps,
                    "status": t.status,
                    "run_id": t.run_id,
                    "stop_reason": t.stop_reason,
                    "step_count": t.step_count,
                    "step_efficiency": t.step_efficiency,
                    "infra_tag": t.infra_tag,
                    "duration_seconds": t.duration_seconds,
                    "error": t.error,
                }
                for t in self.tasks
            ],
        }


def _make_task_result(t: dict[str, Any]) -> TaskResult:
    return TaskResult(
        task_id=t["task_id"],
        difficulty=t["difficulty"],
        category=t["category"],
        site=t["site"],
        intent=t["intent"],
        start_url=t["start_url"],
        source=t.get("source", "operon"),
        optimal_steps=t.get("optimal_steps", 5),
    )


def create_suite(difficulty: str, source: str = "all") -> SuiteState:
    suite_id = uuid.uuid4().hex[:8]
    tasks_data = _load_tasks_filtered(difficulty, source)
    tasks = [_make_task_result(t) for t in tasks_data]
    suite = SuiteState(suite_id=suite_id, difficulty=difficulty, tasks=tasks)
    _SUITES[suite_id] = suite
    return suite


def get_suite(suite_id: str) -> SuiteState | None:
    return _SUITES.get(suite_id)


def get_all_tasks(difficulty: str = "all", source: str = "all") -> list[dict[str, Any]]:
    """Return all tasks from all datasets for UI initialisation."""
    return _load_tasks_filtered(difficulty, source)


async def run_suite_background(suite_id: str, max_steps: int, get_loop_fn: Callable, headless: bool = False) -> None:
    """Run all tasks in the suite sequentially, updating state in place."""
    suite = _SUITES.get(suite_id)
    if suite is None:
        return
    suite.status = "running"

    for task in suite.tasks:
        if suite.cancelled:
            task.status = "failed"
            task.error = "cancelled"
            continue
        task.status = "running"
        t_start = time.time()
        try:
            async with _get_browser_lock():
                loop = get_loop_fn()
                req = RunTaskRequest(intent=task.intent, start_url=task.start_url, headless=headless)
                init_resp = await loop.start_run(req)
                task.run_id = init_resp.run_id

                last_resp = init_resp
                for _ in range(max_steps):
                    last_resp = await loop.step_run(StepRequest(run_id=init_resp.run_id))
                    task.step_count = last_resp.step_count or 0
                    if last_resp.status not in (RunStatus.RUNNING, RunStatus.WAITING_FOR_USER):
                        break

                raw_stop = getattr(last_resp, "stop_reason", None)
                task.stop_reason = raw_stop.value if hasattr(raw_stop, "value") else (str(raw_stop) if raw_stop else None)

                succeeded = last_resp.status == RunStatus.SUCCEEDED
                if not succeeded and task.stop_reason:
                    succeeded = task.stop_reason in _SUCCESS_STOP_REASONS
                task.status = "passed" if succeeded else "failed"

                if task.step_count > 0:
                    task.step_efficiency = round(min(task.optimal_steps / task.step_count, 1.0), 2)

                failure_cat = getattr(last_resp, "failure_category", None)
                failure_cat_str = failure_cat.value if hasattr(failure_cat, "value") else str(failure_cat) if failure_cat else None
                task.infra_tag = _classify_infra_tag(task.stop_reason, failure_cat_str)

                if hasattr(loop.executor, "cleanup_run"):
                    try:
                        loop.executor.cleanup_run(init_resp.run_id)
                    except Exception:
                        pass

        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
        finally:
            task.duration_seconds = round(time.time() - t_start, 1)

    suite.status = "completed"
    suite.completed_at = time.time()


def stop_suite(suite_id: str) -> bool:
    suite = _SUITES.get(suite_id)
    if suite is None or suite.status != "running":
        return False
    suite.cancelled = True
    return True


def create_single_task_suite(task_id: str) -> SuiteState:
    """Create a single-task SuiteState registered in _SUITES; raises ValueError if not found."""
    all_tasks = _load_all_tasks()
    task_data = next((t for t in all_tasks if t["task_id"] == task_id), None)
    if task_data is None:
        raise ValueError(f"task_id {task_id!r} not found")
    suite_id = uuid.uuid4().hex[:8]
    task = _make_task_result(task_data)
    suite = SuiteState(suite_id=suite_id, difficulty=task_data["difficulty"], tasks=[task])
    _SUITES[suite_id] = suite
    return suite


# ---------------------------------------------------------------------------
# Stress Runner — high-reliability benchmark execution
# ---------------------------------------------------------------------------

# Candidate top-left window positions (x, y) used to randomise window placement.
# Spread across a 1920×1080 display; constrained so a typical 1280×800 browser
# window stays fully on-screen.
_WINDOW_POSITIONS: list[tuple[int, int]] = [
    (0, 0),
    (320, 0),
    (0, 140),
    (320, 140),
    (100, 60),
]


@dataclass
class StressAttempt:
    """Result of one repetition of a stress task."""

    repetition: int          # 1-based index within the k repetitions
    run_id: str | None
    succeeded: bool
    step_count: int
    stop_reason: str | None
    window_x: int            # intended window position x
    window_y: int            # intended window position y
    duration_seconds: float
    error: str | None = None


@dataclass
class StressTaskResult:
    """Aggregated result across k repetitions of one benchmark task."""

    task_id: str
    intent: str
    start_url: str
    k: int
    attempts: list[StressAttempt] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(1 for a in self.attempts if a.succeeded)

    @property
    def reliability_score(self) -> float:
        """Fraction of repetitions that succeeded (0.0–1.0)."""
        return self.successes / self.k if self.k > 0 else 0.0

    @property
    def successful_run_ids(self) -> list[str]:
        return [a.run_id for a in self.attempts if a.succeeded and a.run_id]

    @property
    def failed_run_ids(self) -> list[str]:
        return [a.run_id for a in self.attempts if not a.succeeded and a.run_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "intent": self.intent,
            "start_url": self.start_url,
            "k": self.k,
            "successes": self.successes,
            "reliability_score": round(self.reliability_score, 4),
            "attempts": [
                {
                    "repetition": a.repetition,
                    "run_id": a.run_id,
                    "succeeded": a.succeeded,
                    "step_count": a.step_count,
                    "stop_reason": a.stop_reason,
                    "window_x": a.window_x,
                    "window_y": a.window_y,
                    "duration_seconds": a.duration_seconds,
                    "error": a.error,
                }
                for a in self.attempts
            ],
        }


@dataclass
class StressRunResult:
    """Full result of a StressRunner execution across all tasks."""

    suite_id: str
    k: int
    task_results: list[StressTaskResult] = field(default_factory=list)
    trajectory_drift_px: dict[str, float | None] = field(default_factory=dict)
    completed_at: float = field(default_factory=time.time)

    @property
    def total_tasks(self) -> int:
        return len(self.task_results)

    @property
    def total_attempts(self) -> int:
        return self.total_tasks * self.k

    @property
    def total_successes(self) -> int:
        return sum(r.successes for r in self.task_results)

    @property
    def overall_reliability_score(self) -> float:
        """Total successes / (tasks × k). 1.0 = every run of every task passed."""
        return self.total_successes / self.total_attempts if self.total_attempts > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "k": self.k,
            "total_tasks": self.total_tasks,
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "overall_reliability_score": round(self.overall_reliability_score, 4),
            "completed_at": self.completed_at,
            "trajectory_drift_px": {
                k: round(v, 2) if v is not None else None
                for k, v in self.trajectory_drift_px.items()
            },
            "tasks": [r.to_dict() for r in self.task_results],
        }


def _try_set_window_position(x: int, y: int) -> bool:
    """Best-effort: move the foreground window to (x, y) on Windows.

    Uses ctypes.windll.user32 directly — no additional packages required.
    Returns True if the move succeeded, False if the platform is unsupported
    or the call failed.
    """
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        if not hwnd:
            return False
        # GetWindowRect to preserve current width/height
        rect = ctypes.wintypes.RECT()  # type: ignore[attr-defined]
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))  # type: ignore[attr-defined]
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        ctypes.windll.user32.MoveWindow(hwnd, x, y, w, h, True)  # type: ignore[attr-defined]
        return True
    except Exception as exc:
        logger.debug("_try_set_window_position(%d, %d): %s", x, y, exc)
        return False


class StressRunner:
    """Execute a benchmark task k times with randomized window positions.

    Each repetition starts a fresh run via `get_loop_fn()` and records whether
    it succeeded. The `ReliabilityScore` for a task is (successes / k); the
    overall score across all tasks is (total_successes / (tasks × k)).

    Window positions are drawn without replacement from `_WINDOW_POSITIONS` using
    a seeded shuffle so runs are deterministically varied but not monotone.
    Best-effort: if the OS call fails the run proceeds from whatever position the
    window is already in — the score still counts.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = max(1, k)

    async def run_task(
        self,
        task_data: dict[str, Any],
        *,
        max_steps: int,
        get_loop_fn: Callable,
        headless: bool = False,
        seed: int | None = None,
    ) -> StressTaskResult:
        """Run one task k times and return the aggregated StressTaskResult."""
        rng = random.Random(seed)
        positions = list(_WINDOW_POSITIONS)
        rng.shuffle(positions)
        # Pad with random draws if k > len(positions)
        while len(positions) < self.k:
            positions.append(rng.choice(_WINDOW_POSITIONS))

        result = StressTaskResult(
            task_id=task_data["task_id"],
            intent=task_data["intent"],
            start_url=task_data["start_url"],
            k=self.k,
        )

        for rep in range(1, self.k + 1):
            wx, wy = positions[rep - 1]
            t_start = time.time()
            attempt = StressAttempt(
                repetition=rep,
                run_id=None,
                succeeded=False,
                step_count=0,
                stop_reason=None,
                window_x=wx,
                window_y=wy,
                duration_seconds=0.0,
            )
            try:
                await asyncio.to_thread(_try_set_window_position, wx, wy)
                logger.info(
                    "stress rep=%d/%d task=%s window=(%d,%d)",
                    rep, self.k, task_data["task_id"], wx, wy,
                )

                async with _get_browser_lock():
                    loop = get_loop_fn()
                    req = RunTaskRequest(
                        intent=task_data["intent"],
                        start_url=task_data["start_url"],
                        headless=headless,
                    )
                    init_resp = await loop.start_run(req)
                    attempt.run_id = init_resp.run_id

                    last_resp = init_resp
                    for _ in range(max_steps):
                        last_resp = await loop.step_run(StepRequest(run_id=init_resp.run_id))
                        attempt.step_count = last_resp.step_count or 0
                        if last_resp.status not in (RunStatus.RUNNING, RunStatus.WAITING_FOR_USER):
                            break

                    raw_stop = getattr(last_resp, "stop_reason", None)
                    attempt.stop_reason = (
                        raw_stop.value if hasattr(raw_stop, "value") else str(raw_stop) if raw_stop else None
                    )
                    succeeded = last_resp.status == RunStatus.SUCCEEDED
                    if not succeeded and attempt.stop_reason:
                        succeeded = attempt.stop_reason in _SUCCESS_STOP_REASONS
                    attempt.succeeded = succeeded

                    if hasattr(loop.executor, "cleanup_run"):
                        try:
                            loop.executor.cleanup_run(init_resp.run_id)
                        except Exception:
                            pass

            except Exception as exc:
                attempt.error = str(exc)
                logger.warning("stress rep=%d task=%s error: %s", rep, task_data["task_id"], exc)
            finally:
                attempt.duration_seconds = round(time.time() - t_start, 1)
                result.attempts.append(attempt)

            logger.info(
                "stress rep=%d/%d task=%s succeeded=%s reliability_so_far=%.2f",
                rep, self.k, task_data["task_id"],
                attempt.succeeded,
                result.reliability_score,
            )

        return result

    async def run_suite(
        self,
        tasks: list[dict[str, Any]],
        *,
        max_steps: int,
        get_loop_fn: Callable,
        headless: bool = False,
        root_dir: str | Path = "runs",
    ) -> StressRunResult:
        """Run all tasks k times each and return a StressRunResult with reliability scores."""
        suite_result = StressRunResult(suite_id=uuid.uuid4().hex[:8], k=self.k)

        for task_data in tasks:
            task_result = await self.run_task(
                task_data,
                max_steps=max_steps,
                get_loop_fn=get_loop_fn,
                headless=headless,
                seed=hash(task_data["task_id"]) & 0xFFFFFFFF,
            )
            drift = compute_trajectory_drift(
                run_ids=([a.run_id for a in task_result.attempts if a.run_id]),
                successful_run_ids=set(task_result.successful_run_ids),
                root_dir=Path(root_dir),
            )
            suite_result.task_results.append(task_result)
            suite_result.trajectory_drift_px[task_data["task_id"]] = drift

            logger.info(
                "stress task=%s reliability=%.2f drift_px=%s",
                task_data["task_id"],
                task_result.reliability_score,
                f"{drift:.1f}" if drift is not None else "n/a",
            )

        suite_result.completed_at = time.time()
        logger.info(
            "stress suite=%s overall_reliability=%.4f tasks=%d k=%d",
            suite_result.suite_id,
            suite_result.overall_reliability_score,
            suite_result.total_tasks,
            self.k,
        )
        return suite_result


def compute_trajectory_drift(
    run_ids: list[str],
    successful_run_ids: set[str],
    root_dir: Path = Path("runs"),
) -> float | None:
    """Compute mean pixel distance between click coordinates in successful vs failed runs.

    For each failed run, finds the closest matching successful-run click at the
    same step index and measures the Euclidean distance. Returns the mean of all
    such distances, or None if there are no valid pairs to compare.

    This metric surfaces whether failures correlate with coordinate drift — i.e.
    whether the agent is clicking in systematically different positions on failed
    runs vs successful ones.
    """
    success_clicks: dict[int, list[tuple[float, float]]] = {}  # step_idx → [(x, y)]
    failed_clicks: dict[int, list[tuple[float, float]]] = {}

    for run_id in run_ids:
        log_path = root_dir / run_id / "run.jsonl"
        if not log_path.exists():
            continue
        bucket = success_clicks if run_id in successful_run_ids else failed_clicks
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                step = json.loads(line)
                action = step.get("policy_decision", {}).get("action", {})
                if action.get("action_type") not in ("click", "type"):
                    continue
                x = action.get("x")
                y = action.get("y")
                if x is None or y is None:
                    continue
                idx = step.get("step_index", 0)
                bucket.setdefault(idx, []).append((float(x), float(y)))
        except Exception:
            continue

    if not success_clicks or not failed_clicks:
        return None

    distances: list[float] = []
    for step_idx, fail_coords in failed_clicks.items():
        # Find nearest success step (same or adjacent)
        best_success: list[tuple[float, float]] | None = None
        for offset in (0, 1, -1, 2, -2):
            if step_idx + offset in success_clicks:
                best_success = success_clicks[step_idx + offset]
                break
        if best_success is None:
            continue
        for fx, fy in fail_coords:
            min_dist = min(
                math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
                for sx, sy in best_success
            )
            distances.append(min_dist)

    return round(sum(distances) / len(distances), 2) if distances else None
