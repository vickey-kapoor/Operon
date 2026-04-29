"""In-memory benchmark suite runner — runs WebArena/Mind2Web/Operon tasks sequentially."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.models.common import RunStatus, RunTaskRequest, StepRequest

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
