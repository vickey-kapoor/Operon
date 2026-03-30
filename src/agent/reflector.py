"""Post-run reflector that analyzes completed runs and generates memory records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.common import FailureCategory, LoopStage
from src.models.memory import MemoryOutcome, MemoryRecord
from src.models.reflection import ReflectionPattern, RunReflection
from src.store.memory import MemoryStore, benchmark_name_for_intent

logger = logging.getLogger(__name__)

# Taskbar is typically in the bottom ~60px of a 1080p screen.
_TASKBAR_Y_THRESHOLD = 1000


class PostRunReflector:
    """Analyze completed runs and write learnings to the memory store."""

    def __init__(self, memory_store: MemoryStore, root_dir: str | Path = "runs") -> None:
        self.memory_store = memory_store
        self.root_dir = Path(root_dir)

    def reflect(self, run_id: str) -> RunReflection:
        """Analyze a completed run and generate memory records from failure patterns."""
        state, steps = self._load_run(run_id)
        if state is None or not steps:
            return RunReflection(run_id=run_id, success=False, total_steps=0)

        success = state.get("status") == "succeeded"
        intent = state.get("intent", "")
        benchmark = benchmark_name_for_intent(intent)
        patterns: list[ReflectionPattern] = []

        patterns.extend(self._detect_repeated_key_press(steps, run_id))
        patterns.extend(self._detect_taskbar_clicking(steps, run_id))
        patterns.extend(self._detect_stuck_subgoal(steps, run_id))
        patterns.extend(self._detect_no_screen_change_actions(steps, run_id))

        # Convert patterns to memory records
        memories_written = 0
        for pattern in patterns:
            record = MemoryRecord(
                key=pattern.pattern_key,
                benchmark=benchmark,
                hint=pattern.suggested_action,
                outcome=MemoryOutcome.FAILURE,
                failure_category=FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                stage=LoopStage.ORCHESTRATE,
                success=False,
                count=1,
            )
            self.memory_store._append_record(record)
            memories_written += 1

        reflection = RunReflection(
            run_id=run_id,
            success=success,
            total_steps=len(steps),
            patterns=patterns,
            memories_generated=memories_written,
        )

        # Persist reflection artifact
        reflection_path = self.root_dir / run_id / "reflection.json"
        try:
            reflection_path.write_text(reflection.model_dump_json(), encoding="utf-8")
        except Exception:
            logger.debug("Failed to write reflection artifact", exc_info=True)

        return reflection

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_repeated_key_press(steps: list[dict], run_id: str) -> list[ReflectionPattern]:
        """Detect the same press_key or hotkey repeated 3+ times in a row."""
        patterns = []
        streak_key: str | None = None
        streak_start = 0
        streak_count = 0

        for step in steps:
            action = step.get("policy_decision", {}).get("action", {})
            atype = action.get("action_type", "")
            key = action.get("key", "")
            step_idx = step.get("step_index", 0)

            sig = f"{atype}|{key}" if atype in ("press_key", "hotkey") else None

            if sig and sig == streak_key:
                streak_count += 1
            else:
                if streak_count >= 3:
                    patterns.append(ReflectionPattern(
                        pattern_key=f"repeated_{streak_key.replace('|', '_')}",
                        description=f"Pressed {streak_key} {streak_count} times in a row without effect",
                        trigger_context=f"action={streak_key}",
                        suggested_action=f"Do not repeat {streak_key} more than once. If the first press does not work, try a different approach or advance to the next subgoal.",
                        confidence=min(0.5 + streak_count * 0.1, 0.95),
                        source_run_id=run_id,
                        source_steps=list(range(streak_start, streak_start + streak_count)),
                    ))
                streak_key = sig
                streak_start = step_idx
                streak_count = 1

        # Check final streak
        if streak_count >= 3 and streak_key:
            patterns.append(ReflectionPattern(
                pattern_key=f"repeated_{streak_key.replace('|', '_')}",
                description=f"Pressed {streak_key} {streak_count} times in a row without effect",
                trigger_context=f"action={streak_key}",
                suggested_action=f"Do not repeat {streak_key} more than once. If the first press does not work, try a different approach or advance to the next subgoal.",
                confidence=min(0.5 + streak_count * 0.1, 0.95),
                source_run_id=run_id,
                source_steps=list(range(streak_start, streak_start + streak_count)),
            ))

        return patterns

    @staticmethod
    def _detect_taskbar_clicking(steps: list[dict], run_id: str) -> list[ReflectionPattern]:
        """Detect repeated clicks near the taskbar area to open apps."""
        taskbar_clicks = []
        for step in steps:
            action = step.get("policy_decision", {}).get("action", {})
            atype = action.get("action_type", "")
            y = action.get("y")
            step_idx = step.get("step_index", 0)
            if atype == "click" and y is not None and y >= _TASKBAR_Y_THRESHOLD:
                taskbar_clicks.append(step_idx)

        if len(taskbar_clicks) >= 3:
            return [ReflectionPattern(
                pattern_key="taskbar_clicking_instead_of_launch",
                description=f"Clicked the taskbar {len(taskbar_clicks)} times trying to open an app",
                trigger_context="clicks near y>=1000 (taskbar region)",
                suggested_action="Always use launch_app action to open applications. Never click the taskbar — coordinates are unreliable for small icons.",
                confidence=0.9,
                source_run_id=run_id,
                source_steps=taskbar_clicks,
            )]
        return []

    @staticmethod
    def _detect_stuck_subgoal(steps: list[dict], run_id: str) -> list[ReflectionPattern]:
        """Detect when the same subgoal persists for 4+ steps."""
        patterns = []
        current_subgoal: str | None = None
        subgoal_start = 0
        subgoal_count = 0

        for step in steps:
            subgoal = step.get("policy_decision", {}).get("active_subgoal", "")
            step_idx = step.get("step_index", 0)

            if subgoal == current_subgoal:
                subgoal_count += 1
            else:
                if subgoal_count >= 4 and current_subgoal:
                    patterns.append(ReflectionPattern(
                        pattern_key=f"stuck_subgoal",
                        description=f"Subgoal '{current_subgoal}' persisted for {subgoal_count} steps",
                        trigger_context=f"subgoal={current_subgoal}",
                        suggested_action=f"After completing an action, advance to the next subgoal immediately. Do not stay on '{current_subgoal}' if the action already succeeded.",
                        confidence=0.8,
                        source_run_id=run_id,
                        source_steps=list(range(subgoal_start, subgoal_start + subgoal_count)),
                    ))
                current_subgoal = subgoal
                subgoal_start = step_idx
                subgoal_count = 1

        if subgoal_count >= 4 and current_subgoal:
            patterns.append(ReflectionPattern(
                pattern_key=f"stuck_subgoal",
                description=f"Subgoal '{current_subgoal}' persisted for {subgoal_count} steps",
                trigger_context=f"subgoal={current_subgoal}",
                suggested_action=f"After completing an action, advance to the next subgoal immediately. Do not stay on '{current_subgoal}' if the action already succeeded.",
                confidence=0.8,
                source_run_id=run_id,
                source_steps=list(range(subgoal_start, subgoal_start + subgoal_count)),
            ))

        return patterns

    @staticmethod
    def _detect_no_screen_change_actions(steps: list[dict], run_id: str) -> list[ReflectionPattern]:
        """Detect sequences where actions produce no visible screen change."""
        no_change_streak = 0
        no_change_steps: list[int] = []

        for step in steps:
            progress = step.get("progress_state", {})
            step_idx = step.get("step_index", 0)
            # Check the progress_trace if available
            trace_path = step.get("progress_trace_artifact_path")
            no_progress = progress.get("no_progress_streak", 0)

            if no_progress > 0:
                no_change_streak += 1
                no_change_steps.append(step_idx)
            else:
                no_change_streak = 0
                no_change_steps = []

        if len(no_change_steps) >= 3:
            return [ReflectionPattern(
                pattern_key="actions_without_screen_change",
                description=f"{len(no_change_steps)} consecutive actions produced no visible screen change",
                trigger_context="no_progress_streak >= 3",
                suggested_action="If an action does not change the screen, do not repeat it. Try a completely different approach or use the stop action.",
                confidence=0.85,
                source_run_id=run_id,
                source_steps=no_change_steps,
            )]
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_run(self, run_id: str) -> tuple[dict | None, list[dict]]:
        """Load state.json and run.jsonl for a run."""
        run_dir = self.root_dir / run_id
        state_path = run_dir / "state.json"
        log_path = run_dir / "run.jsonl"

        state = None
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("Failed to load state.json for %s", run_id, exc_info=True)

        steps: list[dict] = []
        if log_path.exists():
            try:
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        steps.append(json.loads(line))
            except Exception:
                logger.debug("Failed to load run.jsonl for %s", run_id, exc_info=True)

        return state, steps
