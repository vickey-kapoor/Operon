"""Post-run reflector that analyzes completed runs and generates memory records."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models.common import FailureCategory, LoopStage
from src.models.episode import Episode, EpisodeStep
from src.models.memory import MemoryOutcome, MemoryRecord
from src.models.perception import PageHint
from src.models.policy import ActionType
from src.models.reflection import ReflectionPattern, RunReflection
from src.store.memory import GENERIC_TASK, MemoryStore, normalize_intent

logger = logging.getLogger(__name__)


class PostRunReflector:
    """Analyze completed runs and write learnings to the memory store."""

    def __init__(self, memory_store: MemoryStore, root_dir: str | Path = "runs") -> None:
        self.memory_store = memory_store
        self.root_dir = Path(root_dir)

    def reflect(self, run_id: str, *, reliability_score: float = 1.0) -> RunReflection:
        """Analyze a completed run and generate memory records from failure patterns.

        Args:
            run_id: The completed run to analyze.
            reliability_score: Fraction of stress-test repetitions that succeeded
                (0.0–1.0). Defaults to 1.0 for single-run calls. Episodes are only
                written to episodes.jsonl when this is 1.0 — i.e. the task passed
                every repetition of a k-run stress test — ensuring the Golden Path
                encodes a trajectory that is reproducible, not a lucky one-off.
        """
        state, steps = self._load_run(run_id)
        if state is None or not steps:
            return RunReflection(run_id=run_id, success=False, total_steps=0)

        success = state.get("status") == "succeeded"
        intent = state.get("intent", "")
        benchmark = state.get("benchmark") or GENERIC_TASK
        patterns: list[ReflectionPattern] = []

        patterns.extend(self._detect_repeated_key_press(steps, run_id))
        patterns.extend(self._detect_stuck_subgoal(steps, run_id))
        patterns.extend(self._detect_no_screen_change_actions(steps, run_id))

        # Convert patterns to memory records — failure patterns are always useful
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

        # Extract episode trajectory from successful runs.
        # Shortest-path optimization: only compress steps that succeeded on the
        # first attempt (no retries). Multi-attempt successes add noise — we want
        # the memory to encode the optimal path, not historical struggle.
        #
        # Golden Path gate: episodes are only persisted when reliability_score == 1.0,
        # meaning every repetition of a stress-test run succeeded. A single-run
        # success with reliability_score=1.0 (the default) also qualifies so that
        # non-stress callers keep the existing behaviour.
        if success and reliability_score >= 1.0:
            one_shot_steps = self._filter_one_shot_steps(steps)
            episode = self._extract_episode(run_id, intent, benchmark, one_shot_steps)
            if episode is not None:
                try:
                    self.memory_store.save_episode(episode)
                    logger.info(
                        "golden_path saved: run=%s reliability=%.2f steps=%d",
                        run_id, reliability_score, len(episode.steps),
                    )
                except Exception:
                    logger.debug("Failed to save episode for %s", run_id, exc_info=True)
        elif success and reliability_score < 1.0:
            logger.info(
                "golden_path suppressed: run=%s succeeded but reliability=%.2f < 1.0 — "
                "episode not written to memory",
                run_id, reliability_score,
            )

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
                        pattern_key="stuck_subgoal",
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
                pattern_key="stuck_subgoal",
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
    # Episode extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_one_shot_steps(steps: list[dict]) -> list[dict]:
        """Return only steps that succeeded on the first attempt with no retries.

        A step is considered one-shot when:
        - executed_action.success is True
        - executed_action.failure_category is None (no execution failure)
        - The step's retry_counts did not increase relative to the previous step

        This filters out steps where the agent had to retry an action, keeping only
        the direct-path steps that encode the shortest route to goal completion.
        """
        if not steps:
            return []

        one_shot: list[dict] = []
        prev_retry_total = 0

        for step in steps:
            executed = step.get("executed_action", {})
            # A step is successful when it has no failure_category (matches _extract_episode logic)
            if executed.get("failure_category") is not None:
                prev_retry_total = sum(step.get("retry_counts", {}).values()) if step.get("retry_counts") else 0
                continue
            # Check retry counts didn't grow this step (retries used = multi-attempt, not shortest path)
            current_retry_total = sum(step.get("retry_counts", {}).values()) if step.get("retry_counts") else 0
            if current_retry_total > prev_retry_total:
                prev_retry_total = current_retry_total
                continue
            one_shot.append(step)
            prev_retry_total = current_retry_total

        return one_shot

    @staticmethod
    def _extract_episode(
        run_id: str,
        intent: str,
        benchmark: str,
        steps: list[dict],
    ) -> Episode | None:
        """Compress a successful run into a reusable episode trajectory."""
        _SKIP_ACTIONS = {"stop", "wait"}
        episode_steps: list[EpisodeStep] = []

        for step in steps:
            policy = step.get("policy_decision", {})
            action = policy.get("action", {})
            action_type_str = action.get("action_type", "")

            if action_type_str in _SKIP_ACTIONS:
                continue

            # Only include steps where execution succeeded
            executed = step.get("executed_action", {})
            if executed.get("failure_category") is not None:
                continue

            perception = step.get("perception", {})
            page_hint_str = perception.get("page_hint", "")

            # Resolve target description from perception elements
            target_description = None
            target_id = action.get("target_element_id")
            if target_id:
                for elem in perception.get("visible_elements", []):
                    if elem.get("element_id") == target_id:
                        target_description = elem.get("primary_name")
                        break

            try:
                episode_steps.append(EpisodeStep(
                    step_index=step.get("step_index", len(episode_steps) + 1),
                    page_hint=PageHint(page_hint_str),
                    action_type=ActionType(action_type_str),
                    target_description=target_description,
                    text=action.get("text"),
                    key=action.get("key"),
                    subgoal=policy.get("active_subgoal") or "unknown",
                ))
            except (ValueError, KeyError):
                continue

        if len(episode_steps) < 2:
            return None

        return Episode(
            episode_id=run_id,
            normalized_intent=normalize_intent(intent),
            benchmark=benchmark,
            source_run_id=run_id,
            steps=episode_steps,
            success_count=1,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

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
