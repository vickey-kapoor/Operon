"""Local file-backed advisory memory for agent policy hints."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from src.models.common import FailureCategory, LoopStage
from src.models.episode import Episode
from src.models.execution import ExecutedAction
from src.models.memory import MemoryHint, MemoryOutcome, MemoryRecord
from src.models.perception import PageHint, ScreenPerception, UIElementType
from src.models.policy import ActionType, PolicyDecision
from src.models.recovery import RecoveryDecision
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

GENERIC_TASK = "generic_task"


def normalize_intent(intent: str) -> str:
    """Normalize an intent string for episode matching."""
    text = intent.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:!?")
    return text


class MemoryStore(ABC):
    """Typed advisory memory interface."""

    @abstractmethod
    def get_hints(
        self,
        *,
        benchmark: str,
        page_hint: PageHint,
        subgoal: str | None,
        recent_failure_category: FailureCategory | None,
        limit: int = 4,
    ) -> list[MemoryHint]:
        """Return a small set of relevant advisory hints."""

    @abstractmethod
    def record_step(
        self,
        *,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
        recovery: RecoveryDecision,
    ) -> list[MemoryRecord]:
        """Persist compact memory candidates from a completed step."""

    @abstractmethod
    def save_episode(self, episode: Episode) -> None:
        """Persist a compressed episode trajectory from a successful run."""

    @abstractmethod
    def get_episode(self, normalized_intent: str, benchmark: str) -> Episode | None:
        """Retrieve the best-matching episode for a given intent."""


class FileBackedMemoryStore(MemoryStore):
    """Append-only local advisory memory stored as compact JSONL records."""

    def __init__(self, root_dir: str | Path = "runs") -> None:
        self.root_dir = Path(root_dir)
        self.memory_dir = self.root_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.memory_dir / "memory.jsonl"
        self._cached_records: list[MemoryRecord] | None = None
        self._cached_mtime: float = 0.0
        self.episodes_path = self.memory_dir / "episodes.jsonl"
        self._cached_episodes: list[Episode] | None = None
        self._cached_episodes_mtime: float = 0.0
        self._seeded_guardrail_keys: set[str] = set()
        self._seed_default_guardrails()

    _WEIGHT_PRUNE_THRESHOLD = 0.1

    def get_hints(
        self,
        *,
        benchmark: str,
        page_hint: PageHint,
        subgoal: str | None,
        recent_failure_category: FailureCategory | None,
        limit: int = 4,
    ) -> list[MemoryHint]:
        records = self._load_records()
        ranked: dict[tuple[str, str], dict[str, object]] = {}

        for record in records:
            if record.benchmark != benchmark:
                continue

            score = 0
            if record.outcome is MemoryOutcome.GUARDRAIL:
                score += 1
            if record.page_hint is not None and record.page_hint == page_hint:
                score += 4
            elif record.page_hint is None:
                score += 1
            else:
                continue

            if subgoal and record.subgoal and record.subgoal == subgoal:
                score += 4
            elif record.subgoal is None:
                score += 1

            if recent_failure_category and record.failure_category is recent_failure_category:
                score += 4
            elif record.failure_category is None:
                score += 1

            if score <= 0:
                continue

            bucket_key = (record.key, record.hint)
            bucket = ranked.setdefault(
                bucket_key,
                {"key": record.key, "hint": record.hint, "count": 0, "score": score, "source": "memory",
                 "weight_sum": 0.0, "weight_n": 0},
            )
            bucket["count"] = int(bucket["count"]) + record.count
            bucket["score"] = max(int(bucket["score"]), score) + record.count
            bucket["weight_sum"] = float(bucket["weight_sum"]) + record.weight
            bucket["weight_n"] = int(bucket["weight_n"]) + 1

        # Compute effective weight per bucket (mean of all contributing records) and prune
        pruned = {
            k: v for k, v in ranked.items()
            if (float(v["weight_sum"]) / max(1, int(v["weight_n"]))) >= self._WEIGHT_PRUNE_THRESHOLD
        }
        ordered = sorted(
            pruned.values(),
            key=lambda item: (int(item["score"]), int(item["count"]), str(item["key"])),
            reverse=True,
        )
        return [
            MemoryHint(
                key=str(item["key"]),
                hint=str(item["hint"]),
                source=str(item["source"]),
                count=int(item["count"]),
            )
            for item in ordered[:limit]
        ]

    def record_step(
        self,
        *,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
        recovery: RecoveryDecision,
    ) -> list[MemoryRecord]:
        benchmark = state.benchmark or GENERIC_TASK
        records = self._build_step_records(
            benchmark=benchmark,
            state=state,
            perception=perception,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            recovery=recovery,
        )
        for record in records:
            self._append_record(record)

        # Memory decay: when verification failed, halve the weight of hints that were
        # active for this benchmark+page_hint+subgoal context (they were shown to the
        # agent and correlated with a failing step).
        if verification.status is VerificationStatus.FAILURE:
            self._decay_active_hints(
                benchmark=benchmark,
                page_hint=perception.page_hint,
                subgoal=state.current_subgoal,
            )

        return records

    def _decay_active_hints(
        self,
        *,
        benchmark: str,
        page_hint: PageHint,
        subgoal: str | None,
    ) -> None:
        """Append decay records for hints active in this context after a verification failure."""
        existing = self._load_records()
        # Find the current effective weight per (key, hint) bucket for this context
        bucket_weights: dict[tuple[str, str], list[float]] = {}
        for record in existing:
            if record.benchmark != benchmark:
                continue
            if record.page_hint is not None and record.page_hint != page_hint:
                continue
            bk = (record.key, record.hint)
            bucket_weights.setdefault(bk, []).append(record.weight)

        for (key, hint), weights in bucket_weights.items():
            effective = sum(weights) / len(weights)
            if effective < self._WEIGHT_PRUNE_THRESHOLD:
                continue  # already pruned — no further decay needed
            decayed = max(0.0, effective * 0.5)
            self._append_record(
                MemoryRecord(
                    key=key,
                    benchmark=benchmark,
                    hint=hint,
                    outcome=MemoryOutcome.FAILURE,
                    page_hint=page_hint,
                    subgoal=subgoal,
                    success=False,
                    weight=decayed,
                )
            )
            logger.debug("Memory decay: key=%r effective_weight %.3f → %.3f", key, effective, decayed)

    def _seed_default_guardrails(self) -> None:
        from src.benchmarks.registry import BENCHMARK_REGISTRY
        existing = self._load_records()
        existing_keys = {(record.key, record.benchmark) for record in existing if record.outcome is MemoryOutcome.GUARDRAIL}
        self._seeded_guardrail_keys = {k for k, _ in existing_keys}
        generic_seeds = [
            MemoryRecord(
                key="click_before_type",
                benchmark=GENERIC_TASK,
                hint="When input focus is uncertain, click the input before typing.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
            MemoryRecord(
                key="avoid_identical_type_retry",
                benchmark=GENERIC_TASK,
                hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
        ]
        all_seeds = generic_seeds + BENCHMARK_REGISTRY.all_seeds()
        for record in all_seeds:
            if (record.key, record.benchmark) not in existing_keys:
                self._append_record(record)
                self._seeded_guardrail_keys.add(record.key)

    def _build_step_records(
        self,
        *,
        benchmark: str,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
        verification: VerificationResult,
        recovery: RecoveryDecision,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        action = decision.action
        recent_failure_category = verification.failure_category or executed_action.failure_category
        recent_failure_stage = verification.failure_stage or executed_action.failure_stage

        if (
            action.action_type is ActionType.TYPE
            and recent_failure_category in {
                FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
            }
        ):
            records.append(
                MemoryRecord(
                    key="avoid_identical_type_retry",
                    benchmark=benchmark,
                    hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
                    outcome=MemoryOutcome.FAILURE,
                    page_hint=perception.page_hint,
                    subgoal=state.current_subgoal,
                    action_type=ActionType.TYPE,
                    target_element_id=action.target_element_id,
                    failure_category=recent_failure_category,
                    stage=recent_failure_stage or LoopStage.EXECUTE,
                    success=False,
                )
            )
            records.append(
                MemoryRecord(
                    key="input_target_failure_signal",
                    benchmark=benchmark,
                    hint="A type action recently failed on this input-like target; click to re-establish focus before typing again.",
                    outcome=MemoryOutcome.FAILURE,
                    page_hint=perception.page_hint,
                    subgoal=state.current_subgoal,
                    action_type=ActionType.TYPE,
                    target_element_id=action.target_element_id,
                    failure_category=recent_failure_category,
                    stage=recent_failure_stage or LoopStage.EXECUTE,
                    success=False,
                )
            )

        if (
            action.action_type is ActionType.CLICK
            and action.target_element_id is not None
            and verification.status is VerificationStatus.SUCCESS
            and self._target_is_input_like(perception, action.target_element_id)
        ):
            records.append(
                MemoryRecord(
                    key="click_before_type",
                    benchmark=benchmark,
                    hint="Click-before-type was successful for an input-like target.",
                    outcome=MemoryOutcome.SUCCESS,
                    page_hint=perception.page_hint,
                    subgoal=state.current_subgoal,
                    action_type=ActionType.CLICK,
                    target_element_id=action.target_element_id,
                    stage=LoopStage.CHOOSE_ACTION,
                    success=True,
                )
            )

        if verification.status is VerificationStatus.SUCCESS and action.action_type is not ActionType.STOP:
            records.append(
                MemoryRecord(
                    key="successful_action_pattern",
                    benchmark=benchmark,
                    hint=f"Successful {action.action_type.value} pattern on {perception.page_hint.value}.",
                    outcome=MemoryOutcome.SUCCESS,
                    page_hint=perception.page_hint,
                    subgoal=state.current_subgoal,
                    action_type=action.action_type,
                    target_element_id=action.target_element_id,
                    stage=LoopStage.EXECUTE,
                    success=True,
                )
            )

        return records

    def _target_is_input_like(self, perception: ScreenPerception, target_element_id: str) -> bool:
        for element in perception.visible_elements:
            if element.element_id == target_element_id:
                return element.element_type is UIElementType.INPUT
        return False

    def _load_records(self) -> list[MemoryRecord]:
        if not self.memory_path.exists():
            return []
        try:
            mtime = self.memory_path.stat().st_mtime
        except OSError:
            return []
        if self._cached_records is not None and mtime == self._cached_mtime:
            return self._cached_records
        records: list[MemoryRecord] = []
        for lineno, line in enumerate(self.memory_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(MemoryRecord.model_validate_json(line))
            except Exception as exc:
                logger.warning("memory store: skipping corrupt line %d in %s: %s", lineno, self.memory_path, exc)
        self._cached_records = records
        self._cached_mtime = mtime
        return records

    def _append_record(self, record: MemoryRecord) -> None:
        from src.store.background_writer import bg_writer
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        bg_writer.append(self.memory_path, record.model_dump_json() + "\n")
        # Update in-memory cache immediately so get_hints() sees the new record.
        if self._cached_records is not None:
            self._cached_records.append(record)
            try:
                self._cached_mtime = self.memory_path.stat().st_mtime
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Episode storage
    # ------------------------------------------------------------------

    def save_episode(self, episode: Episode) -> None:
        episodes = self._load_episodes()
        updated = False
        for i, existing in enumerate(episodes):
            if existing.normalized_intent == episode.normalized_intent and existing.benchmark == episode.benchmark:
                episode = episode.model_copy(update={"success_count": existing.success_count + 1})
                episodes[i] = episode
                updated = True
                break
        if not updated:
            episodes.append(episode)
        self._write_episodes(episodes)

    def get_episode(self, normalized_intent: str, benchmark: str) -> Episode | None:
        episodes = self._load_episodes()
        exact: Episode | None = None
        containment: Episode | None = None
        for ep in episodes:
            if ep.benchmark != benchmark:
                continue
            if ep.normalized_intent == normalized_intent:
                if exact is None or ep.success_count > exact.success_count:
                    exact = ep
            elif normalized_intent in ep.normalized_intent or ep.normalized_intent in normalized_intent:
                if containment is None or ep.success_count > containment.success_count:
                    containment = ep
        return exact or containment

    def _load_episodes(self) -> list[Episode]:
        if not self.episodes_path.exists():
            return []
        try:
            mtime = self.episodes_path.stat().st_mtime
        except OSError:
            return []
        if self._cached_episodes is not None and mtime == self._cached_episodes_mtime:
            return list(self._cached_episodes)
        episodes: list[Episode] = []
        for line in self.episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(Episode.model_validate_json(line))
        self._cached_episodes = episodes
        self._cached_episodes_mtime = mtime
        return list(episodes)

    def _write_episodes(self, episodes: list[Episode]) -> None:
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.episodes_path.open("w", encoding="utf-8") as handle:
            for ep in episodes:
                handle.write(ep.model_dump_json())
                handle.write("\n")
        self._cached_episodes = list(episodes)
        try:
            self._cached_episodes_mtime = self.episodes_path.stat().st_mtime
        except OSError:
            pass
