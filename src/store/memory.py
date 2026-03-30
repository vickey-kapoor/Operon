"""Local file-backed advisory memory for agent policy hints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction
from src.models.memory import MemoryHint, MemoryOutcome, MemoryRecord
from src.models.perception import PageHint, ScreenPerception, UIElementType
from src.models.policy import ActionType, PolicyDecision
from src.models.recovery import RecoveryDecision
from src.models.state import AgentState
from src.models.verification import VerificationResult, VerificationStatus

FORM_BENCHMARK = "auth_free_form"
GMAIL_BENCHMARK = "gmail_draft_authenticated"
GENERIC_TASK = "generic_task"
DEFAULT_BENCHMARK = FORM_BENCHMARK


def benchmark_name_for_intent(intent: str) -> str:
    """Map an intent string to a task category key for memory scoping."""

    lowered = intent.lower()
    if "gmail" in lowered:
        return GMAIL_BENCHMARK
    if "form" in lowered and ("submit" in lowered or "fill" in lowered or "complete" in lowered):
        return FORM_BENCHMARK
    return GENERIC_TASK


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


class FileBackedMemoryStore(MemoryStore):
    """Append-only local advisory memory stored as compact JSONL records."""

    def __init__(self, root_dir: str | Path = "runs") -> None:
        self.root_dir = Path(root_dir)
        self.memory_dir = self.root_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.memory_dir / "memory.jsonl"
        self._cached_records: list[MemoryRecord] | None = None
        self._cached_mtime: float = 0.0
        self._seed_default_guardrails()

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
            if record.page_hint is page_hint:
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
                {"key": record.key, "hint": record.hint, "count": 0, "score": score, "source": "memory"},
            )
            bucket["count"] = int(bucket["count"]) + record.count
            bucket["score"] = max(int(bucket["score"]), score) + record.count

        ordered = sorted(
            ranked.values(),
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
        benchmark = benchmark_name_for_intent(state.intent)
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
        return records

    def _seed_default_guardrails(self) -> None:
        existing = self._load_records()
        existing_keys = {record.key for record in existing if record.outcome is MemoryOutcome.GUARDRAIL}
        defaults = [
            MemoryRecord(
                key="click_before_type",
                benchmark=FORM_BENCHMARK,
                hint="When input focus is uncertain, click the input before typing.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
            MemoryRecord(
                key="avoid_identical_type_retry",
                benchmark=FORM_BENCHMARK,
                hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
            MemoryRecord(
                key="click_before_type",
                benchmark=GMAIL_BENCHMARK,
                hint="When input focus is uncertain, click the input before typing.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
            MemoryRecord(
                key="avoid_identical_type_retry",
                benchmark=GMAIL_BENCHMARK,
                hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
                outcome=MemoryOutcome.GUARDRAIL,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
            MemoryRecord(
                key="authenticated_start_required",
                benchmark=GMAIL_BENCHMARK,
                hint="Login pages are out of scope for this benchmark; use an authenticated Gmail start state.",
                outcome=MemoryOutcome.GUARDRAIL,
                page_hint=PageHint.GOOGLE_SIGN_IN,
                stage=LoopStage.CHOOSE_ACTION,
                success=False,
            ),
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
        for record in defaults:
            if record.key not in existing_keys:
                self._append_record(record)

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

        if benchmark == GMAIL_BENCHMARK and perception.page_hint is PageHint.GOOGLE_SIGN_IN:
            records.append(
                MemoryRecord(
                    key="authenticated_start_required",
                    benchmark=GMAIL_BENCHMARK,
                    hint="Login pages are out of scope for this benchmark; use an authenticated Gmail start state.",
                    outcome=MemoryOutcome.FAILURE,
                    page_hint=perception.page_hint,
                    subgoal=state.current_subgoal,
                    failure_category=recent_failure_category,
                    stage=recovery.failure_stage or LoopStage.CHOOSE_ACTION,
                    success=False,
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
        for line in self.memory_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(MemoryRecord.model_validate_json(line))
        self._cached_records = records
        self._cached_mtime = mtime
        return records

    def _append_record(self, record: MemoryRecord) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")
        # Append to in-memory cache instead of full invalidation
        if self._cached_records is not None:
            self._cached_records.append(record)
            try:
                self._cached_mtime = self.memory_path.stat().st_mtime
            except OSError:
                pass
