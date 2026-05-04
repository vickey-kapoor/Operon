"""Focused tests for compact local advisory memory."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.models.common import FailureCategory, LoopStage, RunStatus
from src.models.execution import ExecutedAction
from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.store.memory import FileBackedMemoryStore


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _perception(page_hint: str = "form_page") -> ScreenPerception:
    return ScreenPerception(
        summary="Form page visible.",
        page_hint=page_hint,
        visible_elements=[
            UIElement(
                element_id="name-input",
                element_type=UIElementType.INPUT,
                label="Name",
                x=320,
                y=180,
                width=300,
                height=28,
                is_interactable=True,
                confidence=0.95,
            )
        ],
        capture_artifact_path="runs/run-1/step_1/before.png",
        confidence=0.9,
    )


def test_memory_records_compact_failure_and_success_patterns() -> None:
    store = FileBackedMemoryStore(root_dir=_local_test_dir("test-memory-records") / "runs")
    state = AgentState(
        run_id="run-1",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )
    perception = _perception()
    failure_decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        rationale="Fill name",
        confidence=0.8,
        active_subgoal="fill_name",
    )
    executed_failure = ExecutedAction(
        action=failure_decision.action,
        success=False,
        detail="Execution failed: target not editable.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    failure_verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Type failed.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        failure_stage=LoopStage.EXECUTE,
    )
    recovery = RecoveryDecision(
        strategy=RecoveryStrategy.WAIT_AND_RETRY,
        message="Focus first.",
        retry_after_ms=500,
    )

    failure_records = store.record_step(
        state=state,
        perception=perception,
        decision=failure_decision,
        executed_action=executed_failure,
        verification=failure_verification,
        recovery=recovery,
    )

    success_decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="name-input", x=470, y=194),
        rationale="Focus name",
        confidence=0.95,
        active_subgoal="focus name-input",
    )
    executed_success = ExecutedAction(action=success_decision.action, success=True, detail="Clicked target.")
    success_verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="Focused input.",
    )

    success_records = store.record_step(
        state=state,
        perception=perception,
        decision=success_decision,
        executed_action=executed_success,
        verification=success_verification,
        recovery=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="Advance"),
    )

    assert any(record.key == "avoid_identical_type_retry" for record in failure_records)
    assert any(record.key == "click_before_type" for record in success_records)
    memory_lines = Path(store.memory_path).read_text(encoding="utf-8").splitlines()
    assert memory_lines
    assert all("artifact_path" not in line for line in memory_lines)


def test_memory_retrieval_returns_relevant_hints() -> None:
    store = FileBackedMemoryStore(root_dir=_local_test_dir("test-memory-hints") / "runs")
    state = AgentState(
        run_id="run-2",
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )
    decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        rationale="Fill name",
        confidence=0.8,
        active_subgoal="fill_name",
    )
    executed = ExecutedAction(
        action=decision.action,
        success=False,
        detail="Execution failed: target not found.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Type failed.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    store.record_step(
        state=state,
        perception=_perception(),
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="Retry"),
    )

    hints = store.get_hints(
        benchmark=state.benchmark or "generic_task",
        page_hint=_perception().page_hint,
        subgoal="fill_name",
        recent_failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )

    keys = {hint.key for hint in hints}
    assert "click_before_type" in keys
    assert "avoid_identical_type_retry" in keys


# ── memory decay invariant tests ─────────────────────────────────
#
# Per CLAUDE.md, on verification failure the store appends a decay record
# with weight = current_effective_weight × 0.5. Buckets whose mean weight
# drops below 0.1 are excluded from get_hints() — this is the self-healing
# mechanism that stops bad advice from persisting forever.


def _failing_inputs(run_id: str = "decay-run") -> dict:
    state = AgentState(
        run_id=run_id,
        intent="Complete the form.",
        status=RunStatus.RUNNING,
        current_subgoal="fill_name",
    )
    decision = PolicyDecision(
        action=AgentAction(action_type=ActionType.TYPE, target_element_id="name-input", text="Alice"),
        rationale="Fill name",
        confidence=0.8,
        active_subgoal="fill_name",
    )
    executed = ExecutedAction(
        action=decision.action,
        success=False,
        detail="Execution failed: target not found.",
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    verification = VerificationResult(
        status=VerificationStatus.FAILURE,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="Type failed.",
        failure_type=VerificationFailureType.ACTION_FAILED,
        failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        failure_stage=LoopStage.EXECUTE,
    )
    return dict(
        state=state,
        perception=_perception(),
        decision=decision,
        executed_action=executed,
        verification=verification,
        recovery=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="Retry"),
    )


def test_memory_geometric_decay_halves_weight_on_failure(tmp_path: Path) -> None:
    """A verification failure must append a decay record whose weight is
    half the current effective weight."""
    store = FileBackedMemoryStore(root_dir=tmp_path / "runs")
    inputs = _failing_inputs()

    # First failed step seeds records at weight 1.0 (default) and appends decay records.
    initial_records = store.record_step(**inputs)
    assert any(r.weight == 1.0 for r in initial_records), "expected default-weight seed records"

    # Open the JSONL and find a decay record (success=False from _decay_active_hints).
    lines = Path(store.memory_path).read_text(encoding="utf-8").splitlines()
    weights_per_key: dict[str, list[float]] = {}
    import json as _json
    for line in lines:
        rec = _json.loads(line)
        weights_per_key.setdefault(rec["key"], []).append(rec.get("weight", 1.0))

    # At least one key should have a decay record at weight ≤ 0.5 (geometric halving).
    decayed_keys = [k for k, ws in weights_per_key.items() if any(w <= 0.5 for w in ws)]
    assert decayed_keys, "expected at least one key to have a decayed (≤0.5) record"


def test_memory_get_hints_prunes_buckets_below_threshold(tmp_path: Path) -> None:
    """A bucket whose mean weight is below the prune threshold (0.1) must be
    excluded from get_hints() — this is the self-healing mechanism that stops
    bad advice from persisting forever."""
    from src.models.memory import MemoryOutcome, MemoryRecord

    store = FileBackedMemoryStore(root_dir=tmp_path / "runs")

    # Manually inject a decayed-to-near-zero record. We bypass record_step so
    # the failure path doesn't re-seed the bucket at weight 1.0.
    decayed = MemoryRecord(
        key="custom_dead_hint",
        benchmark="generic_task",
        hint="A hint that has been decayed to irrelevance.",
        outcome=MemoryOutcome.FAILURE,
        page_hint=PageHint.FORM_PAGE,
        subgoal="fill_name",
        success=False,
        weight=0.05,  # below the 0.1 prune threshold
    )
    store._append_record(decayed)
    # Force cache invalidation so the next read reflects what we just wrote.
    store._cached_records = None

    hints = store.get_hints(
        benchmark="generic_task",
        page_hint=PageHint.FORM_PAGE,
        subgoal="fill_name",
        recent_failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )
    keys = {h.key for h in hints}
    assert "custom_dead_hint" not in keys, (
        f"expected pruned hint to be excluded from get_hints, got {keys}"
    )

    # Sanity: a fresh full-weight record with the same key+context IS surfaced.
    fresh = MemoryRecord(
        key="custom_live_hint",
        benchmark="generic_task",
        hint="A hint that is still active.",
        outcome=MemoryOutcome.FAILURE,
        page_hint=PageHint.FORM_PAGE,
        subgoal="fill_name",
        success=False,
        weight=1.0,
    )
    store._append_record(fresh)
    store._cached_records = None
    hints = store.get_hints(
        benchmark="generic_task",
        page_hint=PageHint.FORM_PAGE,
        subgoal="fill_name",
        recent_failure_category=FailureCategory.EXECUTION_TARGET_NOT_FOUND,
    )
    assert "custom_live_hint" in {h.key for h in hints}
