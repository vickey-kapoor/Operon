"""Focused tests for local run replay loading and rendering."""

from __future__ import annotations

from pathlib import Path
import shutil

from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts, StepLog
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_logger import append_step_log
from src.store.replay import load_run_replay, render_run_replay


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _debug(stage: str, step_dir: Path) -> ModelDebugArtifacts:
    parsed_name = "policy_decision.json" if stage == "policy" else "perception_parsed.json"
    return ModelDebugArtifacts(
        prompt_artifact_path=str(step_dir / f"{stage}_prompt.txt"),
        raw_response_artifact_path=str(step_dir / f"{stage}_raw.txt"),
        parsed_artifact_path=str(step_dir / parsed_name),
    )



def test_replay_loader_reads_step_logs() -> None:
    root_dir = _local_test_dir("test-replay-loader") / "runs"
    run_id = "run-1"
    step_dir = root_dir / run_id / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(action=action, rationale="Wait for the UI.", confidence=0.3, active_subgoal="wait")
    executed = ExecutedAction(action=action, success=True, detail="waited", artifact_path=str(step_dir / "after.png"))
    verification = VerificationResult(
        status=VerificationStatus.UNCERTAIN,
        expected_outcome_met=False,
        stop_condition_met=False,
        reason="still loading",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="retry", retry_after_ms=1000)
    perception = ScreenPerception(
        summary="Inbox visible",
        capture_artifact_path=str(step_dir / "before.png"),
        visible_elements=[],
    )
    append_step_log(
        root_dir / run_id / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            after_artifact_path=str(step_dir / "after.png"),
            perception_debug=_debug("perception", step_dir),
            policy_debug=_debug("policy", step_dir),
            perception=perception,
            policy_decision=decision,
            executed_action=executed,
            verification_result=verification,
            recovery_decision=recovery,
        ),
    )

    entries = load_run_replay(run_id, root_dir=root_dir)

    assert len(entries) == 1
    assert entries[0].step_id == "step_1"
    assert entries[0].policy_decision.rationale == "Wait for the UI."



def test_replay_renderer_lists_artifact_refs_and_outcomes() -> None:
    root_dir = _local_test_dir("test-replay-renderer") / "runs"
    run_id = "run-2"
    step_dir = root_dir / run_id / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.9, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked", artifact_path=str(step_dir / "after.png"))
    verification = VerificationResult(
        status=VerificationStatus.SUCCESS,
        expected_outcome_met=True,
        stop_condition_met=False,
        reason="compose opened",
    )
    recovery = RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue")
    perception = ScreenPerception(
        summary="Compose button visible",
        capture_artifact_path=str(step_dir / "before.png"),
        visible_elements=[],
    )
    append_step_log(
        root_dir / run_id / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            after_artifact_path=str(step_dir / "after.png"),
            perception_debug=_debug("perception", step_dir),
            policy_debug=_debug("policy", step_dir),
            perception=perception,
            policy_decision=decision,
            executed_action=executed,
            verification_result=verification,
            recovery_decision=recovery,
        ),
    )

    output = render_run_replay(run_id, root_dir=root_dir)

    assert "before.png" in output
    assert "after.png" in output
    assert "perception_prompt.txt" in output
    assert "policy_decision.json" in output
    assert "action: click" in output
    assert "verification: success | compose opened" in output
    assert "recovery: advance | continue" in output
