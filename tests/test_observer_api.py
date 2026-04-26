"""Focused tests for the local debug observer API and UI."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api.server import app
from src.models.common import FailureCategory, LoopStage, RunStatus, StopReason
from src.models.execution import ExecutedAction, ExecutionAttemptTrace, ExecutionTrace
from src.models.logs import (
    FailureRecord,
    ModelDebugArtifacts,
    PreStepFailureLog,
    StepLog,
)
from src.models.perception import (
    ScreenPerception,
    UIElement,
    UIElementNameSource,
    UIElementType,
)
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.progress import ProgressState
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.state import AgentState
from src.models.usage import ModelUsage
from src.models.verification import VerificationResult, VerificationStatus
from src.store.run_logger import append_step_log


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_state(run_dir: Path, state: AgentState) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _debug(stage: str, step_dir: Path) -> ModelDebugArtifacts:
    parsed_name = "policy_decision.json" if stage == "policy" else "perception_parsed.json"
    return ModelDebugArtifacts(
        prompt_artifact_path=str(step_dir / f"{stage}_prompt.txt"),
        raw_response_artifact_path=str(step_dir / f"{stage}_raw.txt"),
        parsed_artifact_path=str(step_dir / parsed_name),
        retry_log_artifact_path=str(step_dir / "perception_retry_log.txt") if stage == "perception" else None,
        selector_trace_artifact_path=str(step_dir / "selector_trace.json") if stage == "policy" else None,
        diagnostics_artifact_path=str(step_dir / "perception_diagnostics.json") if stage == "perception" else None,
    )


def _usage(provider: str, model: str, request_kind: str, *, total_tokens: int, estimated_cost_usd: float) -> ModelUsage:
    return ModelUsage(
        provider=provider,
        model=model,
        request_kind=request_kind,
        input_tokens=max(total_tokens - 100, 0),
        output_tokens=100,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


def test_observer_ui_and_run_snapshot(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer") / "runs"
    browser_root = _local_test_dir("test-observer-browser-artifacts")
    run_id = "run-observer"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    video_path = browser_root / run_id / "session_video" / "session.webm"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fakewebm")
    (step_dir / "before.png").write_bytes(b"fakepng")
    (step_dir / "after.png").write_bytes(b"fakepng")
    (step_dir / "selector_trace.json").write_text(
        """
        [
          {
            "intent": {"action":"click","target_text":"submit","target_role":null,"expected_element_types":["button"],"value_to_type":null,"expected_section":"form"},
            "candidate_count": 1,
            "top_candidates": [{"element_id":"submit-button","element_type":"button","primary_name":"Submit","total_score":95.0,"matched_signals":["exact_primary_name"],"rejected_by":[],"action_compatible":true,"exact_semantic_match":true,"uses_unlabeled_fallback":false,"nearest_matched_text_candidate_id":null,"spatial_grounding_contributed":false,"confidence_band":"high"}],
            "selected_element_id": "submit-button",
            "decision_reason": "accepted",
            "rejection_reason": null,
            "score_margin": 14.0,
            "initial_failure_reason": null,
            "recovery_attempted": false,
            "recovery_strategy_used": null,
            "adjusted_acceptance_threshold": null,
            "adjusted_ambiguity_margin": null,
            "final_decision": "success",
            "final_stop_reason": null,
            "recovery_changed_selected_candidate": false
          }
        ]
        """,
        encoding="utf-8",
    )
    (step_dir / "execution_trace.json").write_text(
        ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="ok",
                    focus_verification_result="focused",
                    verification_result="click_verified",
                )
            ],
            final_outcome="success",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (step_dir / "progress_trace.json").write_text(
        '{"step_index":1,"page_signature":"form_page|none|submit-button:Submit","action_signature":"click|id:submit-button|","target_signature":"id:submit-button","subgoal_signature":"submit_form","failure_signature":null,"blocked_as_redundant":false,"redundancy_reason":null,"loop_pattern_detected":null,"progress_made":true,"no_progress_streak":0,"final_failure_category":null,"final_stop_reason":null}',
        encoding="utf-8",
    )
    (step_dir / "perception_retry_log.txt").write_text("attempt=1 reason=test unlabeled_pct=25.0 usable_count=1 candidate_count=1 salvage_mode=false", encoding="utf-8")

    state = AgentState(
        run_id=run_id,
        intent="Complete the auth-free form and submit it successfully.",
        status=RunStatus.RUNNING,
        step_count=1,
        progress_state=ProgressState(
            completed_targets=["id:submit-button"],
            completed_subgoals=["submit_form"],
            recent_actions=["click|id:submit-button|"],
            recent_failures=[],
            no_progress_streak=0,
            loop_detected=False,
            latest_page_signature="form_page|none|submit-button:Submit",
        ),
        artifact_paths=[str(video_path)],
    )
    _write_state(run_dir, state)
    append_step_log(
        run_dir / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            after_artifact_path=str(step_dir / "after.png"),
            perception_debug=_debug("perception", step_dir),
            policy_debug=_debug("policy", step_dir),
            perception=ScreenPerception(
                summary="Form visible",
                page_hint="form_page",
                capture_artifact_path=str(step_dir / "before.png"),
                visible_elements=[
                    UIElement(
                        element_id="submit-button",
                        element_type=UIElementType.BUTTON,
                        label="Submit",
                        primary_name="Submit",
                        name_source=UIElementNameSource.LABEL,
                        is_unlabeled=False,
                        usable_for_targeting=True,
                        x=20,
                        y=20,
                        width=100,
                        height=30,
                        is_interactable=True,
                        confidence=0.9,
                    )
                ],
            ),
            policy_decision=PolicyDecision(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button"),
                rationale="Submit form.",
                confidence=0.9,
                active_subgoal="submit_form",
            ),
            executed_action=ExecutedAction(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="submit-button"),
                success=True,
                detail="clicked",
                execution_trace=ExecutionTrace(
                    attempts=[
                        ExecutionAttemptTrace(
                            attempt_index=1,
                            revalidation_result="ok",
                            focus_verification_result="focused",
                            verification_result="click_verified",
                        )
                    ],
                    final_outcome="success",
                ),
                execution_trace_artifact_path=str(step_dir / "execution_trace.json"),
            ),
            verification_result=VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=False,
                reason="ok",
            ),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue"),
            progress_state=state.progress_state,
            progress_trace_artifact_path=str(step_dir / "progress_trace.json"),
        ),
    )

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    monkeypatch.setenv("OPERON_BROWSER_ARTIFACTS_ROOT", str(browser_root))
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Command Center" in page.text

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Billing period" in dashboard.text

    runs = client.get("/observer/api/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["run_id"] == run_id

    snapshot = client.get(f"/observer/api/run/{run_id}")
    body = snapshot.json()
    assert snapshot.status_code == 200
    assert body["run"]["run_id"] == run_id
    assert body["run"]["current_phase"] == "recover"
    assert body["run"]["session_video_path"] == str(video_path)
    assert body["current_step"]["selector"]["trace"]["selected_candidate"] == "submit-button"
    assert body["current_step"]["execution"]["trace"]["final_outcome"] == "success"
    assert body["current_step"]["plan"]["rationale"] == "Submit form."
    assert body["current_step"]["plan"]["proposed_action"]["action_type"] == "click"
    assert body["progress_state"]["completed_targets"] == ["id:submit-button"]

    artifact = client.get(f"/observer/api/artifact?path={step_dir / 'before.png'}")
    assert artifact.status_code == 200
    video_artifact = client.get(f"/observer/api/artifact?path={video_path}")
    assert video_artifact.status_code == 200


def test_observer_export_run_bundle(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-export") / "runs"
    browser_root = _local_test_dir("test-observer-export-browser")
    run_id = "run-export"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "before.png").write_bytes(b"fakepng")
    (run_dir / "state.json").write_text(
        AgentState(
            run_id=run_id,
            intent="Open Chrome and navigate to example.com",
            status=RunStatus.SUCCEEDED,
            step_count=1,
            artifact_paths=[str(browser_root / run_id / "session_video" / "session.webm")],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    video_path = browser_root / run_id / "session_video" / "session.webm"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fakewebm")

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    monkeypatch.setenv("OPERON_BROWSER_ARTIFACTS_ROOT", str(browser_root))
    client = TestClient(app)

    resp = client.get(f"/observer/api/export/{run_id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"PK")


def test_observer_live_browser_frame() -> None:
    """Observer live browser endpoint should return a PNG for active sessions."""
    class _Executor:
        async def live_frame_png(self, run_id: str) -> bytes | None:
            assert run_id == "live-run"
            return b"\x89PNG\r\n\x1a\nlive"

    from unittest.mock import patch

    client = TestClient(app)
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.executor = _Executor()
        resp = client.get("/observer/api/live-browser/live-run")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


def test_observer_live_browser_frame_missing_session() -> None:
    """Observer live browser endpoint should return 404 when no active session exists."""
    class _Executor:
        async def live_frame_png(self, run_id: str) -> bytes | None:
            return None

    from unittest.mock import patch

    client = TestClient(app)
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.executor = _Executor()
        resp = client.get("/observer/api/live-browser/missing-run")

    assert resp.status_code == 404


def test_observer_run_includes_current_browser_url(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-current-url") / "runs"
    run_id = "run-current-url"
    run_dir = root_dir / run_id
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Open Chrome and navigate to example.com",
            status=RunStatus.RUNNING,
            step_count=0,
        ),
    )

    class _Executor:
        async def current_url_for_run(self, requested_run_id: str) -> str | None:
            assert requested_run_id == run_id
            return "https://example.com/"

    from unittest.mock import patch

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.executor = _Executor()
        resp = client.get(f"/observer/api/run/{run_id}")

    assert resp.status_code == 200
    assert resp.json()["run"]["current_url"] == "https://example.com/"


def test_observer_run_reconciles_orphaned_browser_run(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-orphaned-run") / "runs"
    run_id = "run-orphaned-browser"
    run_dir = root_dir / run_id
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Open Chrome and navigate to example.com",
            status=RunStatus.RUNNING,
            step_count=1,
        ),
    )

    class _Executor:
        async def current_url_for_run(self, requested_run_id: str) -> str | None:
            assert requested_run_id == run_id
            return None

    from unittest.mock import patch

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.executor = _Executor()
        resp = client.get(f"/observer/api/run/{run_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["status"] == RunStatus.FAILED.value
    assert body["run"]["stop_reason"] == StopReason.EXECUTION_NO_PROGRESS.value


def test_observer_runs_reconciles_orphaned_browser_runs(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-orphaned-list") / "runs"
    orphaned_run_id = "run-orphaned-list"
    active_run_id = "run-active-list"
    _write_state(
        root_dir / orphaned_run_id,
        AgentState(
            run_id=orphaned_run_id,
            intent="Open Chrome and navigate to example.com",
            status=RunStatus.RUNNING,
            step_count=1,
        ),
    )
    _write_state(
        root_dir / active_run_id,
        AgentState(
            run_id=active_run_id,
            intent="Open Chrome and navigate to example.com",
            status=RunStatus.RUNNING,
            step_count=2,
        ),
    )

    class _Executor:
        async def current_url_for_run(self, requested_run_id: str) -> str | None:
            if requested_run_id == orphaned_run_id:
                return None
            if requested_run_id == active_run_id:
                return "https://example.com/"
            return None

    from unittest.mock import patch

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.executor = _Executor()
        resp = client.get("/observer/api/runs?limit=10")

    assert resp.status_code == 200
    runs = {run["run_id"]: run for run in resp.json()["runs"]}
    assert runs[orphaned_run_id]["status"] == RunStatus.FAILED.value
    assert runs[orphaned_run_id]["stop_reason"] == StopReason.EXECUTION_NO_PROGRESS.value
    assert runs[active_run_id]["status"] == RunStatus.RUNNING.value


def test_observer_handles_missing_artifacts_gracefully(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-missing") / "runs"
    run_id = "run-missing"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "before.png").write_bytes(b"fakepng")
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.FAILED,
            step_count=1,
        ),
    )
    append_step_log(
        run_dir / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            after_artifact_path=str(step_dir / "after.png"),
            perception_debug=_debug("perception", step_dir),
            policy_debug=_debug("policy", step_dir),
            perception=ScreenPerception(summary="Form visible", page_hint="form_page", capture_artifact_path=str(step_dir / "before.png"), visible_elements=[]),
            policy_decision=PolicyDecision(
                action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000),
                rationale="Wait.",
                confidence=0.5,
                active_subgoal="wait",
            ),
            executed_action=ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000), success=False, detail="waited"),
            verification_result=VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="failed",
            ),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.STOP, message="stop"),
            progress_state=ProgressState(no_progress_streak=1, loop_detected=False),
            progress_trace_artifact_path=str(step_dir / "missing_progress_trace.json"),
        ),
    )

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    snapshot = client.get(f"/observer/api/run/{run_id}")
    body = snapshot.json()

    assert snapshot.status_code == 200
    assert body["current_step"]["selector"]["trace"] is None
    assert body["current_step"]["execution"]["trace"] is None
    assert body["current_step"]["progress"]["trace"] is None

    missing = client.get("/observer/api/artifact?path=C:\\not-under-runs\\x.png")
    assert missing.status_code == 400


def test_observer_snapshot_includes_failed_perception_diagnostics(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-prestep") / "runs"
    run_id = "run-prestep"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "before.png").write_bytes(b"fakepng")
    (step_dir / "perception_raw.txt").write_text('{"summary":"Weak form page."}', encoding="utf-8")
    (step_dir / "perception_retry_log.txt").write_text(
        "attempt=1 reason=missing_critical_semantics interactive_count=3 text_count=2 usable_count=0 candidate_count=0 salvage_mode=false\n"
        "attempt=1 reason=zero usable candidates after salvage interactive_count=3 text_count=2 usable_count=3 candidate_count=0 salvage_mode=true",
        encoding="utf-8",
    )
    (step_dir / "perception_parsed.json").write_text(
        """
        {
          "summary":"Weak form page.",
          "page_hint":"form_page",
          "capture_artifact_path":"runs/run-prestep/step_1/before.png",
          "focused_element_id":null,
          "confidence":0.41,
          "visible_elements":[
            {"element_id":"name-label","element_type":"text","label":null,"text":"Name","placeholder":null,"name":null,"role":null,"primary_name":"Name","name_source":"text","is_unlabeled":false,"usable_for_targeting":false,"x":100,"y":100,"width":60,"height":20,"is_interactable":false,"confidence":0.9},
            {"element_id":"name-input","element_type":"input","label":null,"text":null,"placeholder":null,"name":null,"role":null,"primary_name":"unlabeled_input","name_source":"synthetic","is_unlabeled":true,"usable_for_targeting":true,"x":100,"y":130,"width":240,"height":28,"is_interactable":true,"confidence":0.49}
          ]
        }
        """,
        encoding="utf-8",
    )
    (step_dir / "perception_diagnostics.json").write_text(
        """
        {
          "summary":"Weak form page.",
          "page_hint":"form_page",
          "quality_gate_reason":"missing critical semantic fields for form targeting",
          "salvage_attempted":true,
          "salvage_reason":"zero usable candidates after salvage",
          "final_decision":"aborted_low_quality",
          "raw_response_artifact_path":"runs/run-prestep/step_1/perception_raw.txt",
          "parsed_artifact_path":"runs/run-prestep/step_1/perception_parsed.json",
          "retry_log_artifact_path":"runs/run-prestep/step_1/perception_retry_log.txt",
          "quality_metrics":{
            "total_elements":3,
            "labeled_elements":2,
            "unlabeled_elements":1,
            "usable_count":0,
            "candidate_count":0,
            "interactive_count":1,
            "text_count":2,
            "labeled_interactive_count":0,
            "unlabeled_interactive_count":1,
            "spatially_groundable_interactive_count":1,
            "salvage_mode":false
          },
          "normalized_raw_perception_summary":{
            "summary":"Weak form page.",
            "page_hint":"form_page",
            "focused_element_id":null,
            "element_count":3
          },
          "salvage_result":{
            "summary":"Weak form page.",
            "page_hint":"form_page",
            "focused_element_id":null,
            "element_count":3,
            "quality_metrics":{
              "total_elements":3,
              "labeled_elements":2,
              "unlabeled_elements":1,
              "usable_count":1,
              "candidate_count":1,
              "interactive_count":1,
              "text_count":2,
              "labeled_interactive_count":0,
              "unlabeled_interactive_count":1,
              "spatially_groundable_interactive_count":1,
              "salvage_mode":true
            }
          }
        }
        """,
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.FAILED,
            step_count=0,
        ),
    )
    append_step_log(
        run_dir / "run.jsonl",
        PreStepFailureLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(step_dir / "before.png"),
            perception_debug=_debug("perception", step_dir),
            failure=FailureRecord(
                category=FailureCategory.PERCEPTION_LOW_QUALITY,
                stage=LoopStage.PERCEIVE,
                retry_count=0,
                terminal=True,
                recoverable=False,
                reason="Gemini perception output was low quality: zero usable candidates after salvage",
                stop_reason=StopReason.PERCEPTION_LOW_QUALITY,
            ),
            error_message="Gemini perception output was low quality: zero usable candidates after salvage",
        ),
    )

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    snapshot = client.get(f"/observer/api/run/{run_id}")
    body = snapshot.json()

    assert snapshot.status_code == 200
    assert body["current_step"]["perception"]["metrics"]["interactive_count"] == 1
    assert body["current_step"]["perception"]["metrics"]["text_count"] == 2
    assert body["current_step"]["perception"]["metrics"]["salvage_mode_triggered"] is True
    assert body["current_step"]["perception"]["metrics"]["quality_gate_failure_reason"] == "zero usable candidates after salvage"
    assert body["current_step"]["perception"]["metrics"]["raw_response_artifact_path"].endswith("perception_raw.txt")
    assert body["current_step"]["perception"]["elements"][0]["element_id"] == "name-label"


def test_observer_shows_in_progress_perception_before_step_commit(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-partial-perception") / "runs"
    run_id = "run-partial-perception"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "before.png").write_bytes(b"fakepng")
    (step_dir / "perception_parsed.json").write_text(
        """
        {
          "summary":"Form page visible.",
          "page_hint":"form_page",
          "capture_artifact_path":"runs/run-partial-perception/step_1/before.png",
          "focused_element_id":null,
          "confidence":0.88,
          "visible_elements":[
            {"element_id":"name-label","element_type":"text","label":"Name","text":null,"placeholder":null,"name":null,"role":null,"primary_name":"Name","name_source":"label","is_unlabeled":false,"usable_for_targeting":false,"x":24,"y":24,"width":60,"height":20,"is_interactable":false,"confidence":0.9},
            {"element_id":"name-input","element_type":"input","label":"Name","text":null,"placeholder":null,"name":null,"role":null,"primary_name":"Name","name_source":"label","is_unlabeled":false,"usable_for_targeting":true,"x":24,"y":52,"width":200,"height":28,"is_interactable":true,"confidence":0.9}
          ]
        }
        """,
        encoding="utf-8",
    )
    (step_dir / "perception_diagnostics.json").write_text(
        """
        {
          "summary":"Form page visible.",
          "page_hint":"form_page",
          "quality_gate_reason":null,
          "salvage_attempted":false,
          "salvage_reason":null,
          "final_decision":"accepted",
          "raw_response_artifact_path":"runs/run-partial-perception/step_1/perception_raw.txt",
          "quality_metrics":{"total_elements":2,"labeled_elements":2,"unlabeled_elements":0,"usable_count":1,"candidate_count":1,"interactive_count":1,"text_count":1,"labeled_interactive_count":1,"unlabeled_interactive_count":0,"salvage_mode":false}
        }
        """,
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.RUNNING,
            step_count=1,
        ),
    )
    (run_dir / "run.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    snapshot = client.get(f"/observer/api/run/{run_id}")
    body = snapshot.json()

    assert snapshot.status_code == 200
    assert body["current_step"]["is_partial"] is True
    assert body["run"]["current_phase"] == "choose"
    assert body["current_step"]["perception"]["summary"] == "Form page visible."
    assert body["current_step"]["perception"]["metrics"]["candidate_count"] == 1
    assert body["current_step"]["selector"]["trace"] is None


def test_observer_shows_partial_selector_and_execution_before_step_commit(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-partial-selector") / "runs"
    run_id = "run-partial-selector"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_2"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "before.png").write_bytes(b"fakepng")
    (step_dir / "perception_parsed.json").write_text(
        """
        {
          "summary":"Form page visible.",
          "page_hint":"form_page",
          "capture_artifact_path":"runs/run-partial-selector/step_2/before.png",
          "focused_element_id":null,
          "confidence":0.88,
          "visible_elements":[
            {"element_id":"submit-button","element_type":"button","label":"Submit","text":null,"placeholder":null,"name":null,"role":null,"primary_name":"Submit","name_source":"label","is_unlabeled":false,"usable_for_targeting":true,"x":24,"y":52,"width":120,"height":28,"is_interactable":true,"confidence":0.9}
          ]
        }
        """,
        encoding="utf-8",
    )
    (step_dir / "selector_trace.json").write_text(
        """
        [{
          "intent":{"action":"click","target_text":"submit","target_role":null,"expected_element_types":["button"],"value_to_type":null,"expected_section":"form"},
          "candidate_count":1,
          "top_candidates":[{"element_id":"submit-button","total_score":95.0,"matched_signals":["exact_primary_name"]}],
          "selected_element_id":"submit-button",
          "score_margin":14.0,
          "recovery_attempted":false,
          "recovery_strategy_used":null,
          "final_decision":"success",
          "rejection_reason":null,
          "initial_failure_reason":null
        }]
        """,
        encoding="utf-8",
    )
    (step_dir / "execution_trace.json").write_text(
        """
        {
          "action":{"action_type":"click","target_element_id":"submit-button"},
          "retry_attempted":false,
          "final_outcome":"success"
        }
        """,
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Complete the auth-free form and submit it successfully.",
            status=RunStatus.RUNNING,
            step_count=2,
        ),
    )
    (run_dir / "run.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    snapshot = client.get(f"/observer/api/run/{run_id}")
    body = snapshot.json()

    assert snapshot.status_code == 200
    assert body["current_step"]["is_partial"] is True
    assert body["run"]["current_phase"] == "verify"
    assert body["current_step"]["selector"]["trace"]["selected_candidate"] == "submit-button"
    assert body["current_step"]["execution"]["trace"]["final_outcome"] == "success"


def test_observer_usage_dashboard_aggregates_run_and_model_costs(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-usage") / "runs"
    run_id = "run-usage"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    before_path = step_dir / "before.png"
    after_path = step_dir / "after.png"
    before_path.write_bytes(b"fakepng")
    after_path.write_bytes(b"fakepng")

    perception_debug = _debug("perception", step_dir).model_copy(
        update={
            "usage_artifact_path": str(step_dir / "perception_usage.json"),
            "usage": _usage("gemini", "gemini-3-flash-preview", "image", total_tokens=1200, estimated_cost_usd=0.0012),
        }
    )
    policy_debug = _debug("policy", step_dir).model_copy(
        update={
            "usage_artifact_path": str(step_dir / "policy_usage.json"),
            "usage": _usage("anthropic", "claude-sonnet-4-20250514", "text", total_tokens=900, estimated_cost_usd=0.0045),
        }
    )
    verification_debug = ModelDebugArtifacts(
        prompt_artifact_path=str(step_dir / "verification_prompt.txt"),
        raw_response_artifact_path=str(step_dir / "verification_raw.txt"),
        parsed_artifact_path=str(step_dir / "verification_result.json"),
        usage_artifact_path=str(step_dir / "verification_usage.json"),
        usage=_usage("anthropic", "claude-sonnet-4-20250514", "image", total_tokens=600, estimated_cost_usd=0.003),
    )
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Check usage aggregation",
            status=RunStatus.SUCCEEDED,
            step_count=1,
        ),
    )
    append_step_log(
        run_dir / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(before_path),
            after_artifact_path=str(after_path),
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            verification_debug=verification_debug,
            perception=ScreenPerception(summary="Observed", page_hint="unknown", capture_artifact_path=str(before_path), visible_elements=[]),
            policy_decision=PolicyDecision(action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000), rationale="Wait.", confidence=0.5, active_subgoal="wait"),
            executed_action=ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000), success=True, detail="waited"),
            verification_result=VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="ok"),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue"),
        ),
    )

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)

    snapshot = client.get(f"/observer/api/run/{run_id}")
    assert snapshot.status_code == 200
    assert snapshot.json()["usage"]["request_count"] == 3
    assert snapshot.json()["usage"]["estimated_cost_usd"] == 0.0087

    usage = client.get("/observer/api/usage?limit=10")
    assert usage.status_code == 200
    body = usage.json()
    assert body["summary"]["request_count"] == 3
    assert body["summary"]["by_model"]["gemini-3-flash-preview"]["request_count"] == 1
    assert body["summary"]["by_model"]["claude-sonnet-4-20250514"]["request_count"] == 2
    assert body["summary"]["estimated_cost_usd"] == 0.0087


def test_observer_usage_dashboard_deduplicates_combined_backend_calls(monkeypatch) -> None:
    root_dir = _local_test_dir("test-observer-usage-combined") / "runs"
    run_id = "run-usage-combined"
    run_dir = root_dir / run_id
    step_dir = run_dir / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)
    before_path = step_dir / "before.png"
    after_path = step_dir / "after.png"
    before_path.write_bytes(b"fakepng")
    after_path.write_bytes(b"fakepng")

    shared_usage = _usage("gemini", "gemini-3-flash-preview", "image", total_tokens=777, estimated_cost_usd=0.000777)
    perception_debug = ModelDebugArtifacts(
        prompt_artifact_path=str(step_dir / "combined_prompt.txt"),
        raw_response_artifact_path=str(step_dir / "combined_raw.txt"),
        parsed_artifact_path=str(step_dir / "combined_parsed.json"),
        usage_artifact_path=str(step_dir / "combined_usage.json"),
        usage=shared_usage,
    )
    policy_debug = perception_debug.model_copy()
    _write_state(
        run_dir,
        AgentState(
            run_id=run_id,
            intent="Check combined usage dedupe",
            status=RunStatus.SUCCEEDED,
            step_count=1,
        ),
    )
    append_step_log(
        run_dir / "run.jsonl",
        StepLog(
            run_id=run_id,
            step_id="step_1",
            step_index=1,
            before_artifact_path=str(before_path),
            after_artifact_path=str(after_path),
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            perception=ScreenPerception(summary="Observed", page_hint="unknown", capture_artifact_path=str(before_path), visible_elements=[]),
            policy_decision=PolicyDecision(action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000), rationale="Wait.", confidence=0.5, active_subgoal="wait"),
            executed_action=ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000), success=True, detail="waited"),
            verification_result=VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="ok"),
            recovery_decision=RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="continue"),
        ),
    )

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(root_dir))
    client = TestClient(app)
    usage = client.get("/observer/api/usage?limit=10")

    assert usage.status_code == 200
    assert usage.json()["summary"]["request_count"] == 1
    assert usage.json()["summary"]["estimated_cost_usd"] == 0.000777
