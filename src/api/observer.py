"""Helpers for the local real-time debug observer UI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.models.logs import PreStepFailureLog, StepLog
from src.models.perception import UIElementType
from src.store.replay import load_run_replay
from src.store.summary import _load_state_from_path


def runs_root() -> Path:
    """Return the root directory used for local run artifacts."""
    return Path(os.getenv("OPERON_RUNS_ROOT", "runs")).resolve()


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent runs ordered by state file timestamp."""
    root = runs_root()
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True):
        state_path = run_dir / "state.json"
        if not state_path.exists():
            continue
        try:
            state = _load_state_from_path(state_path)
        except Exception:
            continue
        runs.append(
            {
                "run_id": state.run_id,
                "intent": state.intent,
                "status": state.status.value,
                "step_count": state.step_count,
                "stop_reason": state.stop_reason.value if state.stop_reason is not None else None,
                "updated_at": state_path.stat().st_mtime,
            }
        )
        if len(runs) >= limit:
            break
    return runs


def load_run_snapshot(run_id: str) -> dict[str, Any]:
    """Load one run plus current and historical step observer data."""
    root = runs_root()
    state_path = root / run_id / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Run state not found: {state_path}")

    state = _load_state_from_path(state_path)
    entries = load_run_replay(run_id, root_dir=root)
    run_dir = root / run_id
    completed_steps = [entry for entry in entries if isinstance(entry, StepLog)]
    pre_step_failures = [entry for entry in entries if isinstance(entry, PreStepFailureLog)]
    steps = [_step_payload(entry) for entry in completed_steps]
    partial_current_step = _latest_partial_step_payload(run_dir)
    current_step = partial_current_step or (steps[-1] if steps else (_pre_step_payload(pre_step_failures[-1]) if pre_step_failures else None))
    phase = _current_phase(state, current_step, pre_step_failures[-1] if pre_step_failures else None)
    event_log = _build_event_log(state, completed_steps, pre_step_failures, partial_current_step)
    if partial_current_step is not None and not any(step["step_index"] == partial_current_step["step_index"] for step in steps):
        steps.append(partial_current_step)
        steps.sort(key=lambda item: item["step_index"])
    return {
        "run": {
            "run_id": state.run_id,
            "intent": state.intent,
            "status": state.status.value,
            "step_count": state.step_count,
            "stop_reason": state.stop_reason.value if state.stop_reason is not None else None,
            "current_subgoal": state.current_subgoal,
            "current_task_id": _task_id_from_intent(state.intent),
            "current_phase": phase,
        },
        "progress_state": state.progress_state.model_dump(mode="json"),
        "current_step": current_step,
        "steps": steps,
        "pre_step_failures": [_pre_step_payload(entry) for entry in pre_step_failures],
        "event_log": event_log,
    }


def artifact_path_for_request(path_value: str) -> Path:
    """Resolve and validate an artifact path for local observer access."""
    root = runs_root()
    requested = Path(path_value)
    candidate = requested if requested.is_absolute() else (Path.cwd() / requested)
    resolved = candidate.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("Artifact path is outside the runs root")
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact not found: {resolved}")
    return resolved


def _step_payload(entry: StepLog) -> dict[str, Any]:
    before_dimensions = _image_dimensions(entry.before_artifact_path)
    perception_diagnostics = _load_json(entry.perception_debug.diagnostics_artifact_path)
    return {
        "step_id": entry.step_id,
        "step_index": entry.step_index,
        "before_artifact_path": entry.before_artifact_path,
        "after_artifact_path": entry.after_artifact_path,
        "before_dimensions": before_dimensions,
        "page_hint": entry.perception.page_hint.value,
        "perception": {
            "summary": entry.perception.summary,
            "focused_element_id": entry.perception.focused_element_id,
            "confidence": entry.perception.confidence,
            "metrics": _perception_metrics(entry),
            "elements": [element.model_dump(mode="json") for element in entry.perception.visible_elements],
            "retry_log": _read_text(entry.perception_debug.retry_log_artifact_path),
            "diagnostics": perception_diagnostics,
        },
        "selector": {
            "intent": None,
            "candidate_count": None,
            "top_candidates": [],
            "selected_candidate": None,
            "score_margin": None,
            "recovery_attempted": False,
            "recovery_strategy_used": None,
            "final_decision": None,
            "failure_reason": None,
            "trace": _selector_trace_payload(entry.policy_debug.selector_trace_artifact_path),
        },
        "execution": {
            "action": entry.executed_action.action.model_dump(mode="json"),
            "detail": entry.executed_action.detail,
            "failure_category": entry.executed_action.failure_category.value if entry.executed_action.failure_category is not None else None,
            "trace": _load_json(entry.executed_action.execution_trace_artifact_path),
        },
        "verification": entry.verification_result.model_dump(mode="json"),
        "recovery": entry.recovery_decision.model_dump(mode="json"),
        "progress": {
            "state": entry.progress_state.model_dump(mode="json") if entry.progress_state is not None else None,
            "trace": _load_json(entry.progress_trace_artifact_path),
        },
        "failure": entry.failure.model_dump(mode="json") if entry.failure is not None else None,
        "is_partial": False,
    }


def _pre_step_payload(entry: PreStepFailureLog) -> dict[str, Any]:
    diagnostics = _load_json(entry.perception_debug.diagnostics_artifact_path) or {}
    parsed_perception = _load_json(entry.perception_debug.parsed_artifact_path) or {}
    visible_elements = parsed_perception.get("visible_elements", []) if isinstance(parsed_perception, dict) else []
    return {
        "step_id": entry.step_id,
        "step_index": entry.step_index,
        "before_artifact_path": entry.before_artifact_path,
        "before_dimensions": _image_dimensions(entry.before_artifact_path),
        "phase": entry.failure.stage.value,
        "error_message": entry.error_message,
        "failure": entry.failure.model_dump(mode="json"),
        "perception_retry_log": _read_text(entry.perception_debug.retry_log_artifact_path),
        "perception": {
            "summary": diagnostics.get("summary") or parsed_perception.get("summary"),
            "focused_element_id": diagnostics.get("normalized_raw_perception_summary", {}).get("focused_element_id")
            if isinstance(diagnostics.get("normalized_raw_perception_summary"), dict)
            else parsed_perception.get("focused_element_id"),
            "confidence": parsed_perception.get("confidence"),
            "metrics": _pre_step_perception_metrics(diagnostics),
            "elements": visible_elements,
            "retry_log": _read_text(entry.perception_debug.retry_log_artifact_path),
            "diagnostics": diagnostics or None,
        },
        "selector": {"trace": None},
        "execution": None,
        "progress": {"state": None, "trace": None},
        "is_partial": False,
    }


def _latest_partial_step_payload(run_dir: Path) -> dict[str, Any] | None:
    latest_step_dir = _latest_step_dir(run_dir)
    if latest_step_dir is None:
        return None
    return _partial_step_payload(latest_step_dir)


def _latest_step_dir(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"step_(\d+)", child.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _partial_step_payload(step_dir: Path) -> dict[str, Any] | None:
    step_index = int(step_dir.name.split("_", 1)[1])
    before_artifact_path = _existing_path(step_dir / "before.png")
    if before_artifact_path is None:
        return None

    perception = _partial_perception_payload(step_dir, before_artifact_path)
    selector_trace = _selector_trace_payload(_existing_path(step_dir / "selector_trace.json"))
    execution_trace = _load_json(_existing_path(step_dir / "execution_trace.json"))
    progress_trace_path = _existing_path(step_dir / "progress_trace.json")
    progress_trace = _load_json(progress_trace_path)
    return {
        "step_id": f"step_{step_index}",
        "step_index": step_index,
        "before_artifact_path": before_artifact_path,
        "after_artifact_path": _existing_path(step_dir / "after.png"),
        "before_dimensions": _image_dimensions(before_artifact_path),
        "page_hint": (perception or {}).get("metrics", {}).get("page_hint"),
        "phase": _partial_step_phase(step_dir),
        "perception": perception,
        "selector": {
            "intent": None,
            "candidate_count": None,
            "top_candidates": [],
            "selected_candidate": None,
            "score_margin": None,
            "recovery_attempted": False,
            "recovery_strategy_used": None,
            "final_decision": None,
            "failure_reason": None,
            "trace": selector_trace,
        },
        "execution": _partial_execution_payload(execution_trace),
        "verification": None,
        "recovery": None,
        "progress": {
            "state": None,
            "trace": progress_trace,
        },
        "failure": None,
        "is_partial": True,
    }


def _partial_perception_payload(step_dir: Path, before_artifact_path: str) -> dict[str, Any] | None:
    parsed_perception = _load_json(_existing_path(step_dir / "perception_parsed.json"))
    diagnostics = _load_json(_existing_path(step_dir / "perception_diagnostics.json"))
    retry_log = _read_text(_existing_path(step_dir / "perception_retry_log.txt"))
    if parsed_perception is None and diagnostics is None and retry_log is None:
        return None

    if isinstance(parsed_perception, dict):
        elements = parsed_perception.get("visible_elements", [])
        summary = parsed_perception.get("summary")
        focused_element_id = parsed_perception.get("focused_element_id")
        confidence = parsed_perception.get("confidence")
        metrics = _perception_metrics_from_elements(elements, parsed_perception.get("page_hint"), retry_log)
    else:
        elements = []
        summary = diagnostics.get("summary") if isinstance(diagnostics, dict) else None
        focused_element_id = None
        confidence = None
        metrics = _pre_step_perception_metrics(diagnostics or {})

    if isinstance(diagnostics, dict):
        metrics = {**metrics, **{k: v for k, v in _pre_step_perception_metrics(diagnostics).items() if v is not None}}

    return {
        "summary": summary,
        "focused_element_id": focused_element_id,
        "confidence": confidence,
        "metrics": metrics,
        "elements": elements,
        "retry_log": retry_log,
        "diagnostics": diagnostics,
        "capture_artifact_path": before_artifact_path,
    }


def _partial_execution_payload(execution_trace: Any) -> dict[str, Any] | None:
    if execution_trace is None:
        return None
    action = execution_trace.get("action") if isinstance(execution_trace, dict) else None
    failure_category = execution_trace.get("failure_category") if isinstance(execution_trace, dict) else None
    return {
        "action": action,
        "detail": execution_trace.get("final_outcome") if isinstance(execution_trace, dict) else None,
        "failure_category": failure_category,
        "trace": execution_trace,
    }


def _perception_metrics(entry: StepLog) -> dict[str, Any]:
    elements = entry.perception.visible_elements
    retry_log = _read_text(entry.perception_debug.retry_log_artifact_path) or ""
    return _perception_metrics_from_elements(
        [element.model_dump(mode="json") for element in elements],
        entry.perception.page_hint.value,
        retry_log,
    )


def _perception_metrics_from_elements(elements: list[dict[str, Any]], page_hint: Any, retry_log: str | None) -> dict[str, Any]:
    total = len(elements)
    unlabeled = sum(1 for element in elements if bool(element.get("is_unlabeled")))
    usable = sum(1 for element in elements if bool(element.get("usable_for_targeting")))
    interactive = sum(
        1
        for element in elements
        if bool(element.get("is_interactable")) and element.get("element_type") in {"input", "button", "link"}
    )
    text_count = sum(1 for element in elements if element.get("element_type") == "text")
    labeled_interactive_count = sum(
        1
        for element in elements
        if bool(element.get("is_interactable"))
        and element.get("element_type") in {"input", "button", "link"}
        and not bool(element.get("is_unlabeled"))
    )
    return {
        "total_elements": total,
        "labeled_elements": total - unlabeled,
        "unlabeled_elements": unlabeled,
        "usable_elements": usable,
        "interactive_count": interactive,
        "text_count": text_count,
        "labeled_interactive_count": labeled_interactive_count,
        "unlabeled_interactive_count": max(interactive - labeled_interactive_count, 0),
        "page_hint": page_hint,
        "perception_retry_occurred": "attempt=" in (retry_log or ""),
        "salvage_mode_triggered": "salvage_mode=true" in (retry_log or ""),
    }


def _pre_step_perception_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    quality_metrics = diagnostics.get("quality_metrics", {}) if isinstance(diagnostics, dict) else {}
    salvage_result = diagnostics.get("salvage_result", {}) if isinstance(diagnostics, dict) else {}
    salvage_metrics = salvage_result.get("quality_metrics", {}) if isinstance(salvage_result, dict) else {}
    active_metrics = salvage_metrics or quality_metrics
    return {
        "total_elements": active_metrics.get("total_elements"),
        "labeled_elements": active_metrics.get("labeled_elements"),
        "unlabeled_elements": active_metrics.get("unlabeled_elements"),
        "usable_elements": active_metrics.get("usable_count"),
        "interactive_count": active_metrics.get("interactive_count"),
        "text_count": active_metrics.get("text_count"),
        "labeled_interactive_count": active_metrics.get("labeled_interactive_count"),
        "unlabeled_interactive_count": active_metrics.get("unlabeled_interactive_count"),
        "candidate_count": active_metrics.get("candidate_count"),
        "page_hint": diagnostics.get("page_hint"),
        "perception_retry_occurred": bool(diagnostics.get("quality_gate_reason")),
        "salvage_mode_triggered": bool(diagnostics.get("salvage_attempted")),
        "quality_gate_failure_reason": diagnostics.get("salvage_reason") or diagnostics.get("quality_gate_reason"),
        "raw_response_artifact_path": diagnostics.get("raw_response_artifact_path"),
        "final_decision": diagnostics.get("final_decision"),
    }


def _selector_trace_payload(path_value: str | None) -> dict[str, Any] | None:
    payload = _load_json(path_value)
    if payload is None:
        return None
    traces = payload if isinstance(payload, list) else [payload]
    if not traces:
        return None
    current = traces[-1]
    top_candidates = current.get("top_candidates", [])
    return {
        "intent": current.get("intent"),
        "candidate_count": current.get("candidate_count"),
        "top_candidates": top_candidates[:3],
        "selected_candidate": current.get("selected_element_id"),
        "score_margin": current.get("score_margin"),
        "recovery_attempted": current.get("recovery_attempted", False),
        "recovery_strategy_used": current.get("recovery_strategy_used"),
        "final_decision": current.get("final_decision"),
        "failure_reason": current.get("rejection_reason") or current.get("initial_failure_reason"),
    }


def _build_event_log(
    state,
    completed_steps: list[StepLog],
    pre_step_failures: list[PreStepFailureLog],
    partial_current_step: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"step_index": 0, "event": "run_started", "detail": state.intent},
    ]
    for entry in completed_steps:
        events.extend(
            [
                {"step_index": entry.step_index, "event": "screenshot_captured", "detail": entry.before_artifact_path},
                {"step_index": entry.step_index, "event": "perception_requested", "detail": entry.perception.summary},
            ]
        )
        retry_log = _read_text(entry.perception_debug.retry_log_artifact_path) or ""
        if retry_log:
            for line in retry_log.splitlines():
                if line.strip():
                    events.append({"step_index": entry.step_index, "event": "perception_retry", "detail": line})
        if entry.policy_debug.selector_trace_artifact_path:
            events.append({"step_index": entry.step_index, "event": "selector_trace_written", "detail": entry.policy_debug.selector_trace_artifact_path})
        selector = _selector_trace_payload(entry.policy_debug.selector_trace_artifact_path)
        if selector and selector["recovery_attempted"]:
            events.append(
                {
                    "step_index": entry.step_index,
                    "event": "selector_recovery_used" if selector["final_decision"] == "success" else "selector_recovery_failed",
                    "detail": selector["recovery_strategy_used"],
                }
            )
        if entry.executed_action.execution_trace is not None and entry.executed_action.execution_trace.retry_attempted:
            events.append(
                {
                    "step_index": entry.step_index,
                    "event": "execution_retry",
                    "detail": entry.executed_action.execution_trace.retry_reason.value if entry.executed_action.execution_trace.retry_reason is not None else "unknown",
                }
            )
        progress_trace = _load_json(entry.progress_trace_artifact_path) or {}
        if progress_trace.get("blocked_as_redundant"):
            events.append(
                {
                    "step_index": entry.step_index,
                    "event": "progress_blocked_action",
                    "detail": progress_trace.get("redundancy_reason"),
                }
            )
    for entry in pre_step_failures:
        events.append({"step_index": entry.step_index, "event": "screenshot_captured", "detail": entry.before_artifact_path})
        events.append({"step_index": entry.step_index, "event": "perception_requested", "detail": entry.error_message})
        retry_log = _read_text(entry.perception_debug.retry_log_artifact_path) or ""
        if retry_log:
            for line in retry_log.splitlines():
                if line.strip():
                    events.append({"step_index": entry.step_index, "event": "perception_retry", "detail": line})
        events.append({"step_index": entry.step_index, "event": "run_aborted", "detail": entry.error_message})
    if partial_current_step is not None:
        events.extend(_partial_step_events(partial_current_step))
    if state.status.value == "succeeded":
        events.append({"step_index": state.step_count, "event": "run_succeeded", "detail": state.stop_reason.value if state.stop_reason else "succeeded"})
    elif state.status.value == "failed":
        events.append({"step_index": state.step_count, "event": "run_aborted", "detail": state.stop_reason.value if state.stop_reason else "failed"})
    return sorted(events, key=lambda item: (item["step_index"], item["event"]))


def _current_phase(state, current_step: dict[str, Any] | None, pre_step_failure: PreStepFailureLog | None) -> str:
    if state.status.value in {"succeeded", "failed"}:
        if pre_step_failure is not None:
            return pre_step_failure.failure.stage.value
        if current_step and current_step.get("is_partial"):
            return current_step.get("phase") or "capture"
        if current_step and current_step.get("recovery"):
            recovery = current_step["recovery"]
            if recovery.get("strategy") == "stop":
                return "recover"
        return "complete"
    if state.step_count == 0:
        return current_step.get("phase") if current_step is not None else "capture"
    if current_step and current_step.get("is_partial"):
        return current_step.get("phase") or "capture"
    return "recover" if current_step is not None else "capture"


def _task_id_from_intent(intent: str) -> str:
    lowered = intent.lower()
    if "gmail" in lowered:
        return "gmail_draft_authenticated"
    if "form" in lowered:
        return "practice_form_submit"
    return "ad_hoc_run"


def _read_text(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_json(path_value: str | None) -> Any:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _partial_step_phase(step_dir: Path) -> str:
    if (step_dir / "progress_trace.json").exists():
        return "recover"
    if (step_dir / "execution_trace.json").exists():
        return "verify"
    if (step_dir / "selector_trace.json").exists():
        return "execute"
    if (step_dir / "policy_decision.json").exists():
        return "choose"
    if (step_dir / "perception_parsed.json").exists() or (step_dir / "perception_diagnostics.json").exists():
        return "choose"
    if (step_dir / "perception_raw.txt").exists() or (step_dir / "perception_prompt.txt").exists():
        return "perceive"
    if (step_dir / "before.png").exists():
        return "capture"
    return "capture"


def _partial_step_events(step: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    step_index = step["step_index"]
    if step.get("before_artifact_path"):
        events.append({"step_index": step_index, "event": "screenshot_captured", "detail": step["before_artifact_path"]})
    perception = step.get("perception") or {}
    if perception:
        events.append({"step_index": step_index, "event": "perception_requested", "detail": perception.get("summary") or "perception available"})
        retry_log = perception.get("retry_log") or ""
        for line in retry_log.splitlines():
            if line.strip():
                events.append({"step_index": step_index, "event": "perception_retry", "detail": line})
    selector = (step.get("selector") or {}).get("trace")
    if selector is not None:
        events.append({"step_index": step_index, "event": "selector_trace_written", "detail": f"step_{step_index}/selector_trace.json"})
        if selector.get("recovery_attempted"):
            events.append(
                {
                    "step_index": step_index,
                    "event": "selector_recovery_used" if selector.get("final_decision") == "success" else "selector_recovery_failed",
                    "detail": selector.get("recovery_strategy_used"),
                }
            )
    execution = step.get("execution") or {}
    trace = execution.get("trace") if isinstance(execution, dict) else None
    if trace is not None and trace.get("retry_attempted"):
        events.append({"step_index": step_index, "event": "execution_retry", "detail": trace.get("retry_reason") or "unknown"})
    progress = (step.get("progress") or {}).get("trace")
    if isinstance(progress, dict) and progress.get("blocked_as_redundant"):
        events.append({"step_index": step_index, "event": "progress_blocked_action", "detail": progress.get("redundancy_reason")})
    return events


def _image_dimensions(path_value: str | None) -> dict[str, int] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return {"width": image.width, "height": image.height}
    except Exception:
        return None
