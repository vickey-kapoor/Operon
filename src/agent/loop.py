"""Agent loop orchestrator for the active browser benchmark workflow."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.agent.capture import CaptureService
from src.agent.perception import PerceptionLowQualityError, PerceptionService
from src.agent.policy import PolicyService
from src.agent.recovery import RecoveryManager
from src.agent.selector import DeterministicTargetSelector
from src.agent.verifier import VerifierService
from src.executor.browser import BrowserExecutor
from src.models.common import (
    FailureCategory,
    LoopStage,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from src.models.execution import ExecutedAction, ExecutionReresolutionTrace
from src.models.logs import FailureRecord, ModelDebugArtifacts, PreStepFailureLog, StepLog
from src.models.policy import ActionType, AgentAction
from src.models.selector import TargetIntent, TargetIntentAction
from src.models.progress import ProgressTrace
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.perception import PageHint, UIElement
from src.models.verification import VerificationStatus
from src.store.run_logger import append_step_log
from src.store.memory import MemoryStore
from src.store.run_store import RunStore


@dataclass(slots=True)
class _RetryResolution:
    action: AgentAction | None
    trace: ExecutionReresolutionTrace | None


class AgentLoop:
    """Coordinates the MVP control loop and terminal benchmark boundaries."""

    MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS = 2
    MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS = 2
    MAX_NO_PROGRESS_STEPS = 3
    MAX_REPEAT_SAME_FAILURE = 2
    RECENT_WINDOW_SIZE = 6

    SUCCESS_STOP_REASONS = {
        StopReason.STOP_BEFORE_SEND,
        StopReason.FORM_SUBMITTED_SUCCESS,
        StopReason.TASK_COMPLETED,
    }

    def __init__(
        self,
        capture_service: CaptureService,
        perception_service: PerceptionService,
        run_store: RunStore,
        policy_service: PolicyService,
        executor: BrowserExecutor,
        verifier_service: VerifierService,
        recovery_manager: RecoveryManager,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.capture_service = capture_service
        self.perception_service = perception_service
        self.run_store = run_store
        self.policy_service = policy_service
        self.executor = executor
        self.verifier_service = verifier_service
        self.recovery_manager = recovery_manager
        self.memory_store = memory_store
        self.target_selector = DeterministicTargetSelector()

    async def start_run(self, request: RunTaskRequest) -> RunResponse:
        record = self.run_store.create_run(intent=request.intent, start_url=request.start_url)
        if request.start_url:
            await self.executor.execute(
                AgentAction(action_type=ActionType.NAVIGATE, url=request.start_url)
            )
        return RunResponse(
            run_id=record.run_id,
            status=record.status,
            intent=record.intent,
            step_count=record.step_count,
        )

    async def run_live_benchmark(
        self,
        intent: str = "Complete the auth-free form and submit it successfully.",
        *,
        benchmark_url: str = "https://practice-automation.com/form-fields/",
        max_steps: int = 12,
    ) -> RunResponse:
        response = await self.start_run(RunTaskRequest(intent=intent, start_url=benchmark_url))

        while True:
            state = await self.run_store.get_run(response.run_id)
            if state is None:
                raise ValueError(f"Run {response.run_id!r} not found")
            if state.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_FOR_USER}:
                return response
            if state.step_count >= max_steps:
                state.stop_reason = StopReason.MAX_STEP_LIMIT_REACHED
                updated = await self.run_store.set_status(state.run_id, RunStatus.FAILED)
                return RunResponse(
                    run_id=updated.run_id,
                    status=updated.status,
                    intent=updated.intent,
                    step_count=updated.step_count,
                )
            response = await self.step_run(StepRequest(run_id=response.run_id))

    async def resume_run(self, run_id: str) -> RunResponse:
        """Resume a run that was paused waiting for user input."""
        record = await self.run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"Run {run_id!r} not found")
        if record.status is not RunStatus.WAITING_FOR_USER:
            raise ValueError(f"Run {run_id!r} is {record.status.value}, not waiting_for_user")
        record.stop_reason = None
        updated = await self.run_store.set_status(run_id, RunStatus.RUNNING)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    async def step_run(self, request: StepRequest) -> RunResponse:
        record = await self.run_store.get_run(request.run_id)
        if record is None:
            raise ValueError(f"Run {request.run_id!r} not found")

        step_index = record.step_count + 1
        before_artifact_path = self._before_artifact_path(record.run_id, step_index)
        after_artifact_path = self._after_artifact_path(record.run_id, step_index)
        self._prepare_step_artifacts(record.run_id, step_index, before_artifact_path, after_artifact_path)

        frame = await self.capture_service.capture(record)
        try:
            perception = await self.perception_service.perceive(frame, record)
        except Exception as exc:
            return await self._record_pre_step_perception_failure(
                record=record,
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                error=exc,
            )
        perception_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "perception", self.perception_service)
        state = await self.run_store.update_state(record.run_id, perception)
        self._sync_progress_state_with_perception(state, perception)
        decision = await self.policy_service.choose_action(state, perception)
        decision = decision.model_copy(update={"action": self._attach_target_context(decision.action, perception)})
        policy_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "policy", self.policy_service)
        state.current_subgoal = decision.active_subgoal

        # Human-in-the-loop: pause the run and wait for user input
        if decision.action.action_type is ActionType.WAIT_FOR_USER:
            return await self._pause_for_user(
                record=record,
                state=state,
                perception=perception,
                decision=decision,
                perception_debug=perception_debug,
                policy_debug=policy_debug,
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                after_artifact_path=after_artifact_path,
            )

        blocked_action = self._block_redundant_action(state, decision.action, step_index)
        if blocked_action is None:
            state, executed_action = await self._execute_with_hardening(
                record=record,
                state=state,
                perception=perception,
                decision_action=decision.action,
                step_index=step_index,
            )
        else:
            executed_action = blocked_action
        executed_action = self._relocate_after_artifact(executed_action, after_artifact_path)
        verification = await self.verifier_service.verify(state, decision, executed_action)
        recovery: RecoveryDecision = await self.recovery_manager.recover(
            state,
            decision,
            executed_action,
            verification,
        )
        progress_trace = self._update_progress_state(
            state=state,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            recovery=recovery,
            step_index=step_index,
        )
        recovery = self._apply_progress_stop_guard(recovery, progress_trace)
        progress_trace = progress_trace.model_copy(
            update={
                "final_failure_category": progress_trace.final_failure_category
                or recovery.failure_category
                or verification.failure_category
                or executed_action.failure_category,
                "final_stop_reason": progress_trace.final_stop_reason or recovery.stop_reason,
            }
        )
        progress_trace_artifact_path = self._persist_progress_trace(record.run_id, step_index, progress_trace)
        failure = self._build_failure_record(state, decision, executed_action, verification, recovery)

        step_log = StepLog(
            run_id=record.run_id,
            step_id=f"step_{step_index}",
            step_index=step_index,
            before_artifact_path=before_artifact_path,
            after_artifact_path=after_artifact_path,
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            perception=perception,
            policy_decision=decision,
            executed_action=executed_action,
            verification_result=verification,
            recovery_decision=recovery,
            progress_state=state.progress_state.model_copy(deep=True),
            progress_trace_artifact_path=progress_trace_artifact_path,
            failure=failure,
        )
        append_step_log(self._run_log_path(record.run_id), step_log)

        if self.memory_store is not None:
            self.memory_store.record_step(
                state=state,
                perception=perception,
                decision=decision,
                executed_action=executed_action,
                verification=verification,
                recovery=recovery,
            )

        state.action_history.append(executed_action)
        state.verification_history.append(verification)
        self._update_target_failure_signal(state, executed_action, verification)
        state.artifact_paths.extend(
            [
                before_artifact_path,
                after_artifact_path,
                perception_debug.prompt_artifact_path,
                perception_debug.raw_response_artifact_path,
                perception_debug.parsed_artifact_path,
                *( [perception_debug.diagnostics_artifact_path] if perception_debug.diagnostics_artifact_path else [] ),
                policy_debug.prompt_artifact_path,
                policy_debug.raw_response_artifact_path,
                policy_debug.parsed_artifact_path,
                *( [executed_action.execution_trace_artifact_path] if executed_action.execution_trace_artifact_path else [] ),
                progress_trace_artifact_path,
            ]
        )

        final_status = RunStatus.RUNNING
        if (
            verification.stop_condition_met
            and verification.status is VerificationStatus.SUCCESS
            and verification.stop_reason in self.SUCCESS_STOP_REASONS
        ):
            final_status = RunStatus.SUCCEEDED
            state.stop_reason = verification.stop_reason
        elif recovery.strategy is RecoveryStrategy.STOP:
            final_status = RunStatus.FAILED
            state.stop_reason = recovery.stop_reason
        else:
            state.stop_reason = None

        updated = await self.run_store.set_status(record.run_id, final_status)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    async def _pause_for_user(
        self,
        *,
        record,
        state,
        perception,
        decision,
        perception_debug,
        policy_debug,
        step_index: int,
        before_artifact_path: str,
        after_artifact_path: str,
    ) -> RunResponse:
        """Pause the run and set status to WAITING_FOR_USER."""
        executed_action = ExecutedAction(
            action=decision.action,
            success=True,
            detail=f"Paused for user: {decision.action.text}",
        )
        verification = await self.verifier_service.verify(state, decision, executed_action)
        recovery = RecoveryDecision(
            strategy=RecoveryStrategy.STOP,
            message=f"Waiting for user: {decision.action.text}",
            terminal=False,
            recoverable=True,
            stop_reason=StopReason.WAITING_FOR_USER,
        )
        append_step_log(
            self._run_log_path(record.run_id),
            StepLog(
                run_id=record.run_id,
                step_id=f"step_{step_index}",
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                after_artifact_path=after_artifact_path,
                perception_debug=perception_debug,
                policy_debug=policy_debug,
                perception=perception,
                policy_decision=decision,
                executed_action=executed_action,
                verification_result=verification,
                recovery_decision=recovery,
            ),
        )
        state.stop_reason = StopReason.WAITING_FOR_USER
        state.step_count = step_index
        await self.run_store.save_state(state)
        updated = await self.run_store.set_status(record.run_id, RunStatus.WAITING_FOR_USER)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    async def _record_pre_step_perception_failure(
        self,
        *,
        record,
        step_index: int,
        before_artifact_path: str,
        error: Exception,
    ) -> RunResponse:
        perception_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "perception", self.perception_service)
        failure_category = FailureCategory.PRE_STEP_PERCEPTION_FAILED
        stop_reason = StopReason.PRE_STEP_PERCEPTION_FAILED
        if isinstance(error, PerceptionLowQualityError):
            failure_category = error.failure_category
            stop_reason = error.stop_reason

        failure = FailureRecord(
            category=failure_category,
            stage=LoopStage.PERCEIVE,
            retry_count=0,
            terminal=True,
            recoverable=False,
            reason=str(error),
            stop_reason=stop_reason,
        )
        append_step_log(
            self._run_log_path(record.run_id),
            PreStepFailureLog(
                run_id=record.run_id,
                step_id=f"step_{step_index}",
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                perception_debug=perception_debug,
                failure=failure,
                error_message=str(error),
            ),
        )
        record.artifact_paths.extend(
            [
                before_artifact_path,
                perception_debug.prompt_artifact_path,
                perception_debug.raw_response_artifact_path,
                perception_debug.parsed_artifact_path,
                *( [perception_debug.diagnostics_artifact_path] if perception_debug.diagnostics_artifact_path else [] ),
                *( [perception_debug.retry_log_artifact_path] if perception_debug.retry_log_artifact_path else [] ),
            ]
        )
        record.stop_reason = stop_reason
        updated = await self.run_store.set_status(record.run_id, RunStatus.FAILED)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    def _prepare_step_artifacts(self, run_id: str, step_index: int, before_artifact_path: str, after_artifact_path: str) -> None:
        Path(before_artifact_path).parent.mkdir(parents=True, exist_ok=True)
        Path(after_artifact_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self._run_log_path(run_id)).parent.mkdir(parents=True, exist_ok=True)
        Path(self._run_log_path(run_id)).touch(exist_ok=True)

    def _before_artifact_path(self, run_id: str, step_index: int) -> str:
        if hasattr(self.run_store, "before_artifact_path"):
            return str(self.run_store.before_artifact_path(run_id, step_index))
        return str(Path("runs") / run_id / f"step_{step_index}" / "before.png")

    def _after_artifact_path(self, run_id: str, step_index: int) -> str:
        if hasattr(self.run_store, "after_artifact_path"):
            return str(self.run_store.after_artifact_path(run_id, step_index))
        return str(Path("runs") / run_id / f"step_{step_index}" / "after.png")

    def _run_log_path(self, run_id: str) -> str:
        if hasattr(self.run_store, "run_log_path"):
            return str(self.run_store.run_log_path(run_id))
        return str(Path("runs") / run_id / "run.jsonl")

    def _relocate_after_artifact(self, executed_action, planned_path: str):
        if executed_action.artifact_path is None:
            return executed_action.model_copy(update={"artifact_path": planned_path})

        current_path = Path(executed_action.artifact_path)
        target_path = Path(planned_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if current_path != target_path and current_path.exists():
            shutil.move(str(current_path), str(target_path))
        return executed_action.model_copy(update={"artifact_path": str(target_path)})

    def _resolve_model_debug_artifacts(
        self,
        run_id: str,
        step_index: int,
        stage_name: str,
        service,
    ) -> ModelDebugArtifacts:
        if hasattr(service, "latest_debug_artifacts"):
            debug_artifacts = service.latest_debug_artifacts()
            if debug_artifacts is not None:
                return debug_artifacts

        step_dir = Path("runs") / run_id / f"step_{step_index}"
        return ModelDebugArtifacts(
            prompt_artifact_path=str(step_dir / f"{stage_name}_prompt.txt"),
            raw_response_artifact_path=str(step_dir / f"{stage_name}_raw.txt"),
            parsed_artifact_path=str(step_dir / ("policy_decision.json" if stage_name == "policy" else "perception_parsed.json")),
            diagnostics_artifact_path=str(step_dir / f"{stage_name}_diagnostics.json") if stage_name == "perception" else None,
        )

    async def _execute_with_hardening(
        self,
        *,
        record,
        state,
        perception,
        decision_action,
        step_index: int,
    ):
        executed_action = await self.executor.execute(decision_action)
        executed_action = self._apply_no_progress_detection(state, executed_action)

        if not self._should_retry_execution(executed_action):
            return state, self._persist_execution_trace(record.run_id, step_index, executed_action)

        retry_action = decision_action
        retry_state = state
        target_reresolved = False
        try:
            frame = await self.capture_service.capture(record)
            refreshed_perception = await self.perception_service.perceive(frame, record)
            retry_state = await self.run_store.update_state(record.run_id, refreshed_perception)
            resolution = self._resolve_retry_action(
                action=decision_action,
                perception=refreshed_perception,
                retry_reason=executed_action.failure_category,
            )
            retry_action = resolution.action
            target_reresolved = resolution.trace is not None and resolution.trace.succeeded
        except Exception:
            return state, self._persist_execution_trace(record.run_id, step_index, executed_action)

        if retry_action is None:
            failed = self._apply_reresolution_failure(executed_action, resolution.trace)
            return retry_state, self._persist_execution_trace(record.run_id, step_index, failed)

        retry_executed = await self.executor.execute(retry_action)
        retry_executed = self._merge_execution_retry(
            original=executed_action,
            retried=retry_executed,
            retry_reason=executed_action.failure_category,
            target_reresolved=target_reresolved,
            reresolution_trace=resolution.trace,
        )
        retry_executed = self._apply_no_progress_detection(state, retry_executed)
        return retry_state, self._persist_execution_trace(record.run_id, step_index, retry_executed)

    @staticmethod
    def _should_retry_execution(executed_action) -> bool:
        return executed_action.failure_category in {
            FailureCategory.STALE_TARGET_BEFORE_ACTION,
            FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
            FailureCategory.TARGET_LOST_BEFORE_ACTION,
            FailureCategory.FOCUS_VERIFICATION_FAILED,
            FailureCategory.CLICK_BEFORE_TYPE_FAILED,
            FailureCategory.CLICK_NO_EFFECT,
            FailureCategory.CHECKBOX_VERIFICATION_FAILED,
            FailureCategory.SELECT_VERIFICATION_FAILED,
        }

    def _resolve_retry_action(self, *, action, perception, retry_reason: FailureCategory | None) -> _RetryResolution:
        if retry_reason in {
            FailureCategory.STALE_TARGET_BEFORE_ACTION,
            FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
            FailureCategory.TARGET_LOST_BEFORE_ACTION,
        }:
            return self._intent_reresolve_action(action, perception, retry_reason)
        return _RetryResolution(action=self._refresh_action_coordinates(action, perception), trace=None)

    @staticmethod
    def _refresh_action_coordinates(action, perception):
        target_id = action.target_element_id
        if target_id is None:
            return action
        target = next((element for element in perception.visible_elements if element.element_id == target_id), None)
        if target is None:
            return action
        return action.model_copy(
            update={
                "x": target.x + max(1, target.width // 2),
                "y": target.y + max(1, target.height // 2),
            }
        )

    @staticmethod
    def _merge_execution_retry(*, original, retried, retry_reason, target_reresolved, reresolution_trace):
        original_trace = original.execution_trace
        retried_trace = retried.execution_trace
        if original_trace is None or retried_trace is None:
            return retried
        merged_trace = original_trace.model_copy(
            update={
                "attempts": [*original_trace.attempts, *retried_trace.attempts],
                "target_reresolved": target_reresolved,
                "retry_attempted": True,
                "retry_reason": retry_reason,
                "reresolution_trace": reresolution_trace,
                "final_outcome": retried_trace.final_outcome,
                "final_failure_category": retried_trace.final_failure_category,
            }
        )
        return retried.model_copy(update={"execution_trace": merged_trace})

    def _attach_target_context(self, action: AgentAction, perception) -> AgentAction:
        if action.target_context is not None:
            return action
        target = self._resolve_action_target(action, perception)
        intent = self._infer_target_intent(action, target, perception)
        if target is None or intent is None:
            return action
        context = self.target_selector.build_selection_context(
            perception,
            intent,
            target,
            page_signature=self._page_signature(perception),
        )
        return action.model_copy(update={"target_context": context})

    def _intent_reresolve_action(
        self,
        action: AgentAction,
        perception,
        retry_reason: FailureCategory,
    ) -> _RetryResolution:
        if action.target_context is None:
            return _RetryResolution(action=self._refresh_action_coordinates(action, perception), trace=None)

        result = self.target_selector.reresolve(perception, action.target_context)
        selected = result.selected
        trace = ExecutionReresolutionTrace(
            trigger_reason=retry_reason,
            original_target_element_id=action.target_context.original_target.element_id,
            original_intent=action.target_context.intent,
            original_target_signature=action.target_context.original_target,
            original_page_signature=action.target_context.original_page_signature,
            selector_trace=result.trace,
            reused_original_element_id=(
                selected is not None and selected.element_id == action.target_context.original_target.element_id
            ),
            final_target_element_id=selected.element_id if selected is not None else None,
            succeeded=selected is not None,
            detail=(
                f"Re-resolved target to {selected.element_id} from original intent."
                if selected is not None
                else "Intent-based target re-resolution did not find a safe deterministic match."
            ),
        )
        if selected is None:
            return _RetryResolution(action=None, trace=trace)

        return _RetryResolution(
            action=action.model_copy(
                update={
                    "target_element_id": selected.element_id,
                    "x": selected.x + max(1, selected.width // 2),
                    "y": selected.y + max(1, selected.height // 2),
                }
            ),
            trace=trace,
        )

    @staticmethod
    def _apply_reresolution_failure(original, reresolution_trace: ExecutionReresolutionTrace | None):
        trace = original.execution_trace
        if trace is None or reresolution_trace is None:
            return original
        failure_category = (
            FailureCategory.TARGET_RERESOLUTION_AMBIGUOUS
            if reresolution_trace.selector_trace.rejection_reason is FailureCategory.AMBIGUOUS_TARGET_CANDIDATES
            else FailureCategory.TARGET_RERESOLUTION_FAILED
        )
        merged_trace = trace.model_copy(
            update={
                "retry_attempted": True,
                "retry_reason": reresolution_trace.trigger_reason,
                "reresolution_trace": reresolution_trace,
                "final_outcome": "failure",
                "final_failure_category": failure_category,
            }
        )
        return original.model_copy(
            update={
                "success": False,
                "detail": f"Execution failed: {failure_category.value.replace('_', ' ')}.",
                "failure_category": failure_category,
                "failure_stage": LoopStage.EXECUTE,
                "execution_trace": merged_trace,
            }
        )

    @staticmethod
    def _resolve_action_target(action: AgentAction, perception) -> UIElement | None:
        if action.target_element_id is not None:
            target = next((element for element in perception.visible_elements if element.element_id == action.target_element_id), None)
            if target is not None:
                return target
        if action.x is None or action.y is None:
            return None
        containing = [
            element
            for element in perception.visible_elements
            if element.is_interactable
            and element.x <= action.x <= element.x + element.width
            and element.y <= action.y <= element.y + element.height
        ]
        if containing:
            return sorted(containing, key=lambda element: (element.y, element.x, element.element_id))[0]
        return None

    def _infer_target_intent(self, action: AgentAction, target: UIElement | None, perception) -> TargetIntent | None:
        if target is None:
            return None
        if action.action_type is ActionType.CLICK:
            intent_action = TargetIntentAction.CLICK
        elif action.action_type is ActionType.TYPE:
            intent_action = TargetIntentAction.TYPE
        elif action.action_type is ActionType.SELECT:
            intent_action = TargetIntentAction.SELECT
        else:
            return None
        return TargetIntent(
            action=intent_action,
            target_text=target.primary_name if not target.is_unlabeled else None,
            target_role=target.role,
            expected_element_types=[target.element_type],
            value_to_type=action.text if action.action_type in {ActionType.TYPE, ActionType.SELECT} else None,
            expected_section=self._expected_section(perception.page_hint),
        )

    @staticmethod
    def _expected_section(page_hint: PageHint) -> str | None:
        if page_hint in {PageHint.FORM_PAGE, PageHint.FORM_SUCCESS}:
            return "form"
        if page_hint in {PageHint.GMAIL_COMPOSE, PageHint.GMAIL_INBOX, PageHint.GMAIL_MESSAGE_VIEW}:
            return "compose"
        return None

    def _persist_execution_trace(self, run_id: str, step_index: int, executed_action):
        if executed_action.execution_trace is None:
            return executed_action
        step_dir = Path(self._before_artifact_path(run_id, step_index)).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        trace_path = step_dir / "execution_trace.json"
        trace_path.write_text(executed_action.execution_trace.model_dump_json(indent=2), encoding="utf-8")
        return executed_action.model_copy(update={"execution_trace_artifact_path": str(trace_path)})

    def _persist_progress_trace(self, run_id: str, step_index: int, progress_trace: ProgressTrace) -> str:
        step_dir = Path(self._before_artifact_path(run_id, step_index)).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        trace_path = step_dir / "progress_trace.json"
        trace_path.write_text(progress_trace.model_dump_json(indent=2), encoding="utf-8")
        return str(trace_path)

    def _sync_progress_state_with_perception(self, state, perception) -> None:
        page_signature = self._page_signature(perception)
        progress_state = state.progress_state
        previous_signature = progress_state.latest_page_signature
        progress_state.latest_page_signature = page_signature
        if previous_signature is None or previous_signature == page_signature:
            return
        progress_state.repeated_action_count.clear()
        progress_state.repeated_target_count.clear()
        progress_state.recent_failures = []
        progress_state.loop_detected = False
        progress_state.no_progress_streak = max(0, progress_state.no_progress_streak - 1)

    def _block_redundant_action(self, state, action, step_index: int):
        progress_state = state.progress_state
        page_signature = progress_state.latest_page_signature or "unknown_page"
        action_signature = self._action_signature(action)
        target_signature = self._target_signature(action)
        subgoal_signature = self._subgoal_signature(state.current_subgoal)

        if (
            target_signature is not None
            and action.action_type in {ActionType.TYPE, ActionType.SELECT}
            and action.text is not None
            and progress_state.target_value_history.get(target_signature) == action.text
            and progress_state.target_completion_page_signatures.get(target_signature) == page_signature
        ):
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                detail="Action blocked: target already has the verified value on the current page.",
            )

        if (
            subgoal_signature in progress_state.completed_subgoals
            and progress_state.subgoal_completion_page_signatures.get(subgoal_signature) == page_signature
            and action.action_type is not ActionType.CLICK
        ):
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                detail="Action blocked: subgoal is already completed on the current page.",
            )

        repeated_action_count = progress_state.repeated_action_count.get(action_signature, 0)
        if (
            action.action_type is ActionType.CLICK
            and repeated_action_count >= self.MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS
            and progress_state.no_progress_streak > 0
        ):
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                detail="Action blocked: repeated click without meaningful progress.",
            )

        if (
            target_signature is not None
            and progress_state.repeated_target_count.get(target_signature, 0) >= self.MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS
            and progress_state.no_progress_streak > 0
        ):
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS,
                detail="Action blocked: repeated targeting without meaningful progress.",
            )

        return None

    @staticmethod
    def _redundant_action_failure(*, action, category: FailureCategory, detail: str):
        return ExecutedAction(
            action=action,
            success=False,
            detail=detail,
            failure_category=category,
            failure_stage=LoopStage.EXECUTE,
        )

    def _update_progress_state(
        self,
        *,
        state,
        decision,
        executed_action,
        verification,
        recovery,
        step_index: int,
    ) -> ProgressTrace:
        progress_state = state.progress_state
        action_signature = self._action_signature(executed_action.action)
        target_signature = self._target_signature(executed_action.action)
        subgoal_signature = self._subgoal_signature(decision.active_subgoal)
        failure_signature = self._failure_signature(executed_action, verification, recovery, target_signature)
        progress_made = self._meaningful_progress(executed_action, verification)

        progress_state.recent_actions = self._append_window(progress_state.recent_actions, action_signature)
        loop_failure_category = None
        loop_pattern = None

        if progress_made:
            progress_state.no_progress_streak = 0
            progress_state.loop_detected = False
            progress_state.repeated_action_count.clear()
            progress_state.repeated_target_count.clear()
            progress_state.recent_failures = []
            progress_state.last_meaningful_progress_step = step_index
            self._mark_completed_progress(state, decision, executed_action)
        else:
            progress_state.no_progress_streak += 1
            progress_state.repeated_action_count[action_signature] = progress_state.repeated_action_count.get(action_signature, 0) + 1
            if target_signature is not None:
                progress_state.repeated_target_count[target_signature] = progress_state.repeated_target_count.get(target_signature, 0) + 1
            if failure_signature is not None:
                progress_state.recent_failures = self._append_window(progress_state.recent_failures, failure_signature)

            loop_failure_category, loop_pattern = self._detect_loop_failure(
                progress_state=progress_state,
                action_signature=action_signature,
                target_signature=target_signature,
                failure_signature=failure_signature,
            )
            progress_state.loop_detected = loop_failure_category is not None

        return ProgressTrace(
            step_index=step_index,
            page_signature=progress_state.latest_page_signature or "unknown_page",
            action_signature=action_signature,
            target_signature=target_signature,
            subgoal_signature=subgoal_signature,
            failure_signature=failure_signature,
            blocked_as_redundant=executed_action.failure_category
            in {
                FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS,
            },
            redundancy_reason=executed_action.failure_category
            if executed_action.failure_category
            in {
                FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS,
            }
            else None,
            loop_pattern_detected=loop_pattern,
            progress_made=progress_made,
            no_progress_streak=progress_state.no_progress_streak,
            final_failure_category=loop_failure_category,
            final_stop_reason=self._stop_reason_for_failure(loop_failure_category),
        )

    def _apply_progress_stop_guard(self, recovery: RecoveryDecision, progress_trace: ProgressTrace) -> RecoveryDecision:
        if progress_trace.final_failure_category is None:
            return recovery
        stop_reason = progress_trace.final_stop_reason or self._stop_reason_for_failure(progress_trace.final_failure_category)
        return RecoveryDecision(
            strategy=RecoveryStrategy.STOP,
            message=f"Loop suppressed: {progress_trace.final_failure_category.value}.",
            failure_category=progress_trace.final_failure_category,
            failure_stage=LoopStage.ORCHESTRATE,
            terminal=True,
            recoverable=False,
            stop_reason=stop_reason,
        )

    def _detect_loop_failure(
        self,
        *,
        progress_state,
        action_signature: str,
        target_signature: str | None,
        failure_signature: str | None,
    ) -> tuple[FailureCategory | None, str | None]:
        if progress_state.repeated_action_count.get(action_signature, 0) > self.MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS:
            return FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS, "same_action_repeated_without_progress"

        if (
            target_signature is not None
            and progress_state.repeated_target_count.get(target_signature, 0) > self.MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS
        ):
            return FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS, "same_target_repeated_without_progress"

        if failure_signature is not None and progress_state.recent_failures.count(failure_signature) > self.MAX_REPEAT_SAME_FAILURE:
            return FailureCategory.REPEATED_FAILURE_LOOP, "repeated_identical_failure_signature"

        if self._alternating_action_loop(progress_state.recent_actions):
            return FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS, "alternating_action_pattern_without_progress"

        if progress_state.no_progress_streak >= self.MAX_NO_PROGRESS_STEPS:
            return FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS, "no_meaningful_progress_threshold_reached"

        return None, None

    @staticmethod
    def _alternating_action_loop(recent_actions: list[str]) -> bool:
        if len(recent_actions) < 4:
            return False
        a, b, c, d = recent_actions[-4:]
        return a == c and b == d and a != b

    def _mark_completed_progress(self, state, decision, executed_action) -> None:
        progress_state = state.progress_state
        target_signature = self._target_signature(executed_action.action)
        subgoal_signature = self._subgoal_signature(decision.active_subgoal)
        page_signature = progress_state.latest_page_signature or "unknown_page"

        if target_signature is not None:
            if target_signature not in progress_state.completed_targets:
                progress_state.completed_targets.append(target_signature)
            progress_state.target_completion_page_signatures[target_signature] = page_signature
            if executed_action.action.action_type in {ActionType.TYPE, ActionType.SELECT} and executed_action.action.text is not None:
                progress_state.target_value_history[target_signature] = executed_action.action.text

        if subgoal_signature not in progress_state.completed_subgoals:
            progress_state.completed_subgoals.append(subgoal_signature)
        progress_state.subgoal_completion_page_signatures[subgoal_signature] = page_signature

    @staticmethod
    def _meaningful_progress(executed_action, verification) -> bool:
        return executed_action.success and verification.status is VerificationStatus.SUCCESS and verification.expected_outcome_met

    @classmethod
    def _failure_signature(cls, executed_action, verification, recovery, target_signature: str | None) -> str | None:
        category = recovery.failure_category or verification.failure_category or executed_action.failure_category
        if category is None:
            return None
        target_part = target_signature or "no_target"
        return f"{category.value}|{target_part}"

    @staticmethod
    def _stop_reason_for_failure(category: FailureCategory | None) -> StopReason | None:
        if category is None:
            return None
        return StopReason(category.value)

    @classmethod
    def _page_signature(cls, perception) -> str:
        top_elements = perception.visible_elements[:6]
        element_part = "|".join(f"{element.element_id}:{element.primary_name}" for element in top_elements) or "none"
        focused = perception.focused_element_id or "none"
        return f"{perception.page_hint.value}|{focused}|{element_part}"

    @staticmethod
    def _append_window(entries: list[str], value: str) -> list[str]:
        return [*entries, value][-AgentLoop.RECENT_WINDOW_SIZE :]

    @classmethod
    def _action_signature(cls, action: AgentAction) -> str:
        target = cls._target_signature(action) or "no_target"
        payload = action.text or action.key or action.url or (str(action.wait_ms) if action.wait_ms is not None else "")
        return f"{action.action_type.value}|{target}|{payload.strip().lower()}"

    @staticmethod
    def _target_signature(action: AgentAction) -> str | None:
        if action.target_element_id is not None:
            return f"id:{action.target_element_id}"
        if action.selector is not None:
            return f"selector:{action.selector}"
        if action.x is not None and action.y is not None:
            return f"xy:{action.x}:{action.y}"
        return None

    @classmethod
    def _subgoal_signature(cls, subgoal: str | None) -> str:
        if not subgoal:
            return "unknown_subgoal"
        normalized = "".join(character.lower() if character.isalnum() else "_" for character in subgoal)
        return "_".join(part for part in normalized.split("_") if part) or "unknown_subgoal"

    @staticmethod
    def _apply_no_progress_detection(state, executed_action):
        trace = executed_action.execution_trace
        if trace is None or not trace.attempts:
            return executed_action
        latest_attempt = trace.attempts[-1]
        if not latest_attempt.no_progress_detected:
            return executed_action
        if not state.action_history:
            return executed_action
        previous = state.action_history[-1]
        previous_trace = previous.execution_trace
        if previous.action.target_element_id != executed_action.action.target_element_id:
            return executed_action
        if previous_trace is None or not previous_trace.attempts or not previous_trace.attempts[-1].no_progress_detected:
            return executed_action
        merged_trace = trace.model_copy(update={"final_failure_category": FailureCategory.EXECUTION_NO_PROGRESS})
        return executed_action.model_copy(
            update={
                "success": False,
                "detail": "Execution failed: repeated no-progress detected on the same target.",
                "failure_category": FailureCategory.EXECUTION_NO_PROGRESS,
                "failure_stage": LoopStage.EXECUTE,
                "execution_trace": merged_trace,
            }
        )

    def _build_failure_record(self, state, decision, executed_action, verification, recovery) -> FailureRecord | None:
        retry_count = self._retry_count(state, decision, verification)
        if recovery.failure_category is not None:
            return FailureRecord(
                category=recovery.failure_category,
                stage=recovery.failure_stage or LoopStage.RECOVER,
                retry_count=retry_count,
                terminal=recovery.terminal,
                recoverable=recovery.recoverable,
                reason=recovery.message,
                stop_reason=recovery.stop_reason,
            )
        if verification.failure_category is not None:
            return FailureRecord(
                category=verification.failure_category,
                stage=verification.failure_stage or LoopStage.VERIFY,
                retry_count=retry_count,
                terminal=False,
                recoverable=True,
                reason=verification.reason,
            )
        if executed_action.failure_category is not None:
            return FailureRecord(
                category=executed_action.failure_category,
                stage=executed_action.failure_stage or LoopStage.EXECUTE,
                retry_count=retry_count,
                terminal=False,
                recoverable=True,
                reason=executed_action.detail,
            )
        return None

    @staticmethod
    def _retry_count(state, decision, verification) -> int:
        suffix = verification.failure_type.value if verification.failure_type else verification.status.value
        return state.retry_counts.get(f"{decision.active_subgoal}:{suffix}", 0)

    @staticmethod
    def _update_target_failure_signal(state, executed_action, verification) -> None:
        action = executed_action.action
        if action.action_type is not ActionType.TYPE or action.target_element_id is None:
            return

        failure_category = verification.failure_category or executed_action.failure_category
        if failure_category in {
            FailureCategory.EXECUTION_TARGET_NOT_FOUND,
            FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        }:
            key = f"type:{action.target_element_id}:{failure_category.value}"
            state.target_failure_counts[key] = state.target_failure_counts.get(key, 0) + 1
