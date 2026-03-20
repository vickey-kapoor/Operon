"""Agent loop orchestrator for the Gmail draft MVP workflow."""

import shutil
from pathlib import Path

from src.agent.capture import CaptureService
from src.agent.perception import PerceptionService
from src.agent.policy import PolicyService
from src.agent.recovery import RecoveryManager
from src.agent.verifier import VerifierService
from src.executor.browser import BrowserExecutor
from src.models.common import (
    LoopStage,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from src.models.logs import FailureRecord, ModelDebugArtifacts, StepLog
from src.models.policy import ActionType, AgentAction
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.verification import VerificationStatus
from src.store.run_logger import append_step_log
from src.store.run_store import RunStore


class AgentLoop:
    """Coordinates the MVP control loop and stop-before-send boundary."""

    def __init__(
        self,
        capture_service: CaptureService,
        perception_service: PerceptionService,
        run_store: RunStore,
        policy_service: PolicyService,
        browser_executor: BrowserExecutor,
        verifier_service: VerifierService,
        recovery_manager: RecoveryManager,
    ) -> None:
        self.capture_service = capture_service
        self.perception_service = perception_service
        self.run_store = run_store
        self.policy_service = policy_service
        self.browser_executor = browser_executor
        self.verifier_service = verifier_service
        self.recovery_manager = recovery_manager

    async def start_run(self, request: RunTaskRequest) -> RunResponse:
        record = self.run_store.create_run(intent=request.intent)
        return RunResponse(
            run_id=record.run_id,
            status=record.status,
            intent=record.intent,
            step_count=record.step_count,
        )

    async def run_live_benchmark(
        self,
        intent: str = "Create a Gmail draft and stop before send.",
        *,
        gmail_url: str = "https://mail.google.com/",
        max_steps: int = 12,
    ) -> RunResponse:
        response = await self.start_run(RunTaskRequest(intent=intent))
        await self.browser_executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url=gmail_url,
            )
        )

        while True:
            state = await self.run_store.get_run(response.run_id)
            if state is None:
                raise ValueError(f"Run {response.run_id!r} not found")
            if state.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
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

    async def step_run(self, request: StepRequest) -> RunResponse:
        record = await self.run_store.get_run(request.run_id)
        if record is None:
            raise ValueError(f"Run {request.run_id!r} not found")

        step_index = record.step_count + 1
        before_artifact_path = self._before_artifact_path(record.run_id, step_index)
        after_artifact_path = self._after_artifact_path(record.run_id, step_index)

        frame = await self.capture_service.capture(record)
        perception = await self.perception_service.perceive(frame, record)
        perception_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "perception", self.perception_service)
        state = await self.run_store.update_state(record.run_id, perception)
        decision = await self.policy_service.choose_action(state, perception)
        policy_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "policy", self.policy_service)
        state.current_subgoal = decision.active_subgoal
        executed_action = await self.browser_executor.execute(decision.action)
        executed_action = self._relocate_after_artifact(executed_action, after_artifact_path)
        verification = await self.verifier_service.verify(state, decision, executed_action)
        recovery: RecoveryDecision = await self.recovery_manager.recover(
            state,
            decision,
            executed_action,
            verification,
        )
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
            failure=failure,
        )
        append_step_log(self._run_log_path(record.run_id), step_log)

        state.action_history.append(executed_action)
        state.verification_history.append(verification)
        state.artifact_paths.extend(
            [
                before_artifact_path,
                after_artifact_path,
                perception_debug.prompt_artifact_path,
                perception_debug.raw_response_artifact_path,
                perception_debug.parsed_artifact_path,
                policy_debug.prompt_artifact_path,
                policy_debug.raw_response_artifact_path,
                policy_debug.parsed_artifact_path,
            ]
        )

        final_status = RunStatus.RUNNING
        if verification.stop_condition_met and verification.status is VerificationStatus.SUCCESS:
            final_status = RunStatus.SUCCEEDED
            state.stop_reason = StopReason.STOP_BEFORE_SEND
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
