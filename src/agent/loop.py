"""Agent loop orchestrator for the vision-driven automation workflow."""

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.agent.capture import CaptureService
from src.agent.perception import PerceptionLowQualityError, PerceptionService
from src.agent.policy import PolicyService
from src.agent.recovery import RecoveryManager, validate_benchmark_integrity
from src.agent.reflector import PostRunReflector
from src.agent.selector import DeterministicTargetSelector
from src.agent.verifier import VerifierService
from src.agent.video_verifier import VideoVerifier
from src.clients.gemini import GeminiClient
from src.core.contracts.perception import Environment as UnifiedEnvironment
from src.core.router import RoutingError
from src.executor.browser import Executor
from src.executor.browser_adapter import BrowserExecutor as UnifiedBrowserExecutor
from src.executor.desktop_adapter import DesktopExecutor as UnifiedDesktopExecutor
from src.models.capture import CaptureFrame
from src.models.common import (
    FailureCategory,
    LoopStage,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from src.models.execution import (
    AnchorSnapInfo,
    ExecutedAction,
    ExecutionReresolutionTrace,
)
from src.models.logs import (
    FailureRecord,
    ModelDebugArtifacts,
    PreStepFailureLog,
    StepLog,
)
from src.models.perception import UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.progress import ProgressTrace
from src.models.recovery import RecoveryDecision, RecoveryStrategy
from src.models.selector import TargetIntent, TargetIntentAction
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.runtime.legacy_adapter import LegacyOperonContractAdapter
from src.runtime.orchestrator import UnifiedOrchestrator
from src.runtime.state import AgentRuntimeState
from src.store.background_writer import bg_writer
from src.store.memory import MemoryStore
from src.store.run_logger import append_step_log, append_step_log_critical
from src.store.run_store import RunStore

logger = logging.getLogger(__name__)

_TRACE = os.getenv("OPERON_TRACE", "").lower() in {"1", "true", "yes"}
_LIVENESS_RETRY_MAX = 3
_LIVENESS_RETRY_FIRST_SLEEP_S = 0.5   # fast recovery for brief UI transitions (e.g. search overlay animating in)
_LIVENESS_RETRY_SLEEP_S = 1.5         # subsequent retries give slower loads more time


def _trace(stage: str, detail: str = "") -> None:
    if not _TRACE:
        return
    suffix = f"  {detail}" if detail else ""
    msg = f"[TRACE] {stage}{suffix}"
    try:
        print(f"\033[36m{msg}\033[0m", flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


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
        executor: Executor,
        verifier_service: VerifierService,
        recovery_manager: RecoveryManager,
        memory_store: MemoryStore | None = None,
        gemini_client: GeminiClient | None = None,
        environment: UnifiedEnvironment = UnifiedEnvironment.BROWSER,
        unified_orchestrator: UnifiedOrchestrator | None = None,
    ) -> None:
        self.capture_service = capture_service
        self.perception_service = perception_service
        self.run_store = run_store
        self.policy_service = policy_service
        self.executor = executor
        self.verifier_service = verifier_service
        self.recovery_manager = recovery_manager
        self.memory_store = memory_store
        self.gemini_client = gemini_client
        self.target_selector = DeterministicTargetSelector()
        self.video_verifier: VideoVerifier | None = (
            VideoVerifier(gemini_client) if isinstance(gemini_client, GeminiClient) else None
        )
        self.reflector: PostRunReflector | None = (
            PostRunReflector(memory_store) if memory_store is not None else None
        )
        self.environment = environment
        self.unified_orchestrator = unified_orchestrator or UnifiedOrchestrator()
        self.legacy_contract_adapter = LegacyOperonContractAdapter(environment=environment)
        self.unified_states: dict[str, AgentRuntimeState] = {}
        # Per-run anchor cache: element_id → (cx, cy) of the last successful click.
        # Used by _apply_coord_anchor to suppress small perception jitter on TYPE actions.
        self._coord_anchors: dict[str, dict[str, tuple[int, int]]] = {}
        if environment is UnifiedEnvironment.BROWSER:
            self.executor_adapter = UnifiedBrowserExecutor(executor)
        else:
            self.executor_adapter = UnifiedDesktopExecutor(executor)

    async def start_run(self, request: RunTaskRequest) -> RunResponse:
        _trace("START_RUN", f"intent={request.intent!r}  url={request.start_url!r}")
        # Minimize all windows (including the Pilot UI browser) for a clean desktop
        if hasattr(self.executor, "reset_desktop") and not self._test_safe_mode_enabled():
            await self.executor.reset_desktop()
        record = self.run_store.create_run(
            intent=request.intent,
            start_url=request.start_url,
            headless=request.headless,
            benchmark=request.benchmark,
        )
        _trace("START_RUN", f"run_id={record.run_id}")
        # Benchmark-safe session management: wipe per-run coordinator state (episodic
        # memory replay, hint cache, rule trace) so prior task context doesn't leak.
        # Browser auth state (cookies/session tokens) is preserved at the executor level.
        if hasattr(self.policy_service, "reset_run_context"):
            self.policy_service.reset_run_context(record.run_id)
        if hasattr(self.executor, "set_current_run_id"):
            self.executor.set_current_run_id(record.run_id)
        if hasattr(self.executor, "configure_run"):
            self.executor.configure_run(record.run_id, headless=request.headless)
        if hasattr(self.executor, "start_run_recording"):
            try:
                await self.executor.start_run_recording(record.run_id, root_dir=self.run_store.root_dir)
            except Exception as exc:
                logger.warning("start_run_recording failed for %s: %s", record.run_id, exc)
        if request.start_url:
            _trace("START_RUN -> NAVIGATE", request.start_url)
            await self.executor.execute(
                AgentAction(action_type=ActionType.NAVIGATE, url=request.start_url)
            )
        self.unified_states[record.run_id] = AgentRuntimeState(
            environment=self.environment,
            active_app="browser" if self.environment is UnifiedEnvironment.BROWSER else "desktop",
            current_url=request.start_url if self.environment is UnifiedEnvironment.BROWSER else None,
        )
        return RunResponse(
            run_id=record.run_id,
            status=record.status,
            intent=record.intent,
            step_count=record.step_count,
        )

    def unified_state_for_run(self, run_id: str) -> AgentRuntimeState | None:
        """Return the unified runtime state for one run, if present."""

        return self.unified_states.get(run_id)

    @staticmethod
    def _test_safe_mode_enabled() -> bool:
        return os.getenv("OPERON_TEST_SAFE_MODE", "false").lower() == "true"

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
            if state.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_FOR_USER, RunStatus.CANCELLED}:
                return response
            if state.step_count >= max_steps:
                state.stop_reason = StopReason.MAX_STEP_LIMIT_REACHED
                updated = await self.run_store.set_status(state.run_id, RunStatus.FAILED)
                try:
                    await self._cleanup_completed_run(state.run_id)
                except Exception as exc:
                    logger.warning("cleanup after max_steps failed: %s", exc)
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
        if record.status is RunStatus.CANCELLED:
            return RunResponse(
                run_id=record.run_id,
                status=record.status,
                intent=record.intent,
                step_count=record.step_count,
            )

        step_index = record.step_count + 1
        _trace("─" * 60)
        _trace("STEP", f"step={step_index}  run_id={record.run_id}")
        before_artifact_path = self._before_artifact_path(record.run_id, step_index)
        after_artifact_path = self._after_artifact_path(record.run_id, step_index)
        self._prepare_step_artifacts(record.run_id, step_index, before_artifact_path, after_artifact_path)

        if step_index == 1 and not self._test_safe_mode_enabled():
            logger.info("Bootstrap warmup: Allowing OS and Browser window to hydrate...")
            if (
                self.environment is UnifiedEnvironment.BROWSER
                and not record.start_url
                and hasattr(self.executor, "execute")
            ):
                logger.info("Bootstrap warmup: no start URL — navigating to google.com")
                await self.executor.execute(
                    AgentAction(action_type=ActionType.NAVIGATE, url="https://www.google.com")
                )
            await asyncio.sleep(2.5)

        if record.force_fresh_perception:
            record.force_fresh_perception = False
            await asyncio.sleep(0.5)  # let UI settle after visual perturbation

        _trace("  1 CAPTURE", "taking screenshot via CaptureService")
        _t0 = time.perf_counter()
        frame = await self.capture_service.capture(record)
        _trace("  1 CAPTURE OK", f"{time.perf_counter()-_t0:.2f}s  frame={type(frame).__name__}")

        # High-velocity guard: if the screen changed by >5% since the last burst
        # capture, the UI underwent a major transition (app switch, navigation, modal
        # opening). Stale element coordinates in the rolling buffer would produce
        # ghost elements from the previous context — clear it now to prevent
        # cross-app state contamination.
        if frame.visual_velocity > 0.05:
            _element_buffer = getattr(self.perception_service, "element_buffer", None)
            if _element_buffer is not None:
                _element_buffer.clear()
                logger.info(
                    "element_buffer cleared: visual_velocity=%.3f > 0.05 — "
                    "major UI transition detected, stale ghost elements purged (run=%s)",
                    frame.visual_velocity, record.run_id[:8],
                )
        # Inject memory hints before perceive() so combined mode includes them
        if hasattr(self.policy_service, "prepare_hints"):
            self.policy_service.prepare_hints(record, None)
        # Inject stagnation hint if subgoal hasn't changed for 3+ steps
        self._inject_stagnation_hint(record)
        _trace("  2 PERCEIVE", f"sending screenshot to Gemini  backend={type(self.perception_service).__name__}")
        _t0 = time.perf_counter()
        try:
            perception = await self._perceive_with_liveness_retry(record, frame)
        except Exception as exc:
            _trace("  2 PERCEIVE FAIL", str(exc))
            return await self._record_pre_step_perception_failure(
                record=record,
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                error=exc,
            )
        _trace("  2 PERCEIVE OK", f"{time.perf_counter()-_t0:.2f}s  page_hint={perception.page_hint.value!r}  elements={len(perception.visible_elements)}")
        perception = self._infer_focused_element(record, perception)
        perception_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "perception", self.perception_service)
        state = await self.run_store.update_state(record.run_id, perception)
        self._sync_progress_state_with_perception(state, perception)
        _trace("  3 POLICY", "PolicyCoordinator -> rule engine first, then LLM fallback")
        _t0 = time.perf_counter()
        decision = await self.policy_service.choose_action(state, perception)
        _enriched = self._attach_target_context(decision.action, perception, state.benchmark)
        if _enriched is not decision.action:
            decision = decision.model_copy(update={"action": _enriched})
        _trace("  3 POLICY OK", f"{time.perf_counter()-_t0:.2f}s  action={decision.action.action_type.value!r}  target={decision.action.target_element_id!r}  rationale={decision.rationale[:80]!r}")
        policy_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "policy", self.policy_service)
        state.current_subgoal = decision.active_subgoal
        _anchor_snap_info: AnchorSnapInfo | None = None
        decision, _anchor_snap_info = self._apply_coord_anchor(record.run_id, decision)
        decision = self._tag_input_zone(decision, perception)

        # Human-in-the-loop: pause the run and wait for user input
        if decision.action.action_type is ActionType.WAIT_FOR_USER:
            _trace("  3 POLICY -> WAIT_FOR_USER", "pausing run")
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

        # Retry Exhaustion / Coordinate Thrash guard: if this action targets a (x, y)
        # that has already failed 3 times, escalate to HITL instead of attempting again.
        if self._coordinate_thrash_detected(state, decision.action):
            _trace("  3 POLICY -> COORDINATE_THRASH_HITL", "3 coord failures at same point — escalating to HITL")
            thrash_x, thrash_y = decision.action.x, decision.action.y
            hitl_decision = decision.model_copy(update={
                "action": AgentAction(
                    action_type=ActionType.WAIT_FOR_USER,
                    text=(
                        f"Coordinate thrashing detected: the target at ({thrash_x}, {thrash_y}) "
                        "has failed 3 consecutive times. The element may have shifted, be obscured, "
                        "or be unreachable at these coordinates. Please manually verify the target "
                        "location and click Resume when ready."
                    ),
                )
            })
            return await self._pause_for_user(
                record=record,
                state=state,
                perception=perception,
                decision=hitl_decision,
                perception_debug=perception_debug,
                policy_debug=policy_debug,
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                after_artifact_path=after_artifact_path,
            )

        # Set run context on the executor so launched processes are tracked per run
        if hasattr(self.executor, "set_current_run_id"):
            self.executor.set_current_run_id(record.run_id)
        unified_state = self.unified_states.get(record.run_id)
        adaptation_trace: list[str] = []
        retries_used = 0
        attempt_index = 1

        while True:
            _trace("  4 EXECUTE", f"attempt={attempt_index}  action={decision.action.action_type.value!r}  executor={type(self.executor).__name__}")
            blocked_action = self._block_redundant_action(state, decision.action, step_index)
            if blocked_action is None:
                _t0 = time.perf_counter()
                state, executed_action = await self._execute_with_hardening(
                    record=record,
                    state=state,
                    perception=perception,
                    decision_action=decision.action,
                    step_index=step_index,
                    use_unified_executor=True,
                )
                _trace("  4 EXECUTE OK" if executed_action.success else "  4 EXECUTE FAIL", f"{time.perf_counter()-_t0:.2f}s  success={executed_action.success}  detail={executed_action.detail!r}")
                if executed_action.success and decision.action.action_type is ActionType.CLICK:
                    self._update_coord_anchor(record.run_id, executed_action.action)
            else:
                _trace("  4 EXECUTE blocked", f"redundant action suppressed: {blocked_action.detail!r}")
                executed_action = blocked_action
            executed_action = self._relocate_after_artifact(executed_action, after_artifact_path)

            # If the executor intercepted a JS dialog (alert/confirm/prompt),
            # inject the dialog message into the local perception summary so the
            # verifier and success-stop rule can detect form-submission confirmations.
            # Only updates in-memory state — avoids a second run_store.update_state call.
            if hasattr(self.executor, "pop_last_dialog_message"):
                _dialog_msg = self.executor.pop_last_dialog_message(record.run_id)
                if _dialog_msg:
                    logger.info("JS dialog captured: %r — injecting into perception summary", _dialog_msg)
                    perception = perception.model_copy(
                        update={"summary": (perception.summary or "") + f" {_dialog_msg}"}
                    )
                    if state.observation_history:
                        _obs = list(state.observation_history)
                        _obs[-1] = perception
                        state = state.model_copy(update={"observation_history": _obs})

            _trace("  5 VERIFY", "DeterministicVerifierService checking outcome")
            verification = await self.verifier_service.verify(state, decision, executed_action)
            if verification.status is VerificationStatus.PENDING:
                verification = await self._wait_for_page_load(
                    record=record,
                    state=state,
                    decision=decision,
                    executed_action=executed_action,
                    before_artifact_path=before_artifact_path,
                )
            elif verification.status is VerificationStatus.STABLE_WAIT:
                verification = await self._wait_for_ui_stable(
                    record=record,
                    state=state,
                    decision=decision,
                    executed_action=executed_action,
                )
            _trace("  5 VERIFY OK", f"status={verification.status.value!r}  stop={verification.stop_condition_met}  stop_reason={verification.stop_reason!r}")
            verification_debug = self._resolve_model_debug_artifacts(record.run_id, step_index, "verification", self.verifier_service)

            # Compute screen change ratio once; reused by video verify and progress tracking.
            from src.agent.screen_diff import compute_screen_change_ratio as _scr
            _after_path = executed_action.artifact_path
            _screen_change_ratio: float | None = (
                _scr(before_artifact_path, _after_path)
                if before_artifact_path and _after_path
                else None
            )

            # Video verification: when screen didn't change, record and ask Gemini.
            # Skipped when the verifier's reaction check already set video_verified=True.
            if not verification.stop_condition_met and not verification.video_verified and self.video_verifier is not None:
                video_result = await self._maybe_video_verify(
                    state=state,
                    decision=decision,
                    executed_action=executed_action,
                    before_artifact_path=before_artifact_path,
                    step_index=step_index,
                    screen_change_ratio=_screen_change_ratio,
                )
                if video_result is not None:
                    _trace("  5b VIDEO_VERIFY OK", f"status={video_result.status.value!r}")
                    verification = video_result

            _trace("  6 UNIFIED_CONTRACT", "translating step -> contracts via LegacyOperonContractAdapter -> UnifiedOrchestrator")
            try:
                unified_step = self._record_unified_step(
                    record=record,
                    state=state,
                    perception=perception,
                    decision=decision,
                    executed_action=executed_action,
                    verification=verification,
                    unified_state=unified_state,
                    attempt_index=attempt_index,
                )
                unified_state = unified_step.after
                self.unified_states[record.run_id] = unified_state
                strategy = self.unified_orchestrator.adaptation_strategy_for(unified_step.critic.failure_type)
                _trace("  6 UNIFIED_CONTRACT OK", f"critic={unified_step.critic.outcome.value!r}  failure={unified_step.critic.failure_type!r}  strategy={strategy!r}")
            except (RoutingError, ValueError) as _exc:
                # RoutingError: policy returned an action that violates environment routing rules.
                # ValueError: terminal actions (stop, etc.) not representable in the unified contract.
                # Degrade gracefully: keep the current unified_state and break the retry loop.
                _trace("  6 UNIFIED_CONTRACT skipped", f"{type(_exc).__name__}: {_exc}")
                strategy = None
                unified_step = None
            # The unified layer may request an in-step adaptation retry, but only
            # when (a) verification produced a hard FAILURE — not merely UNCERTAIN,
            # which the outer recovery stage handles — and (b) the legacy hardening
            # path did not already retry this action (e.g. stale-target
            # re-resolution, successful or failed). Looping capture/perceive/choose
            # inside a single step on UNCERTAIN verifications breaks the single-
            # iteration contract that legacy tests rely on.
            legacy_retry_attempted = bool(
                executed_action.execution_trace is not None
                and executed_action.execution_trace.retry_attempted
            )
            allow_adaptation_retry = (
                verification.status is VerificationStatus.FAILURE
                and not legacy_retry_attempted
            )
            if (
                unified_step is None
                or unified_step.critic.outcome.value != "retry"
                or strategy is None
                or not allow_adaptation_retry
            ):
                if retries_used > 0 and unified_step is not None:
                    unified_state = unified_state.model_copy(
                        update={
                            "retry_count": retries_used,
                            "last_strategy": adaptation_trace[-1],
                            "last_failure_type": None
                            if unified_step.critic.outcome.value == "success"
                            else unified_step.critic.failure_type,
                        }
                    )
                    self.unified_states[record.run_id] = unified_state
                _trace("  RETRY_LOOP -> break", f"outcome={unified_step.critic.outcome.value if unified_step else 'n/a'}  retries_used={retries_used}")
                break
            if retries_used >= self.unified_orchestrator.max_retries:
                _trace("  RETRY_LOOP -> max_retries", f"retries_used={retries_used}")
                break

            retries_used += 1
            adaptation_trace.append(strategy)
            _trace("  RETRY_LOOP -> retry", f"retries_used={retries_used}  strategy={strategy!r}")
            unified_state = unified_state.apply_retry_feedback(
                perception=unified_step.perception,
                planner=unified_step.planner,
                actor=unified_step.actor,
                critic=unified_step.critic,
                retry_count=retries_used,
                strategy=strategy,
            )
            self.unified_states[record.run_id] = unified_state

            frame = await self.capture_service.capture(state)
            if hasattr(self.policy_service, "prepare_hints"):
                self.policy_service.prepare_hints(state, None)
            self._inject_stagnation_hint(state)
            perception = await self._perceive_with_liveness_retry(state, frame)
            perception = self._infer_focused_element(state, perception)
            state.observation_history.append(perception)
            self._sync_progress_state_with_perception(state, perception)
            decision = await self.policy_service.choose_action(state, perception)
            _enriched = self._attach_target_context(decision.action, perception, state.benchmark)
            if _enriched is not decision.action:
                decision = decision.model_copy(update={"action": _enriched})
            state.current_subgoal = decision.active_subgoal
            decision, _retry_snap = self._apply_coord_anchor(record.run_id, decision)
            if _retry_snap is not None:
                _anchor_snap_info = _retry_snap
            attempt_index += 1

        _trace("  7 RECOVER", f"RuleBasedRecoveryManager  verification={verification.status.value!r}")
        recovery: RecoveryDecision = await self.recovery_manager.recover(
            state,
            decision,
            executed_action,
            verification,
        )
        # Benchmark Integrity Check: reject recovery decisions that bypass verification
        # or claim success without visual confirmation.
        recovery = validate_benchmark_integrity(recovery, verification)
        _trace("  7 RECOVER OK", f"strategy={recovery.strategy.value!r}  stop_reason={recovery.stop_reason!r}")
        progress_trace = self._update_progress_state(
            state=state,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            recovery=recovery,
            step_index=step_index,
            before_artifact_path=before_artifact_path,
            screen_change_ratio=_screen_change_ratio,
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
        if _anchor_snap_info is not None:
            executed_action = executed_action.model_copy(update={"anchor_snap": _anchor_snap_info})

        failure = self._build_failure_record(state, decision, executed_action, verification, recovery)

        _decision_source = f"[RULE] {decision.rule_name}" if decision.rule_name else "[LLM] gemini"
        step_log = StepLog(
            run_id=record.run_id,
            step_id=f"step_{step_index}",
            step_index=step_index,
            before_artifact_path=before_artifact_path,
            after_artifact_path=after_artifact_path,
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            verification_debug=verification_debug,
            perception=perception,
            policy_decision=decision,
            executed_action=executed_action,
            verification_result=verification,
            recovery_decision=recovery,
            progress_state=state.progress_state,
            progress_trace_artifact_path=progress_trace_artifact_path,
            failure=failure,
            decision_source=_decision_source,
            visual_variance=executed_action.visual_variance,
        )
        append_step_log(self._run_log_path(record.run_id), step_log)

        # Rule-Augmented Generation: record rule outcome so the next LLM prompt knows
        # what the rule tried. Clear the trace when the LLM (not a rule) decided this step.
        if decision.rule_name:
            state.last_rule_trace = (
                f"[RULE TRACE] Rule '{decision.rule_name}' fired at step {state.step_count} | "
                f"action={decision.action.action_type.value} | "
                f"outcome={verification.status.value}"
            )
        else:
            state.last_rule_trace = None

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
                *( [perception_debug.usage_artifact_path] if perception_debug.usage_artifact_path else [] ),
                *( [perception_debug.diagnostics_artifact_path] if perception_debug.diagnostics_artifact_path else [] ),
                policy_debug.prompt_artifact_path,
                policy_debug.raw_response_artifact_path,
                policy_debug.parsed_artifact_path,
                *( [policy_debug.usage_artifact_path] if policy_debug.usage_artifact_path else [] ),
                verification_debug.prompt_artifact_path,
                verification_debug.raw_response_artifact_path,
                verification_debug.parsed_artifact_path,
                *( [verification_debug.usage_artifact_path] if verification_debug.usage_artifact_path else [] ),
                *( [verification_debug.diagnostics_artifact_path] if verification_debug.diagnostics_artifact_path else [] ),
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

        if final_status is RunStatus.RUNNING:
            await self._apply_recovery_actions(state, recovery)

        _trace("  8 LOG", f"StepLog -> run.jsonl  final_status={final_status.value!r}")
        updated = await self.run_store.set_status(record.run_id, final_status)
        if final_status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
            self._coord_anchors.pop(record.run_id, None)
            try:
                await self._cleanup_completed_run(record.run_id)
            except Exception as exc:
                logger.warning("cleanup after %s failed, continuing: %s", final_status.value, exc)

        # Post-run reflection: analyze completed/failed runs and learn
        if final_status in {RunStatus.SUCCEEDED, RunStatus.FAILED} and self.reflector is not None:
            _trace("  9 REFLECT", "PostRunReflector analyzing run -> MemoryRecord")
            try:
                self.reflector.reflect(record.run_id)
                _trace("  9 REFLECT OK")
            except Exception as exc:
                logger.warning("post-run reflection failed for %s: %s", record.run_id, exc)
                _trace("  9 REFLECT ERROR", str(exc))

        _trace("STEP DONE", f"step={step_index}  status={final_status.value!r}")
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
        """Pause the run, generate an LLM explanation, notify the human, and start the escalation timer."""
        from src.agent.hitl import (
            generate_hitl_message,
            notify_desktop,
            post_hitl_webhook,
            start_escalation_timer,
        )

        # Generate a human-readable LLM explanation of why we're pausing.
        element_names = [e.primary_name for e in perception.visible_elements[:12]]
        hitl_message = await generate_hitl_message(
            intent=state.intent,
            page_hint=perception.page_hint.value,
            url=state.start_url,
            visible_element_names=element_names,
            gemini_client=self.gemini_client,
        )
        state.hitl_message = hitl_message

        executed_action = ExecutedAction(
            action=decision.action,
            success=True,
            detail=f"Paused for user: {decision.action.text}",
        )
        verification = await self.verifier_service.verify(state, decision, executed_action)
        recovery = RecoveryDecision(
            strategy=RecoveryStrategy.STOP,
            message=hitl_message,
            failure_category=FailureCategory.HUMAN_INTERVENTION_REQUIRED,
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
                decision_source=f"[RULE] {decision.rule_name}" if decision.rule_name else "[LLM] gemini",
            ),
        )
        state.stop_reason = StopReason.WAITING_FOR_USER
        state.step_count = step_index
        await self.run_store.save_state(state)
        updated = await self.run_store.set_status(record.run_id, RunStatus.WAITING_FOR_USER)

        # Fire desktop notification (best-effort).
        notify_desktop(
            title="Operon needs you",
            message=hitl_message[:120],
        )

        # Start escalating re-notification in the background.
        run_store = self.run_store
        run_id = record.run_id

        # POST webhook (reliable channel — always attempted, always logged).
        asyncio.create_task(
            post_hitl_webhook(
                run_id=run_id,
                intent=state.intent,
                page_hint=perception.page_hint.value,
                message=hitl_message,
                url=state.start_url,
            )
        )

        async def _get_status():
            s = await run_store.get_run(run_id)
            return s.status.value if s else None

        def _re_notify():
            notify_desktop("Operon still waiting", f"Run {run_id[:8]}… still needs your help.")

        task = asyncio.create_task(
            start_escalation_timer(run_id, get_status_fn=_get_status, notify_fn=_re_notify)
        )
        task.add_done_callback(
            lambda t: logger.warning("HITL escalation error: %s", t.exception())
            if not t.cancelled() and t.exception() is not None else None
        )

        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    async def _perceive_with_liveness_retry(self, state, initial_frame: CaptureFrame):
        """Perceive with flag-based liveness retries for transient zero-element frames.

        When perception returns is_empty_frame=True (blank/loading screen), re-capture
        and retry up to _LIVENESS_RETRY_MAX times before raising PerceptionLowQualityError.
        Other quality failures (unlabeled elements, etc.) propagate immediately.
        The liveness_retries count is stamped onto the returned ScreenPerception so it
        appears in run.jsonl for observability.
        """
        frame = initial_frame
        for attempt in range(1, _LIVENESS_RETRY_MAX + 1):
            result = await self.perception_service.perceive(frame, state)
            if not result.is_empty_frame:
                liveness_retries = attempt - 1
                if liveness_retries > 0:
                    logger.info("liveness_retry: resolved after %d retries", liveness_retries)
                    return result.model_copy(update={"liveness_retries": liveness_retries})
                return result

            if attempt == _LIVENESS_RETRY_MAX:
                logger.error(
                    "perception_low_quality: no visible elements after %d liveness retries — raising terminal signal",
                    _LIVENESS_RETRY_MAX,
                )
                raise PerceptionLowQualityError(
                    f"no visible elements after {_LIVENESS_RETRY_MAX} liveness retries"
                )
            _sleep_s = _LIVENESS_RETRY_FIRST_SLEEP_S if attempt == 1 else _LIVENESS_RETRY_SLEEP_S
            logger.warning(
                "liveness_retry=%d/%d: zero elements — waiting %.1fs before recapture",
                attempt,
                _LIVENESS_RETRY_MAX,
                _sleep_s,
            )
            await asyncio.sleep(_sleep_s)
            frame = await self.capture_service.capture(state)
        raise PerceptionLowQualityError("no visible elements after all liveness retries")  # pragma: no cover

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
            reason=str(error) or type(error).__name__,
            stop_reason=stop_reason,
        )
        append_step_log_critical(
            self._run_log_path(record.run_id),
            PreStepFailureLog(
                run_id=record.run_id,
                step_id=f"step_{step_index}",
                step_index=step_index,
                before_artifact_path=before_artifact_path,
                perception_debug=perception_debug,
                failure=failure,
                error_message=str(error) or type(error).__name__,
            ),
        )
        record.artifact_paths.extend(
            [
                before_artifact_path,
                perception_debug.prompt_artifact_path,
                perception_debug.raw_response_artifact_path,
                perception_debug.parsed_artifact_path,
                *( [perception_debug.usage_artifact_path] if perception_debug.usage_artifact_path else [] ),
                *( [perception_debug.diagnostics_artifact_path] if perception_debug.diagnostics_artifact_path else [] ),
                *( [perception_debug.retry_log_artifact_path] if perception_debug.retry_log_artifact_path else [] ),
            ]
        )
        record.stop_reason = stop_reason
        updated = await self.run_store.set_status(record.run_id, RunStatus.FAILED)
        try:
            await self._cleanup_completed_run(record.run_id)
        except Exception as exc:
            logger.warning("cleanup after pre-step failure: %s", exc)
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

    async def _maybe_video_verify(
        self,
        *,
        state,
        decision,
        executed_action,
        before_artifact_path: str,
        step_index: int,
        screen_change_ratio: float | None = None,
    ) -> VerificationResult | None:
        """Record a video and verify with Gemini when screen_diff shows no change.

        Gate: only fires when expected_change ∈ {content, navigation, dialog} AND
        screen_change_ratio < SCREEN_CHANGE_THRESHOLD. Actions expected to produce
        no pixel change (none, focus) are correct with low delta — video-verifying
        them wastes Gemini calls.
        """
        if not executed_action.success:
            return None

        after_path = executed_action.artifact_path
        if not before_artifact_path or not after_path:
            return None

        from src.agent.screen_diff import SCREEN_CHANGE_THRESHOLD
        from src.models.policy import ExpectedChange

        if screen_change_ratio is None:
            from src.agent.screen_diff import compute_screen_change_ratio
            screen_change_ratio = compute_screen_change_ratio(before_artifact_path, after_path)

        ratio = screen_change_ratio
        if ratio >= SCREEN_CHANGE_THRESHOLD:
            return None  # Screen changed — no video needed

        # Gate: video-verify only when the planner expected a visible change.
        # none/focus are correct with low pixel delta — skip without recording.
        _VIDEO_VERIFY_EXPECTATIONS = {
            ExpectedChange.CONTENT, ExpectedChange.NAVIGATION, ExpectedChange.DIALOG,
        }
        expected_change = decision.expected_change
        if expected_change not in _VIDEO_VERIFY_EXPECTATIONS:
            _trace(
                "  5b VIDEO_VERIFY SKIPPED",
                f"expected_change={expected_change!r}  ratio={ratio:.4f}  "
                "low-change expectation is correct — no Gemini call",
            )
            return None

        # Belt-and-suspenders: re-execution of TYPE/DRAG/SELECT could double the action.
        action = decision.action
        if action.action_type in {ActionType.TYPE, ActionType.DRAG, ActionType.SELECT}:
            _trace(
                "  5b VIDEO_VERIFY SKIPPED",
                f"action_type={action.action_type.value!r} is non-idempotent — skipping re-execution",
            )
            return None

        if not hasattr(self.executor, "execute_with_recording"):
            return None

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "No screen change (ratio=%.4f, expected_change=%r) at step %d — recording video for verification",
            ratio,
            expected_change,
            step_index,
        )

        step_dir = Path(before_artifact_path).parent
        try:
            replay_result, video_path = await self.executor.execute_with_recording(action, step_dir)
        except Exception:
            logger.debug("Video recording failed", exc_info=True)
            return None

        if video_path is None:
            _trace("  5b VIDEO_VERIFY SKIPPED", "browser executor returned video_path=None (no screen recording)")
            return None

        # Update executed_action recording path (informational)
        executed_action = executed_action.model_copy(update={"recording_path": str(video_path)})

        video_result = await self.video_verifier.verify_action(
            video_path=video_path, action=action, intent=state.intent,
        )

        confidence = video_result.confidence_score
        motion_detail = f"[{video_result.motion_class}, confidence={confidence:.2f}] {video_result.what_happened}"

        if confidence < 0.2:
            # Hung app or Gemini confirmed no effect — hard failure.
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason=f"Video showed no effect: {motion_detail}",
                failure_type=VerificationFailureType.ACTION_FAILED,
                recovery_hint="retry_same_step",
                video_verified=True,
                video_detail=video_result.suggested_next_action,
                failure_category=FailureCategory.EXECUTION_ERROR,
                failure_stage=LoopStage.VERIFY,
            )

        if confidence < 0.5:
            # Loading spinner or ambiguous — stay uncertain so recovery can wait.
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason=f"Video detected loading motion; waiting for completion: {motion_detail}",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                recovery_hint="wait_and_retry",
                video_verified=True,
                video_detail=video_result.what_happened,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )

        # High confidence: action produced directed progress.
        if self._passive_wait_needs_more_signal(state, action, video_result.what_happened):
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason=f"Video verified the wait action, but it did not reveal enough new browser-task signal: {motion_detail}",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                video_verified=True,
                video_detail=video_result.what_happened,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )
        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason=f"Video verified: {motion_detail}",
            recovery_hint="advance",
            video_verified=True,
            video_detail=video_result.what_happened,
        )

    @classmethod
    def _passive_wait_needs_more_signal(
        cls,
        state,
        action: AgentAction,
        video_detail: str,
    ) -> bool:
        if action.action_type is not ActionType.WAIT:
            return False
        intent = state.intent.lower()
        if not any(token in intent for token in ("inspect", "look", "find", "open", "check", "view", "read")):
            return False
        if not cls._latest_observation_lacks_browser_signal(state):
            return False
        detail = video_detail.lower()
        return any(
            token in detail
            for token in (
                "loading",
                "still loading",
                "spinner",
                "blank",
                "empty",
                "waiting",
                "redirect",
                "navigating",
                "transition",
            )
        )

    @staticmethod
    def _latest_observation_lacks_browser_signal(state) -> bool:
        if not state.observation_history:
            return True
        latest = state.observation_history[-1]
        if latest.visible_elements:
            return False
        return str(latest.page_hint) == "unknown"

    async def _wait_for_ui_stable(
        self,
        *,
        record,
        state,
        decision,
        executed_action,
    ) -> VerificationResult:
        """Single 200ms re-verify for STABLE_WAIT (UI actively transitioning post-action).

        Waits 200ms, re-captures, re-perceives, and calls verify() once more.
        If the screen has settled the verifier will return its normal verdict.
        If the UI is still moving (STABLE_WAIT again) or the outcome is still
        uncertain, we downgrade to UNCERTAIN so the recovery ladder can decide.
        """
        from src.agent.perception import PerceptionLowQualityError

        _trace("  5-STABLE_WAIT", "UI still transitioning — waiting 200ms before re-verify")
        logger.info("stable_wait: UI in motion — waiting 200ms before re-verifying (run=%s)", record.run_id[:8])
        await asyncio.sleep(0.2)

        try:
            frame = await self.capture_service.capture(record)
            fresh_perception = await self._perceive_with_liveness_retry(record, frame)
        except PerceptionLowQualityError:
            logger.info("stable_wait: perception still unstable after 200ms — downgrading to UNCERTAIN")
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="UI still transitioning after 200ms stability wait; perception quality too low.",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )
        except Exception as exc:
            logger.warning("stable_wait: re-capture/perceive failed: %s", exc)
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason=f"UI stability wait failed during re-capture: {exc}",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )

        fresh_perception = self._infer_focused_element(record, fresh_perception)
        state.observation_history.append(fresh_perception)
        self._sync_progress_state_with_perception(state, fresh_perception)
        _trace("  5-STABLE_WAIT re-verify", f"elements={len(fresh_perception.visible_elements)} hint={fresh_perception.page_hint.value!r}")

        verification = await self.verifier_service.verify(state, decision, executed_action)
        if verification.status is VerificationStatus.STABLE_WAIT:
            # Still moving after one retry — downgrade to UNCERTAIN, let recovery decide.
            logger.info("stable_wait: still transitioning after re-verify — downgrading to UNCERTAIN")
            return verification.model_copy(update={
                "status": VerificationStatus.UNCERTAIN,
                "reason": "UI still transitioning after 200ms stability wait; treating as uncertain.",
            })

        _trace("  5-STABLE_WAIT resolved", f"status={verification.status.value!r}")
        logger.info("stable_wait: resolved to %s after re-verify", verification.status.value)
        return verification

    _PENDING_BACKOFF_DELAYS: tuple[float, ...] = (2.0, 4.0, 8.0)

    async def _wait_for_page_load(
        self,
        *,
        record,
        state,
        decision,
        executed_action,
        before_artifact_path: str,
    ) -> VerificationResult:
        """Exponential-backoff re-verify loop for PENDING (page-loading) states.

        Waits 2s → 4s → 8s (14s total) between re-captures.  Downgrades to
        UNCERTAIN if the page never settles within the budget.
        """
        from src.agent.perception import PerceptionLowQualityError

        retries = 0
        for delay in self._PENDING_BACKOFF_DELAYS:
            retries += 1
            _trace("  5-PENDING wait", f"retry={retries} delay={delay:.0f}s")
            logger.info("patience_retry=%d: waiting %.0fs for page load (run=%s)", retries, delay, record.run_id[:8])
            await asyncio.sleep(delay)
            try:
                frame = await self.capture_service.capture(record)
                fresh_perception = await self._perceive_with_liveness_retry(record, frame)
            except PerceptionLowQualityError:
                # Zero-element liveness retries exhausted — page still blank; keep waiting.
                logger.info("patience_retry=%d: perception still blank after liveness retries", retries)
                continue
            except Exception as exc:
                logger.warning("_wait_for_page_load: perception error: %s", exc)
                continue

            if fresh_perception.is_empty_frame:
                # Guard: shouldn't reach here after liveness retries, but handle defensively.
                logger.info("patience_retry=%d: empty frame — still loading", retries)
                continue

            if fresh_perception.liveness_retries > 0:
                logger.info(
                    "patience_retry=%d: page settled after %d liveness retries",
                    retries, fresh_perception.liveness_retries,
                )

            # Update state observation so the verifier sees fresh elements.
            fresh_perception = self._infer_focused_element(record, fresh_perception)
            state.observation_history.append(fresh_perception)
            self._sync_progress_state_with_perception(state, fresh_perception)
            _trace("  5-PENDING re-verify", f"retry={retries} elements={len(fresh_perception.visible_elements)} hint={fresh_perception.page_hint.value!r}")
            verification = await self.verifier_service.verify(state, decision, executed_action)
            if verification.status is not VerificationStatus.PENDING:
                _trace("  5-PENDING resolved", f"retry={retries} status={verification.status.value!r}")
                logger.info("patience_retry=%d: resolved to %s", retries, verification.status.value)
                return verification.model_copy(update={"patience_retries": retries})

        # Budget exhausted — page never settled; downgrade to UNCERTAIN.
        _trace("  5-PENDING timeout", f"exhausted after {retries} retries — downgrading to UNCERTAIN")
        logger.warning("patience_retry: budget exhausted after %d retries (run=%s)", retries, record.run_id[:8])
        return VerificationResult(
            status=VerificationStatus.UNCERTAIN,
            expected_outcome_met=False,
            stop_condition_met=False,
            reason=f"Page did not settle within the 14s loading budget ({retries} patience retries); treating as uncertain.",
            failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
            failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
            failure_stage=LoopStage.VERIFY,
            patience_retries=retries,
        )

    async def _apply_recovery_actions(self, state, recovery: RecoveryDecision) -> None:
        """Execute concrete side effects for CONTEXT_RESET and SESSION_RESET tiers.

        Runs after the step is logged so the log accurately reflects the failure,
        and before set_status so the next step's capture sees the cleaned state.
        Failures are swallowed — a broken reset must never terminate a live run.
        """
        strategy = recovery.strategy
        if strategy is RecoveryStrategy.CONTEXT_RESET:
            _trace("  7b CONTEXT_RESET", "escape×2 + body-click + scroll-top")
            try:
                await self.executor.context_reset()
            except Exception as exc:
                logger.warning("context_reset executor call failed: %s", exc)
        elif strategy is RecoveryStrategy.SESSION_RESET:
            _trace("  7b SESSION_RESET", f"start_url={state.start_url!r}")
            try:
                await self.executor.session_reset(state.start_url)
            except Exception as exc:
                logger.warning("session_reset executor call failed: %s", exc)

    async def _cleanup_completed_run(self, run_id: str) -> None:
        if hasattr(self.executor, "stop_run_recording"):
            try:
                await self.executor.stop_run_recording(run_id)
            except Exception as exc:
                logger.warning("stop_run_recording failed for %s: %s", run_id, exc)
        if not hasattr(self.executor, "aclose_run"):
            await self._persist_cleanup_artifacts(run_id)
            return
        try:
            await self.executor.aclose_run(run_id)
        except Exception as exc:
            logger.warning("cleanup_completed_run: aclose_run failed for %s: %s", run_id, exc)
        await self._persist_cleanup_artifacts(run_id)

    async def _persist_cleanup_artifacts(self, run_id: str) -> None:
        if not hasattr(self.executor, "recorded_video_path_for_run"):
            return
        try:
            video_path = self.executor.recorded_video_path_for_run(run_id)
        except Exception:
            return
        if video_path is None:
            return
        state = await self.run_store.get_run(run_id)
        if state is None:
            return
        video_str = str(video_path)
        if video_str in state.artifact_paths:
            return
        state.artifact_paths.append(video_str)
        await self.run_store.save_state(state)

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
            parsed_artifact_path=str(step_dir / ("policy_decision.json" if stage_name == "policy" else "verification_result.json" if stage_name == "verification" else "perception_parsed.json")),
            diagnostics_artifact_path=str(step_dir / f"{stage_name}_diagnostics.json") if stage_name in {"perception", "verification"} else None,
        )

    async def _execute_with_hardening(
        self,
        *,
        record,
        state,
        perception,
        decision_action,
        step_index: int,
        use_unified_executor: bool = False,
    ):
        executor_runner = self.executor_adapter if use_unified_executor else self.executor
        executed_action = await executor_runner.execute(decision_action)
        executed_action = self._apply_no_progress_detection(state, executed_action)

        if not self._should_retry_execution(executed_action):
            return state, self._persist_execution_trace(record.run_id, step_index, executed_action)

        retry_action = decision_action
        retry_state = state
        target_reresolved = False
        try:
            frame = await self.capture_service.capture(record)
            # In combined mode, don't overwrite the primary cached decision
            if hasattr(self.perception_service, "set_perception_only"):
                self.perception_service.set_perception_only(True)
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
        finally:
            if hasattr(self.perception_service, "set_perception_only"):
                self.perception_service.set_perception_only(False)

        if retry_action is None:
            failed = self._apply_reresolution_failure(executed_action, resolution.trace)
            return retry_state, self._persist_execution_trace(record.run_id, step_index, failed)

        retry_executed = await executor_runner.execute(retry_action)
        retry_executed = self._merge_execution_retry(
            original=executed_action,
            retried=retry_executed,
            retry_reason=executed_action.failure_category,
            target_reresolved=target_reresolved,
            reresolution_trace=resolution.trace,
        )
        retry_executed = self._apply_no_progress_detection(state, retry_executed)
        return retry_state, self._persist_execution_trace(record.run_id, step_index, retry_executed)

    def _record_unified_step(
        self,
        *,
        record,
        state,
        perception,
        decision,
        executed_action,
        verification,
        unified_state: AgentRuntimeState | None,
        attempt_index: int,
    ):
        bundle = self.legacy_contract_adapter.bundle(
            state=state,
            perception=perception,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            attempt_index=attempt_index,
        )
        return self.unified_orchestrator.process_step(
            perception=bundle.perception,
            planner=bundle.planner,
            actor=bundle.actor,
            critic=bundle.critic,
            current_state=unified_state,
        )

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

    # ------------------------------------------------------------------
    # Coordinate anchor cache
    # ------------------------------------------------------------------

    _ANCHOR_SNAP_THRESHOLD_PX: int = 50

    # Element-id substrings and element_type values that indicate a valid but
    # blank interactive surface (text editors, input fields, search boxes).
    _INPUT_ZONE_ID_TOKENS: frozenset[str] = frozenset({
        "text_area", "text_field", "text_input", "notepad_text_area",
        "input", "textarea", "editor", "search_input",
    })

    def _tag_input_zone(self, decision, perception) -> "PolicyDecision":
        """Mark the action as is_input_zone=True if the target is a blank input surface.

        The visual servo rejects clicks on uniform (low-variance) regions — but a
        blank Notepad text area or empty input field is intentionally white/blank and
        must still accept clicks.  This method detects that case and sets the bypass
        flag so _region_has_content skips the variance gate.
        """
        from src.models.perception import UIElementType
        action = decision.action
        if action.action_type not in (ActionType.CLICK, ActionType.TYPE):
            return decision
        if action.is_input_zone:
            return decision  # already flagged

        target_id = (action.target_element_id or "").lower()
        # Check element_id substrings
        if any(tok in target_id for tok in self._INPUT_ZONE_ID_TOKENS):
            return decision.model_copy(update={"action": action.model_copy(update={"is_input_zone": True})})

        # Check element_type from current perception
        if action.target_element_id and perception is not None:
            for el in perception.visible_elements:
                if el.element_id == action.target_element_id:
                    if el.element_type is UIElementType.INPUT:
                        return decision.model_copy(update={"action": action.model_copy(update={"is_input_zone": True})})
                    break

        return decision

    def _update_coord_anchor(self, run_id: str, action: AgentAction) -> None:
        """Record the click coordinates for an element as the known-good anchor.

        Called after every successful CLICK so subsequent TYPE actions on the same
        element can snap back to this position if perception jitter shifts the
        reported coordinates by a small amount.
        """
        target_id = action.target_element_id
        if target_id is None or action.x is None or action.y is None:
            return
        self._coord_anchors.setdefault(run_id, {})[target_id] = (action.x, action.y)
        logger.debug("coord_anchor: stored %r → (%d, %d) for run %s", target_id, action.x, action.y, run_id[:8])

    def _apply_coord_anchor(
        self,
        run_id: str,
        decision,
    ) -> tuple:
        """Snap TYPE action coordinates to the post-click anchor when drift is small.

        Returns (possibly-modified decision, AnchorSnapInfo | None).  The snap only
        fires when all of the following hold:
        - Action is TYPE with a target_element_id and x/y coordinates.
        - An anchor exists for that element_id in this run.
        - Euclidean distance between fresh perception coords and the anchor is
          strictly less than _ANCHOR_SNAP_THRESHOLD_PX (small jitter, not a
          genuine element move).
        """
        action = decision.action
        if action.action_type is not ActionType.TYPE:
            return decision, None
        target_id = action.target_element_id
        if target_id is None or action.x is None or action.y is None:
            return decision, None
        anchors = self._coord_anchors.get(run_id, {})
        anchor = anchors.get(target_id)
        if anchor is None:
            return decision, None
        ax, ay = anchor
        drift = ((action.x - ax) ** 2 + (action.y - ay) ** 2) ** 0.5
        if drift >= self._ANCHOR_SNAP_THRESHOLD_PX:
            logger.debug(
                "coord_anchor: %r drift=%.1fpx ≥ threshold — trusting fresh perception",
                target_id, drift,
            )
            return decision, None
        logger.info(
            "snap_to_anchor: %r drift=%.1fpx original=(%d,%d) anchor=(%d,%d) run=%s",
            target_id, drift, action.x, action.y, ax, ay, run_id[:8],
        )
        snapped_action = action.model_copy(update={"x": ax, "y": ay})
        snap_info = AnchorSnapInfo(
            element_id=target_id,
            original_x=action.x,
            original_y=action.y,
            anchored_x=ax,
            anchored_y=ay,
            drift_px=round(drift, 1),
        )
        return decision.model_copy(update={"action": snapped_action}), snap_info

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

    def _attach_target_context(self, action: AgentAction, perception, benchmark: str | None = None) -> AgentAction:
        target = self._resolve_action_target(action, perception)
        normalized_action = self._normalize_action_target_coordinates(action, target)
        # Apply monitor origin: perception coords are monitor-local; pyautogui needs
        # virtual-desktop coords. This is the single transform point — perception
        # and policy layers stay in monitor-local space throughout.
        normalized_action = self._apply_monitor_origin(normalized_action, perception)
        if normalized_action.target_context is not None:
            return normalized_action
        intent = self._infer_target_intent(action, target, perception, benchmark)
        if target is None or intent is None:
            return normalized_action
        context = self.target_selector.build_selection_context(
            perception,
            intent,
            target,
            page_signature=self._page_signature(perception),
        )
        return normalized_action.model_copy(update={"target_context": context})

    @staticmethod
    def _apply_monitor_origin(action: AgentAction, perception) -> AgentAction:
        """Translate monitor-local coords to virtual-desktop coords by adding the
        monitor origin stored in perception. No-op when origin is (0, 0).

        Covers x/y (start point) and x_end/y_end (drag/screenshot_region end point).
        """
        origin = getattr(perception, "monitor_origin", (0, 0))
        ox, oy = origin
        if ox == 0 and oy == 0:
            return action
        updates: dict = {}
        if action.x is not None:
            updates["x"] = action.x + ox
        if action.y is not None:
            updates["y"] = action.y + oy
        if action.x_end is not None:
            updates["x_end"] = action.x_end + ox
        if action.y_end is not None:
            updates["y_end"] = action.y_end + oy
        if not updates:
            return action
        return action.model_copy(update=updates)

    @staticmethod
    def _normalize_action_target_coordinates(action: AgentAction, target: UIElement | None) -> AgentAction:
        if target is None:
            return action
        center_x = target.x + max(1, target.width // 2)
        center_y = target.y + max(1, target.height // 2)
        if action.action_type in {
            ActionType.CLICK,
            ActionType.HOVER,
            ActionType.UPLOAD_FILE_NATIVE,
        }:
            return action.model_copy(update={"x": center_x, "y": center_y})
        if action.action_type is ActionType.TYPE and (action.x is None or action.y is None):
            return action.model_copy(update={"x": center_x, "y": center_y})
        return action

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

    def _infer_target_intent(self, action: AgentAction, target: UIElement | None, perception, benchmark: str | None = None) -> TargetIntent | None:
        from src.benchmarks.registry import BENCHMARK_REGISTRY
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
            expected_section=BENCHMARK_REGISTRY.get_section(benchmark, perception.page_hint),
        )

    def _persist_execution_trace(self, run_id: str, step_index: int, executed_action):
        if executed_action.execution_trace is None:
            return executed_action
        # Step dir already created by _prepare_step_artifacts — no mkdir needed.
        step_dir = Path(self._before_artifact_path(run_id, step_index)).resolve().parent
        trace_path = step_dir / "execution_trace.json"
        bg_writer.enqueue(trace_path, executed_action.execution_trace.model_dump_json())
        return executed_action.model_copy(update={"execution_trace_artifact_path": str(trace_path)})

    def _persist_progress_trace(self, run_id: str, step_index: int, progress_trace: ProgressTrace) -> str:
        # Step dir already created by _prepare_step_artifacts — no mkdir needed.
        step_dir = Path(self._before_artifact_path(run_id, step_index)).resolve().parent
        trace_path = step_dir / "progress_trace.json"
        bg_writer.enqueue(trace_path, progress_trace.model_dump_json())
        return str(trace_path)

    def _inject_stagnation_hint(self, state) -> None:
        """If the same subgoal persists for 3+ steps without progress, inject a hint."""
        ps = state.progress_state
        if ps.no_progress_streak >= 3 and hasattr(self.perception_service, "add_advisory_hints"):
            hint = (
                f"WARNING: You have been on the same subgoal for {ps.no_progress_streak} steps "
                "without visible progress. The screen has not changed. "
                "Try a DIFFERENT action or update your subgoal. "
                "If the task is already complete, use the stop action."
            )
            self.perception_service.add_advisory_hints([hint], source="no_progress")

    def _sync_progress_state_with_perception(self, state, perception) -> None:
        page_signature = self._page_signature(perception)
        progress_state = state.progress_state
        previous_signature = progress_state.latest_page_signature
        # Track previous for page-change detection in _meaningful_progress
        progress_state.previous_page_signature = previous_signature
        progress_state.latest_page_signature = page_signature
        if previous_signature is None or previous_signature == page_signature:
            return
        # Page changed — reset action/target repeat counters (but NOT no_progress_streak,
        # which is managed by _meaningful_progress based on actual screen change detection)
        progress_state.repeated_action_count.clear()
        progress_state.repeated_target_count.clear()
        progress_state.recent_failures = []
        progress_state.loop_detected = False

    def _infer_focused_element(self, state, perception):
        if perception.focused_element_id is not None:
            return perception
        if not state.action_history or not state.observation_history:
            return perception

        last_executed = state.action_history[-1]
        if not last_executed.success:
            return perception

        action = last_executed.action
        if action.action_type not in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}:
            return perception

        current_inputs = [
            element
            for element in perception.visible_elements
            if element.element_type is UIElementType.INPUT and element.usable_for_targeting
        ]
        if not current_inputs:
            return perception

        target_id = action.target_element_id
        if target_id is not None:
            direct_match = next((element for element in current_inputs if element.element_id == target_id), None)
            if direct_match is not None:
                return perception.model_copy(update={"focused_element_id": direct_match.element_id})

        previous_perception = state.observation_history[-1]
        previous_target = None
        if target_id is not None:
            previous_target = next(
                (element for element in previous_perception.visible_elements if element.element_id == target_id),
                None,
            )

        if previous_target is None:
            return perception

        matched_inputs = [
            element
            for element in current_inputs
            if element.primary_name == previous_target.primary_name
        ]
        if len(matched_inputs) == 1:
            return perception.model_copy(update={"focused_element_id": matched_inputs[0].element_id})

        return perception

    # Minimum screen change ratio below which a prior step is classified as Stalled.
    # 1% (0.01) is tighter than the existing 0.2% progress threshold — it means
    # the screen was essentially static after the action.
    _STALL_SCREEN_CHANGE_THRESHOLD = 0.01

    def _block_redundant_action(self, state, action, step_index: int):
        progress_state = state.progress_state
        page_signature = progress_state.latest_page_signature or "unknown_page"
        action_signature = self._action_signature(action)
        target_signature = self._target_signature(action)
        subgoal_signature = self._subgoal_signature(state.current_subgoal)

        # Stalled State check: if the previous action produced < 1% screen change
        # AND two or more consecutive no-progress steps have occurred, force a
        # subgoal reset. Requiring streak >= 2 avoids false positives on the first
        # quiet step where last_screen_change_ratio is at its default 0.0.
        if (
            progress_state.last_screen_change_ratio < self._STALL_SCREEN_CHANGE_THRESHOLD
            and progress_state.no_progress_streak >= 2
            and action.action_type in {
                ActionType.CLICK, ActionType.DOUBLE_CLICK,
                ActionType.TYPE, ActionType.PRESS_KEY, ActionType.HOTKEY,
            }
        ):
            stale_subgoal = state.current_subgoal or "current step"
            logger.warning(
                "Stalled State: screen change ratio %.4f < %.2f threshold after %d no-progress steps. "
                "Forcing subgoal reset from %r.",
                progress_state.last_screen_change_ratio,
                self._STALL_SCREEN_CHANGE_THRESHOLD,
                progress_state.no_progress_streak,
                stale_subgoal,
            )
            state.current_subgoal = f"Stalled — choose a completely different approach for: {stale_subgoal}"
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS,
                detail=(
                    f"Stalled State detected: screen change ratio {progress_state.last_screen_change_ratio:.4f} "
                    f"< 1% threshold after {progress_state.no_progress_streak} no-progress step(s). "
                    "Subgoal reset forced — the agent must choose a different strategy."
                ),
            )

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
        _REPEATABLE_ACTIONS = {
            ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK,
            ActionType.PRESS_KEY, ActionType.HOTKEY, ActionType.TYPE,
            ActionType.HOVER, ActionType.SCROLL, ActionType.LAUNCH_APP,
        }
        if (
            action.action_type in _REPEATABLE_ACTIONS
            and repeated_action_count >= self.MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS
            and progress_state.no_progress_streak > 0
        ):
            return self._redundant_action_failure(
                action=action,
                category=FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                detail=f"Action blocked: repeated {action.action_type.value} without meaningful progress.",
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
        before_artifact_path: str | None = None,
        screen_change_ratio: float | None = None,
    ) -> ProgressTrace:
        progress_state = state.progress_state
        action_signature = self._action_signature(executed_action.action)
        target_signature = self._target_signature(executed_action.action)
        subgoal_signature = self._subgoal_signature(decision.active_subgoal)
        failure_signature = self._failure_signature(executed_action, verification, recovery, target_signature)

        # Subgoal-based progress signal
        previous_subgoal = progress_state.latest_subgoal_signature
        subgoal_changed = previous_subgoal is None or subgoal_signature != previous_subgoal
        progress_state.latest_subgoal_signature = subgoal_signature

        # Use pre-computed ratio if available; compute only as fallback.
        if screen_change_ratio is None:
            after_path = executed_action.artifact_path if hasattr(executed_action, "artifact_path") else None
            if before_artifact_path and after_path:
                from src.agent.screen_diff import compute_screen_change_ratio
                screen_change_ratio = compute_screen_change_ratio(before_artifact_path, after_path)

        # An action is "novel" if its signature hasn't been seen before in this run
        is_novel_action = action_signature not in progress_state.repeated_action_count

        progress_made = self._meaningful_progress(
            executed_action, verification,
            subgoal_changed=subgoal_changed,
            screen_change_ratio=screen_change_ratio,
            is_novel_action=is_novel_action,
        )

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
            self._mark_completed_progress(state, decision, executed_action, verification)
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

        # Persist the raw screen change ratio so _block_redundant_action can
        # detect a Stalled State on the *next* step without recomputing it.
        if screen_change_ratio is not None:
            progress_state.last_screen_change_ratio = screen_change_ratio

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
            screen_change_ratio=screen_change_ratio,
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
        # Period-2: abab
        if len(recent_actions) >= 4:
            a, b, c, d = recent_actions[-4:]
            if a == c and b == d and a != b:
                return True
        # Period-3: abcabc
        if len(recent_actions) >= 6:
            tail = recent_actions[-6:]
            if tail[:3] == tail[3:] and len(set(tail[:3])) > 1:
                return True
        return False

    def _mark_completed_progress(self, state, decision, executed_action, verification) -> None:
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

        if self._should_mark_subgoal_complete(progress_state, executed_action, verification):
            if subgoal_signature not in progress_state.completed_subgoals:
                progress_state.completed_subgoals.append(subgoal_signature)
            progress_state.subgoal_completion_page_signatures[subgoal_signature] = page_signature

    @staticmethod
    def _should_mark_subgoal_complete(progress_state, executed_action, verification) -> bool:
        if not executed_action.success:
            return False
        if verification.stop_condition_met:
            return True

        action = executed_action.action
        previous_page = progress_state.previous_page_signature
        current_page = progress_state.latest_page_signature
        page_changed = previous_page is not None and current_page is not None and previous_page != current_page

        if action.action_type is ActionType.NAVIGATE:
            return True
        if action.action_type in {ActionType.TYPE, ActionType.SELECT}:
            return page_changed or not bool(action.press_enter)
        if action.action_type in {
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.PRESS_KEY,
            ActionType.HOTKEY,
            ActionType.LAUNCH_APP,
        }:
            return page_changed
        return page_changed

    @staticmethod
    def _meaningful_progress(
        executed_action,
        verification,
        *,
        subgoal_changed: bool = True,
        screen_change_ratio: float | None = None,
        is_novel_action: bool = True,
    ) -> bool:
        """Progress requires visual change AND a novel action.

        Signals:
        - screen_change_ratio: did the screen actually change?
        - is_novel_action: is this a new action signature (not seen before)?
        - subgoal_changed: did the model advance its subgoal?

        Screen change + novel action = progress (typing text in Notepad).
        Screen change + repeated action = NOT progress (launch_app Notepad 5x).
        No screen change = NOT progress regardless.
        Subgoal change always counts as progress (model advancing intentionally).
        """
        if not executed_action.success:
            return False
        if verification.stop_condition_met:
            return True
        if subgoal_changed:
            return True

        if screen_change_ratio is not None:
            from src.agent.screen_diff import SCREEN_CHANGE_THRESHOLD
            screen_changed = screen_change_ratio >= SCREEN_CHANGE_THRESHOLD
            return screen_changed and is_novel_action

        # Screenshots unavailable — require novelty but don't treat as confirmed progress;
        # return False so the no-progress streak is not reset without visual evidence.
        return False

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
        top_elements = perception.visible_elements[:15]
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
            # Bucket coordinates to 50px grid so nearby clicks are detected as repeats
            bx, by = action.x // 50 * 50, action.y // 50 * 50
            return f"xy:{bx}:{by}"
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
        suffix = (
            verification.failure_category.value
            if verification.failure_category is not None
            else verification.failure_type.value
            if verification.failure_type is not None
            else verification.status.value
        )
        prefix = f"{decision.active_subgoal}:"
        matches = [count for key, count in state.retry_counts.items() if key.startswith(prefix) and key.endswith(f":{suffix}")]
        return max(matches, default=0)

    @staticmethod
    def _update_target_failure_signal(state, executed_action, verification) -> None:
        action = executed_action.action
        failure_category = verification.failure_category or executed_action.failure_category

        # Track TYPE failures per element (existing logic)
        if action.action_type is ActionType.TYPE and action.target_element_id is not None:
            if failure_category in {
                FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
            }:
                key = f"type:{action.target_element_id}:{failure_category.value}"
                state.target_failure_counts[key] = state.target_failure_counts.get(key, 0) + 1

        # Track CLICK failures per coordinate pair — feeds coordinate-thrash HITL detection.
        # Visual servo failures (blank region) and target-not-found failures both count.
        if action.action_type in {ActionType.CLICK, ActionType.DOUBLE_CLICK} and action.x is not None and action.y is not None:
            if failure_category in {
                FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                FailureCategory.EXECUTION_ERROR,
                FailureCategory.STALE_TARGET_BEFORE_ACTION,
                FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
                FailureCategory.TARGET_LOST_BEFORE_ACTION,
            }:
                coord_key = f"click:{action.x}:{action.y}"
                state.target_failure_counts[coord_key] = state.target_failure_counts.get(coord_key, 0) + 1

    _COORDINATE_THRASH_THRESHOLD = 3

    def _coordinate_thrash_detected(self, state, action) -> bool:
        """Return True when the same click coordinate has failed 3+ times this run.

        Only applies to CLICK/DOUBLE_CLICK actions with explicit (x, y) coordinates.
        Coordinate thrashing signals that the target is unreachable by automated
        re-resolution — human intervention is required to verify the target location.
        """
        if action.action_type not in {ActionType.CLICK, ActionType.DOUBLE_CLICK}:
            return False
        if action.x is None or action.y is None:
            return False
        coord_key = f"click:{action.x}:{action.y}"
        return state.target_failure_counts.get(coord_key, 0) >= self._COORDINATE_THRASH_THRESHOLD
