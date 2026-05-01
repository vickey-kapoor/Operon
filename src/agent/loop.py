"""Agent loop orchestrator for the vision-driven automation workflow."""

import asyncio
import logging
import os
import time
from pathlib import Path

from src.agent.anchor_cache import AnchorCache, tag_input_zone
from src.agent.capture import CaptureService
from src.agent.perception import PerceptionLowQualityError, PerceptionService
from src.agent.policy import PolicyService
from src.agent.progress_tracker import (
    ProgressTracker,
    action_signature,
    alternating_action_loop,
    append_window,
    apply_no_progress_detection,
    failure_signature,
    meaningful_progress,
    page_signature,
    redundant_action_failure,
    should_mark_subgoal_complete,
    stop_reason_for_failure,
    subgoal_signature,
    target_signature,
)
from src.agent.recovery import RecoveryManager, validate_benchmark_integrity
from src.agent.reflector import PostRunReflector
from src.agent.retry_hardening import (
    RetryHardening,
    apply_reresolution_failure,
    merge_execution_retry,
    refresh_action_coordinates,
    should_retry,
)
from src.agent.retry_hardening import (
    RetryResolution as _RetryResolution,
)
from src.agent.selector import DeterministicTargetSelector
from src.agent.step_artifacts import StepArtifactsManager
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
from src.models.policy import ActionType, AgentAction
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


def run_logger(run_id: str, step: int | None = None) -> logging.LoggerAdapter:
    """Return a logger that binds run_id (and optionally step) to every record.

    Lets oncall grep across many concurrent runs by run-id — without hand-threading
    the field through every call site. Use as a drop-in replacement for `logger`
    inside per-run code paths.
    """
    extra = {"run_id": run_id[:8] if run_id else "-", "step": step if step is not None else "-"}
    return logging.LoggerAdapter(logger, extra)


class AgentLoop:
    """Coordinates the MVP control loop and terminal benchmark boundaries."""

    # Loop-progress thresholds: kept here as compatibility aliases for any
    # external readers; ProgressTracker is the source of truth.
    MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS = ProgressTracker.MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS
    MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS = ProgressTracker.MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS
    MAX_NO_PROGRESS_STEPS = ProgressTracker.MAX_NO_PROGRESS_STEPS
    MAX_REPEAT_SAME_FAILURE = ProgressTracker.MAX_REPEAT_SAME_FAILURE
    RECENT_WINDOW_SIZE = 6  # used only by tests; ProgressTracker has its own constant

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
        self._retry = RetryHardening(self.target_selector)
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
        self._anchor_cache = AnchorCache()
        self._artifacts = StepArtifactsManager(run_store)
        self._progress = ProgressTracker()
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

    def _maybe_reuse_prior_perception(self, state, frame: CaptureFrame):
        """Return a reused ScreenPerception when conditions are safe, else None.

        Conservative gates:
        - The previous action was WAIT (idempotent — agent expected no change).
        - The frame's visual_velocity is exactly 0.0 (no pixel movement at all).
        - The previous verification was UNCERTAIN (not SUCCESS, not FAILURE).
        - There IS a previous perception in observation_history.
        - We did NOT already reuse on the previous step (no back-to-back skips —
          forces a fresh look at least every other step to catch slow renders).
        """
        if frame.visual_velocity != 0.0:
            return None
        if not state.observation_history or not state.action_history:
            return None
        # Avoid back-to-back reuse: track per-run via a small set on self.
        run_id = state.run_id
        prev_skipped = getattr(self, "_perception_skipped_runs", set())
        if run_id in prev_skipped:
            prev_skipped.discard(run_id)
            self._perception_skipped_runs = prev_skipped
            return None
        last_action = state.action_history[-1].action
        if last_action.action_type is not ActionType.WAIT:
            return None
        if state.verification_history:
            from src.models.verification import VerificationStatus as _VS
            if state.verification_history[-1].status is not _VS.UNCERTAIN:
                return None
        prior = state.observation_history[-1]
        # Mark this run as having just skipped — next step must re-perceive.
        prev_skipped.add(run_id)
        self._perception_skipped_runs = prev_skipped
        # Update the artifact path so step records reflect the current frame.
        return prior.model_copy(update={"capture_artifact_path": frame.artifact_path})

    async def _consume_force_fresh_perception(self, record) -> None:
        """When force_fresh_perception is set, wait for the UI to settle THEN
        reset the flag. The order matters: an early return between the reset
        and the wait would silently drop the settle delay.

        Browser DOM hydration commonly takes longer than desktop, so use 1.0s
        there vs 0.5s on desktop.
        """
        if not record.force_fresh_perception:
            return
        settle_delay = 1.0 if self.environment is UnifiedEnvironment.BROWSER else 0.5
        await asyncio.sleep(settle_delay)
        record.force_fresh_perception = False

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

        # Per-stage timing — written to step_timing.json at end-of-step for
        # tail-latency analysis. Values are accumulated in milliseconds.
        stage_timings: dict[str, float] = {}

        def _record_stage(name: str, t0: float) -> float:
            elapsed = time.perf_counter() - t0
            stage_timings[name] = stage_timings.get(name, 0.0) + elapsed * 1000.0
            return elapsed

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

        await self._consume_force_fresh_perception(record)

        _trace("  1 CAPTURE", "taking screenshot via CaptureService")
        _t0 = time.perf_counter()
        frame = await self.capture_service.capture(record)
        _trace("  1 CAPTURE OK", f"{_record_stage('capture', _t0):.2f}s  frame={type(frame).__name__}")

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
        _trace("  2 PERCEIVE OK", f"{_record_stage('perceive', _t0):.2f}s  page_hint={perception.page_hint.value!r}  elements={len(perception.visible_elements)}")
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
        _trace("  3 POLICY OK", f"{_record_stage('policy', _t0):.2f}s  action={decision.action.action_type.value!r}  target={decision.action.target_element_id!r}  rationale={decision.rationale[:80]!r}")
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
                _trace("  4 EXECUTE OK" if executed_action.success else "  4 EXECUTE FAIL", f"{_record_stage('execute', _t0):.2f}s  success={executed_action.success}  detail={executed_action.detail!r}")
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
            _t0 = time.perf_counter()
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
            _trace("  5 VERIFY OK", f"{_record_stage('verify', _t0):.2f}s  status={verification.status.value!r}  stop={verification.stop_condition_met}  stop_reason={verification.stop_reason!r}")
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

            # Native retry decision (no longer routed through the unified contract).
            # The unified contract conversion below is now observability-only — kept
            # so benchmark_runner can read retry_count/last_failure_type via
            # unified_state_for_run(). Retry strategy comes from a direct
            # FailureCategory→strategy lookup.
            from src.agent.adaptation import strategy_for_failure
            recent_failure = verification.failure_category or executed_action.failure_category
            strategy = strategy_for_failure(
                recent_failure,
                verification_failure=verification.status is VerificationStatus.FAILURE,
                verification_uncertain=verification.status is VerificationStatus.UNCERTAIN,
            )

            # Best-effort unified contract conversion for observability/benchmarks.
            unified_step = None
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
                _trace("  6 OBSERVABILITY OK", f"failure_category={recent_failure!r}  strategy={strategy!r}")
            except (RoutingError, ValueError) as _exc:
                # Conversion failure must not affect the retry decision — that's
                # decoupled now. Just log and continue.
                _trace("  6 OBSERVABILITY skipped", f"{type(_exc).__name__}: {_exc}")

            # In-step retry only when (a) verification produced a hard FAILURE —
            # not merely UNCERTAIN, which the outer recovery stage handles — and
            # (b) the legacy hardening path did not already retry this action.
            legacy_retry_attempted = bool(
                executed_action.execution_trace is not None
                and executed_action.execution_trace.retry_attempted
            )
            allow_adaptation_retry = (
                verification.status is VerificationStatus.FAILURE
                and not legacy_retry_attempted
            )
            if strategy is None or not allow_adaptation_retry:
                if retries_used > 0 and unified_step is not None:
                    unified_state = unified_state.model_copy(
                        update={
                            "retry_count": retries_used,
                            "last_strategy": adaptation_trace[-1],
                            "last_failure_type": None
                            if verification.status is VerificationStatus.SUCCESS
                            else unified_step.critic.failure_type,
                        }
                    )
                    self.unified_states[record.run_id] = unified_state
                _trace("  RETRY_LOOP -> break", f"strategy={strategy!r}  retries_used={retries_used}  allow_retry={allow_adaptation_retry}")
                break
            if retries_used >= self.unified_orchestrator.max_retries:
                _trace("  RETRY_LOOP -> max_retries", f"retries_used={retries_used}")
                break

            retries_used += 1
            adaptation_trace.append(strategy)
            _trace("  RETRY_LOOP -> retry", f"retries_used={retries_used}  strategy={strategy!r}")
            if unified_step is not None:
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
        _t0 = time.perf_counter()
        recovery: RecoveryDecision = await self.recovery_manager.recover(
            state,
            decision,
            executed_action,
            verification,
        )
        # Benchmark Integrity Check: reject recovery decisions that bypass verification
        # or claim success without visual confirmation.
        recovery = validate_benchmark_integrity(recovery, verification)
        _trace("  7 RECOVER OK", f"{_record_stage('recover', _t0):.2f}s  strategy={recovery.strategy.value!r}  stop_reason={recovery.stop_reason!r}")
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
            self._anchor_cache.discard_run(record.run_id)
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

        # Persist per-stage durations for tail-latency analysis. Best-effort —
        # the step has already completed by this point.
        self._persist_step_timing(record.run_id, step_index, stage_timings)

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
        # Stable-frame perception bypass: when the previous action was WAIT (idempotent),
        # the screen hasn't moved at all (visual_velocity exactly 0), and verification
        # was UNCERTAIN (not terminal), re-perceiving an unchanged frame returns the
        # same answer modulo Gemini noise. Reuse the prior perception object with the
        # current capture path.
        cached = self._maybe_reuse_prior_perception(state, initial_frame)
        if cached is not None:
            logger.info(
                "perception_skip: reusing prior perception (last_action=WAIT, velocity=0.0, status=UNCERTAIN)",
            )
            return cached

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
        self._artifacts.prepare(run_id, step_index, before_artifact_path, after_artifact_path)

    def _before_artifact_path(self, run_id: str, step_index: int) -> str:
        return self._artifacts.before_path(run_id, step_index)

    def _after_artifact_path(self, run_id: str, step_index: int) -> str:
        return self._artifacts.after_path(run_id, step_index)

    def _run_log_path(self, run_id: str) -> str:
        return self._artifacts.run_log_path(run_id)

    def _relocate_after_artifact(self, executed_action, planned_path: str):
        return self._artifacts.relocate_after_artifact(executed_action, planned_path)

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
        return should_retry(executed_action)

    def _resolve_retry_action(self, *, action, perception, retry_reason: FailureCategory | None) -> _RetryResolution:
        return self._retry.resolve_retry_action(action=action, perception=perception, retry_reason=retry_reason)

    # ------------------------------------------------------------------
    # Coordinate anchor cache (delegated to src.agent.anchor_cache)
    # ------------------------------------------------------------------

    def _tag_input_zone(self, decision, perception):
        return tag_input_zone(decision, perception)

    def _update_coord_anchor(self, run_id: str, action: AgentAction) -> None:
        self._anchor_cache.update(run_id, action)

    def _apply_coord_anchor(self, run_id: str, decision) -> tuple:
        return self._anchor_cache.apply(run_id, decision)

    @staticmethod
    def _refresh_action_coordinates(action, perception):
        return refresh_action_coordinates(action, perception)

    @staticmethod
    def _merge_execution_retry(*, original, retried, retry_reason, target_reresolved, reresolution_trace):
        return merge_execution_retry(
            original=original,
            retried=retried,
            retry_reason=retry_reason,
            target_reresolved=target_reresolved,
            reresolution_trace=reresolution_trace,
        )

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
        # Respect planner-supplied coordinates: only fall back to the bbox center
        # when x/y are missing. Overwriting valid coords forces every click to the
        # geometric center and amplifies perception jitter.
        if action.x is not None and action.y is not None:
            return action
        if action.action_type in {
            ActionType.CLICK,
            ActionType.HOVER,
            ActionType.UPLOAD_FILE_NATIVE,
            ActionType.TYPE,
        }:
            center_x = target.x + max(1, target.width // 2)
            center_y = target.y + max(1, target.height // 2)
            return action.model_copy(update={"x": center_x, "y": center_y})
        return action

    def _intent_reresolve_action(
        self,
        action: AgentAction,
        perception,
        retry_reason: FailureCategory,
    ) -> _RetryResolution:
        return self._retry.intent_reresolve_action(action, perception, retry_reason)

    @staticmethod
    def _apply_reresolution_failure(original, reresolution_trace: ExecutionReresolutionTrace | None):
        return apply_reresolution_failure(original, reresolution_trace)

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
        return self._artifacts.persist_execution_trace(run_id, step_index, executed_action)

    def _persist_progress_trace(self, run_id: str, step_index: int, progress_trace: ProgressTrace) -> str:
        return self._artifacts.persist_progress_trace(run_id, step_index, progress_trace)

    def _persist_step_timing(self, run_id: str, step_index: int, stage_timings: dict[str, float]) -> None:
        self._artifacts.persist_step_timing(run_id, step_index, stage_timings)

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
        self._progress.sync_with_perception(state, perception)

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

    # ------------------------------------------------------------------
    # Progress-tracking and signatures (delegated to src.agent.progress_tracker)
    # ------------------------------------------------------------------

    def _block_redundant_action(self, state, action, step_index: int):
        return self._progress.block_redundant_action(state, action, step_index, logger)

    @staticmethod
    def _redundant_action_failure(*, action, category: FailureCategory, detail: str):
        return redundant_action_failure(action=action, category=category, detail=detail)

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
        return self._progress.update_progress_state(
            state=state,
            decision=decision,
            executed_action=executed_action,
            verification=verification,
            recovery=recovery,
            step_index=step_index,
            before_artifact_path=before_artifact_path,
            screen_change_ratio=screen_change_ratio,
        )

    def _apply_progress_stop_guard(self, recovery: RecoveryDecision, progress_trace: ProgressTrace) -> RecoveryDecision:
        return self._progress.apply_progress_stop_guard(recovery, progress_trace)

    def _detect_loop_failure(
        self,
        *,
        progress_state,
        action_signature: str,
        target_signature: str | None,
        failure_signature: str | None,
    ) -> tuple[FailureCategory | None, str | None]:
        return self._progress.detect_loop_failure(
            progress_state=progress_state,
            action_signature=action_signature,
            target_signature=target_signature,
            failure_signature=failure_signature,
        )

    @staticmethod
    def _alternating_action_loop(recent_actions: list[str]) -> bool:
        return alternating_action_loop(recent_actions)

    def _mark_completed_progress(self, state, decision, executed_action, verification) -> None:
        self._progress._mark_completed_progress(state, decision, executed_action, verification)

    @staticmethod
    def _should_mark_subgoal_complete(progress_state, executed_action, verification) -> bool:
        return should_mark_subgoal_complete(progress_state, executed_action, verification)

    @staticmethod
    def _meaningful_progress(
        executed_action,
        verification,
        *,
        subgoal_changed: bool = True,
        screen_change_ratio: float | None = None,
        is_novel_action: bool = True,
    ) -> bool:
        return meaningful_progress(
            executed_action,
            verification,
            subgoal_changed=subgoal_changed,
            screen_change_ratio=screen_change_ratio,
            is_novel_action=is_novel_action,
        )

    @classmethod
    def _failure_signature(cls, executed_action, verification, recovery, target_sig: str | None) -> str | None:
        return failure_signature(executed_action, verification, recovery, target_sig)

    @staticmethod
    def _stop_reason_for_failure(category: FailureCategory | None) -> StopReason | None:
        return stop_reason_for_failure(category)

    @classmethod
    def _page_signature(cls, perception) -> str:
        return page_signature(perception)

    @staticmethod
    def _append_window(entries: list[str], value: str) -> list[str]:
        return append_window(entries, value)

    @classmethod
    def _action_signature(cls, action: AgentAction) -> str:
        return action_signature(action)

    @staticmethod
    def _target_signature(action: AgentAction) -> str | None:
        return target_signature(action)

    @classmethod
    def _subgoal_signature(cls, subgoal: str | None) -> str:
        return subgoal_signature(subgoal)

    @staticmethod
    def _apply_no_progress_detection(state, executed_action):
        return apply_no_progress_detection(state, executed_action)

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
