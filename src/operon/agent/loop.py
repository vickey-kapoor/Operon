"""Agent loop orchestrator for the vision-driven automation workflow."""

import asyncio
import inspect
import json
import logging
import time
import weakref
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

from operon.agent import control as run_control
from operon.agent.actions.grounding import element_center
from operon.agent.actions.selector import DeterministicTargetSelector
from operon.agent.artifacts.step_artifacts import StepArtifactsManager
from operon.agent.backends.base import ModelBlockedError
from operon.agent.efficiency import run_efficiency
from operon.agent.failure_cause import (
    append_run_end,
    cause_of_error,
    clear_failure,
    error_text,
    note_failure,
)
from operon.agent.perception import PerceptionLowQualityError, PerceptionService
from operon.agent.perception.capture import CaptureService
from operon.agent.perception.expect_check import (
    EXPECT_INSTRUCTION,
    EXPECT_REMINDER_EVERY,
    NO_SCREEN_EFFECT_ACTIONS,
    compare,
    mismatch_note,
    parse_expectation,
)
from operon.agent.planner import (
    PLANNER_DEADLINE_S,
    PLANNER_USAGE_FILE,
    make_plan,
    plan_hint,
)
from operon.agent.policy import PolicyService
from operon.agent.policy.click_text import make_click_reader, read_click_text
from operon.agent.policy.file_guard import FILE_PATH_ACTIONS, blocked_file_path
from operon.agent.policy.progress_tracker import (
    ProgressTracker,
    apply_no_progress_detection,
    page_signature,
)
from operon.agent.policy.recovery import RecoveryManager, validate_benchmark_integrity
from operon.agent.policy.trust_gate import (
    PermissionMode,
    TrustVerdict,
    apply_permission_mode,
    make_trust_gate,
    names_no_target,
    resolve_permission_mode,
)
from operon.agent.policy.verifier import VerifierService
from operon.agent.resume_hint import RESUME_HINT_SOURCE, build_resume_hint
from operon.agent.step_context import StepContext
from operon.agent.task_note import build_task_note
from operon.clients.gemini import GeminiClient
from operon.core.contracts.perception import Environment as UnifiedEnvironment
from operon.core.logging_utils import url_origin_for_log
from operon.core.settings import get_settings
from operon.core.vault import action_uses_secret, available_names, secrets_hint
from operon.executor.browser import Executor
from operon.executor.browser_adapter import BrowserExecutor as UnifiedBrowserExecutor
from operon.executor.desktop_adapter import DesktopExecutor as UnifiedDesktopExecutor
from operon.models.capture import CaptureFrame
from operon.models.common import (
    FailureCategory,
    LoopStage,
    RunFailureCause,
    RunResponse,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from operon.models.execution import (
    ExecutedAction,
)
from operon.models.logs import (
    FailureRecord,
    ModelDebugArtifacts,
    PreStepFailureLog,
    StepLog,
)
from operon.models.perception import ScreenPerception, UIElement, UIElementType
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.progress import ProgressTrace
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.selector import TargetIntent, TargetIntentAction
from operon.models.state import AgentState
from operon.models.verification import (
    VerificationStatus,
)
from operon.store.approval_audit import append_event, request_event
from operon.store.background_writer import bg_writer
from operon.store.memory import MemoryStore
from operon.store.run_logger import append_step_log, append_step_log_critical
from operon.store.run_store import RunStore

logger = logging.getLogger(__name__)


_LIVENESS_RETRY_MAX = 3
_LIVENESS_RETRY_FIRST_SLEEP_S = 0.5   # fast recovery for brief UI transitions (e.g. search overlay animating in)
_LIVENESS_RETRY_SLEEP_S = 1.5         # subsequent retries give slower loads more time

# KAN-158: exception types that mean a bug in Operon, not a flaky screen, model or
# executor. With OPERON_STRICT_ERRORS on (the offline test suite turns it on), a
# recovery site re-raises them instead of carrying on, so a bug fails a test
# rather than looking like an ordinary recovery.
_BUG_EXCEPTIONS: tuple[type[BaseException], ...] = (AssertionError, AttributeError, ImportError, NameError)


def _strict_errors() -> bool:
    return get_settings().strict_errors


def _trace(stage: str, detail: str = "") -> None:
    if not get_settings().trace:
        return
    suffix = f"  {detail}" if detail else ""
    msg = f"[TRACE] {stage}{suffix}"
    try:
        print(f"\033[36m{msg}\033[0m", flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _leaf_actions(action: AgentAction) -> list[AgentAction]:
    """The actions a batch actually runs, nested batches flattened; a plain action is its own leaf."""
    if action.action_type is not ActionType.BATCH:
        return [action]
    return [leaf for child in action.actions or [] for leaf in _leaf_actions(child)]


def _describe_for_approval(action: AgentAction) -> str:
    """One line a person can approve: the action type, its target and what it types."""
    parts = [action.action_type.value]
    ctx = action.target_context
    name = getattr(getattr(ctx, "original_target", None), "primary_name", None) if ctx is not None else None
    if name:
        parts.append(f"'{name}'")
    if action.url:
        parts.append(action.url)
    if action.text:
        parts.append(f"text '{action.text[:80]}'")
    if action.key:
        parts.append(f"key {action.key}")
    if action.x is not None and action.y is not None and not name:
        parts.append(f"at ({action.x}, {action.y})")
    return " ".join(parts)


class StartUrlBlockedError(ValueError):
    """``start_url`` is outside ``OPERON_TRUST_ALLOW_DOMAINS``; the run is not created."""


class TakeoverActiveError(ValueError):
    """The run is under the person's control: only a hand-back resumes it (KAN-172)."""


class _DialogPause(NamedTuple):
    """A run paused on a page dialog the executor cancelled (KAN-172)."""

    kind: str
    message: str
    held: bool  # the backend holds the calls that raised it
    child_index: int | None  # top-level batch child that raised it
    action_count: int  # actions in the turn (1 for a single action)


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
        planner_client: Any = None,
        environment: UnifiedEnvironment = UnifiedEnvironment.BROWSER,
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
        self.planner_client = planner_client
        self.target_selector = DeterministicTargetSelector()
        self.environment = environment
        self._artifacts = StepArtifactsManager(run_store)
        self._progress = ProgressTracker()
        # RFC 0001 Move 5: deny/confirm policy gate before execution (OPERON_TRUST_GATE).
        self._trust_gate = make_trust_gate()
        # KAN-163: read the text under a click from the screenshot (OPERON_CLICK_TEXT, off).
        self._click_reader = make_click_reader()
        # One lock per run so /step, /resume and the auto-run task never interleave
        # steps on the same run. Weak values: a lock lives only while a step holds
        # or awaits it, so the map does not grow with every run ever started.
        self._run_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._executor_lock = asyncio.Lock()
        # Whether a human approved the trust gate for the action about to run.
        self._trust_approved: dict[str, bool] = {}
        # Human approval of a gate pause, per run. A pause records (action fingerprint,
        # gate) as pending; a resume that approves adds that gate to the action's
        # approved set. The same action then passes the gates it was approved for, and
        # the approval is used up once it gets through (or the policy picks another).
        # run_id -> (action fingerprint, gate, risk category, trust verdict "deny"/"confirm" or "")
        self._pending_gate_approvals: dict[str, tuple[str, str, str, str]] = {}
        self._pending_audit: dict[str, dict[str, str | int]] = {}
        # Host a person approved a secret type on, per run (None: none known; KAN-103).
        self._secret_site_approved: dict[str, str | None] = {}
        # Runs this process created: they run where they started, so resume skips the provider check.
        self._created_runs: set[str] = set()
        # Runs a human paused by hand (UI Pause, POST /run/{id}/pause). A step already
        # in flight finishes its action but ends WAITING_FOR_USER instead of RUNNING.
        self._pause_requests: set[str] = set()
        self._approved_gate_actions: dict[str, tuple[str, frozenset[str]]] = {}
        # KAN-172: run_id -> (dialog type, message, whether the backend holds the action that
        # raised it) for a run paused on a page dialog the executor cancelled.
        self._dialog_pauses: dict[str, _DialogPause] = {}
        # The (type, message) of a dialog the person approved, until its replay has run
        # (kept across a gate pause on the replay, which re-arms the executor on approval).
        self._dialog_approved: dict[str, tuple[str, str]] = {}
        # KAN-158: errors the loop recovered from, per run and site, not yet saved
        # into the run's state (``AgentState.recovered_errors``). Drained by
        # ``_flush_recovered_errors`` when a step, start or resume ends.
        self._recovered_errors: dict[str, dict[str, int]] = {}
        # KAN-116: per run, the last step that got the "write an Expect line" reminder.
        self._expect_reminded: dict[str, int] = {}
        if environment is UnifiedEnvironment.BROWSER:
            self.executor_adapter = UnifiedBrowserExecutor(executor)
        else:
            self.executor_adapter = UnifiedDesktopExecutor(executor)

    async def start_run(self, request: RunTaskRequest) -> RunResponse:
        _trace("START_RUN", f"intent={request.intent!r}  origin={url_origin_for_log(request.start_url)!r}")
        if request.start_url:
            # The start URL is opened before any step, so the trust gate's per-action
            # check never sees it: judge it here, before the run exists.
            verdict = self._trust_gate.evaluate(
                AgentAction(action_type=ActionType.NAVIGATE, url=request.start_url)
            )
            if verdict.verdict is TrustVerdict.DENY:
                raise StartUrlBlockedError(
                    f"start_url blocked by policy ({verdict.reason}: '{verdict.matched}')"
                )
        # Minimize all windows (Win+D) only in desktop mode. Browser-mode runs
        # use Playwright in its own window — issuing Win+D would minimize the
        # USER's foreground app on every task start, which yanks focus during
        # multi-task benchmark sweeps (especially headless runs where the
        # browser is invisible and there's nothing to "clear" anyway).
        if (
            self.environment is not UnifiedEnvironment.BROWSER
            and hasattr(self.executor, "reset_desktop")
            and not self._test_safe_mode_enabled()
        ):
            await self.executor.reset_desktop()
        record = self.run_store.create_run(
            intent=request.intent,
            start_url=request.start_url,
            headless=request.headless,
            benchmark=request.benchmark,
            mode=request.mode,
            max_steps=request.max_steps,
        )
        self._created_runs.add(record.run_id)
        record_profile = getattr(self.run_store, "record_harness_profile", None)
        if record_profile is not None:
            record_profile(record.run_id, "thin")
        if request.permission_mode is not None:
            # Kept on the run so every step (and its report) uses the mode it started with.
            record.permission_mode = request.permission_mode
            await self.run_store.save_state(record)
        _trace("START_RUN", f"run_id={record.run_id}")
        # Selecting this run on the executor must wait for any other run's step in
        # flight, or that step's next capture or batch child would land in this run's
        # browser (see step_run).
        async with self._executor_lock:
            if hasattr(self.executor, "set_current_run_id"):
                self.executor.set_current_run_id(record.run_id)
            if hasattr(self.executor, "configure_run"):
                self.executor.configure_run(
                    record.run_id,
                    headless=request.headless,
                    mode=record.mode,
                )
            if hasattr(self.executor, "start_run_recording"):
                try:
                    await self.executor.start_run_recording(record.run_id, root_dir=self.run_store.root_dir)
                except Exception as exc:
                    self._recovered(record.run_id, "start_run_recording", exc)
            if request.start_url:
                _trace("START_RUN -> NAVIGATE", url_origin_for_log(request.start_url) or "")
                await self.executor.execute(
                    AgentAction(action_type=ActionType.NAVIGATE, url=request.start_url)
                )
        await self._write_plan(record)
        self._send_expect_instruction(record.run_id)
        await self._flush_recovered_errors(record.run_id)
        return RunResponse(
            run_id=record.run_id,
            status=record.status,
            intent=record.intent,
            step_count=record.step_count,
        )

    async def _write_plan(self, record) -> None:
        """One-shot planner (OPERON_PLANNER): a cheaper model plans once; the plan is a first-step hint.

        Any failure means no plan, never a failed run. The plan and its cost are saved as
        ``plan.json`` in the run folder so a benchmark can count them.
        """
        settings = get_settings()
        if not settings.planner:
            return
        try:
            client = self.planner_client
            if client is None:
                from operon.clients.gemini import GeminiHttpClient

                client = GeminiHttpClient(model=settings.planner_model, timeout_seconds=8.0, max_retries=0)
            # The run's start waits on this (the remote CLI gives start 30 s), so planning
            # gets a short overall deadline and then the run goes on without a plan.
            try:
                steps, usage = await asyncio.wait_for(make_plan(client, record.intent), timeout=PLANNER_DEADLINE_S)
            except TimeoutError:
                logger.warning("planner: no answer in %.0fs, running without a plan", PLANNER_DEADLINE_S)
                return
            run_dir = Path(self.run_store.root_dir) / record.run_id
            if usage is not None:
                # Where run-cost accounting looks (evals run_usage, observer summaries).
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / PLANNER_USAGE_FILE).write_text(usage.model_dump_json(indent=2), encoding="utf-8")
            if not steps:
                return
            add_hints = getattr(self.perception_service, "add_advisory_hints", None)
            if callable(add_hints):
                add_hints([plan_hint(steps)], source="planner", run_id=record.run_id)
            saved = {
                "model": settings.planner_model,
                "steps": steps,
                "usage": usage.model_dump(mode="json") if usage is not None else None,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "plan.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - a planner problem must never fail a run
            self._recovered(record.run_id, "planner", exc)

    @staticmethod
    def _test_safe_mode_enabled() -> bool:
        return get_settings().test_safe_mode

    async def _run_cancelled(self, run_id: str) -> bool:
        """Whether the run was stopped (e.g. ``/desktop/stop``) since its step began."""
        current = await self.run_store.get_run(run_id)
        return current is not None and current.status is RunStatus.CANCELLED

    @staticmethod
    def _cancelled_response(record) -> RunResponse:
        return RunResponse(
            run_id=record.run_id,
            status=RunStatus.CANCELLED,
            intent=record.intent,
            step_count=record.step_count,
        )

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
            from operon.models.verification import VerificationStatus as _VS
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
        # With OPERON_SETTLE_DETECT on, the executor waits for the screen to stop
        # changing instead; off, or an executor without it, this is the same sleep.
        settle_now = getattr(self.executor, "settle_now", None)
        if inspect.iscoroutinefunction(settle_now):
            await settle_now(settle_delay)
        else:
            await asyncio.sleep(settle_delay)
        record.force_fresh_perception = False

    async def pause_run(self, run_id: str) -> AgentState:
        """Pause a running run for a human. Takes effect now if no step is running;
        a step in flight runs its action, then ends paused instead of RUNNING."""
        record = await self.run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"Run {run_id!r} not found")
        if record.status in {RunStatus.PENDING, RunStatus.RUNNING}:
            self._pause_requests.add(run_id)
            record = await self.run_store.set_status(run_id, RunStatus.WAITING_FOR_USER)
        return record

    def pending_gate(self, run_id: str) -> str | None:
        """The gate a paused run waits on for approval (``trust``, ``confidence``,
        ``coordinate_thrash``), or None when its pause is not an approvable gate."""
        pending = self._pending_gate_approvals.get(run_id)
        if pending is None:
            # A page dialog the executor cancelled (KAN-172) is also an approval the person
            # answers; auto-confirm must never answer it.
            return "dialog" if run_id in self._dialog_pauses else None
        return pending[1]

    def pending_gate_category(self, run_id: str) -> str | None:
        """The named risk category (KAN-103) of the trust pause a run waits on, if any."""
        pending = self._pending_gate_approvals.get(run_id)
        return (pending[2] or None) if pending is not None and pending[1] == "trust" else None

    def pending_gate_verdict(self, run_id: str) -> str | None:
        """``deny`` or ``confirm`` for the trust pause a run waits on, else None.
        A deny is a hard policy block: an unattended caller must not approve it."""
        pending = self._pending_gate_approvals.get(run_id)
        if pending is None or pending[1] != "trust":
            return None
        return pending[3] or None

    async def take_over(self, run_id: str) -> AgentState:
        """Hand the screen to the person: pause the run and mark it under their control.

        No step runs while the person is in control, so nothing is captured or sent to
        the model; a step already in flight finishes its action first (see pause_run).
        Raises ValueError when the run is missing or already finished."""
        record = await self.pause_run(run_id)
        if record.status is not RunStatus.WAITING_FOR_USER:
            raise ValueError(f"Run {run_id!r} is {record.status.value}; nothing to take over")
        # The pause stops any new step; wait for one already in flight to end, so the
        # agent never captures, decides or acts on a screen the person is changing.
        lock = self._run_locks.get(run_id)
        if lock is None:
            lock = self._run_locks[run_id] = asyncio.Lock()
        async with lock:
            record = await self.run_store.get_run(run_id)
        if record is None or record.status is not RunStatus.WAITING_FOR_USER:
            state = record.status.value if record is not None else "gone"
            raise ValueError(f"Run {run_id!r} is {state}; nothing to take over")
        if not getattr(record, "in_takeover", False):
            record.in_takeover = True
            record.takeover_count = getattr(record, "takeover_count", 0) + 1
            await self.run_store.save_state(record)
        self._set_person_driving(run_id, True)
        return record

    def _set_person_driving(self, run_id: str, driving: bool) -> None:
        """Tell the executor the person has the screen (or gave it back): their dialogs are theirs to answer."""
        set_driving = getattr(self.executor, "set_person_driving", None)
        if callable(set_driving):
            set_driving(run_id, driving)

    async def hand_back(self, run_id: str, note: str | None = None) -> RunResponse:
        """End a takeover and let the agent continue from the screen as the person left it.

        Resuming is not an approval: an action a gate paused on before the takeover
        stays unapproved, and the model is told the screen may have changed."""
        record = await self.run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"Run {run_id!r} not found")
        if not getattr(record, "in_takeover", False):
            raise ValueError(f"Run {run_id!r} is not under the user's control")
        self.check_resume_environment(record)  # before the hint and in_takeover change
        hint = (
            "The user took control of the screen and has now handed it back. The screen "
            "may have changed: look at it again before acting."
        )
        if note and note.strip():
            hint += f" The user says: {note.strip()}"
        run_control.set_override_hint(run_id, hint)
        record.in_takeover = False
        self._set_person_driving(run_id, False)
        record.force_fresh_perception = True
        return await self.resume_run(run_id, approve_gated_action=False)

    async def resume_run(self, run_id: str, *, approve_gated_action: bool = True) -> RunResponse:
        """Resume a run that was paused waiting for user input.

        ``approve_gated_action``: when the pause came from the confidence or trust
        gate or the coordinate-thrash guard, resuming is the human's approval of that
        action, so the next step may run it once. Pass False when resuming with a
        correction instead (override).
        """
        record = await self.run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"Run {run_id!r} not found")
        if record.status is not RunStatus.WAITING_FOR_USER:
            raise ValueError(f"Run {run_id!r} is {record.status.value}, not waiting_for_user")
        if getattr(record, "in_takeover", False):
            # A resume would put the agent back in charge with the person's dialog rule (they
            # answer their own dialogs) still on; only hand_back ends a takeover.
            raise TakeoverActiveError(f"Run {run_id!r} is under the user's control; hand it back instead")
        self._set_person_driving(run_id, False)
        self.check_resume_environment(record)
        if record.stop_reason is StopReason.INTERRUPTED:
            await self._prepare_interrupted(run_id, record)  # a restart lost the conversation (KAN-80)
        dialog_pause = self._dialog_pauses.pop(run_id, None)
        approved_dialog = self._dialog_approved.pop(run_id, None)
        self._disarm_dialog_accept(run_id)  # nothing stays armed across a resume
        if dialog_pause is not None:
            self._answer_dialog_pause(run_id, dialog_pause, approved=approve_gated_action)
        elif approved_dialog is not None and approve_gated_action and run_id in self._pending_gate_approvals:
            # The replay of an approved dialog was gate-paused and the gate is approved now.
            self._approve_dialog(run_id, approved_dialog)
        elif not approve_gated_action:
            # A correction is not an approval: a backend holding the calls it paused
            # on (Gemini's require_confirmation) must not run them on the next step.
            self._discard_held_actions(
                run_id, "Not executed: the user declined this action and sent a correction."
            )
        if (
            approve_gated_action
            and run_id not in self._pending_gate_approvals
            and self._has_held_actions(run_id)
            and hasattr(record, "approval_count")
        ):
            # The model asked to confirm its action (Gemini's require_confirmation)
            # and the person said yes: that is an approval like a gate's. A gate
            # pause also holds the calls; its approval is counted below, once.
            record.approval_count += 1
        pending = self._pending_gate_approvals.pop(run_id, None)
        if approve_gated_action and pending is not None:
            fingerprint, gate, _category, _verdict = pending
            prior = self._approved_gate_actions.get(run_id)
            gates = prior[1] if prior is not None and prior[0] == fingerprint else frozenset()
            self._approved_gate_actions[run_id] = (fingerprint, gates | {gate})
            if hasattr(record, "approval_count"):
                record.approval_count += 1
        self._pause_requests.discard(run_id)
        record.stop_reason = None
        clear_failure(record)
        if hasattr(record, "pause_reason"):
            record.pause_reason = None
        updated = await self.run_store.set_status(run_id, RunStatus.RUNNING)
        if updated.status is RunStatus.RUNNING:
            self._resolve_approval_audit(run_id, "approved" if approve_gated_action else "declined")
        if hasattr(self.executor, "restore_work_window"):
            # The person answered through the API client, which is now in front of the app
            # the run was using: put that app back before the next screenshot.
            try:
                # Under the executor lock: another run's step must not have its window
                # switched while it waits on the model or acts.
                async with self._executor_lock:
                    await self.executor.restore_work_window(run_id)
            except Exception as exc:
                self._recovered(run_id, "restore_work_window", exc)
        await self._flush_recovered_errors(run_id)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    def _approve_dialog(self, run_id: str, dialog: tuple[str, str]) -> None:
        """Remember the dialog the person approved. The executor is armed for it only right
        before the replay runs (``_set_dialog_policy``), never earlier."""
        if callable(getattr(self.executor, "arm_accept_next_dialog", None)):
            self._dialog_approved[run_id] = dialog

    def _answer_dialog_pause(self, run_id: str, pause: _DialogPause, *, approved: bool) -> None:
        """The person's answer to a page dialog the executor cancelled (KAN-172).

        Yes: the executor accepts the next dialog of the run, once, but only this one (same
        type and message), and the action that raised it runs again: the backend replays the
        calls it holds from the one that raised the dialog on (the same path a gate approval
        takes); without held calls the model is told to repeat the action. No: the held
        calls are dropped and the model is told the person declined."""
        kind, message, held, child_index, action_count = pause
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if approved:
            self._approve_dialog(run_id, (kind, message))
            if held and child_index is not None:
                # Calls before the dialog-raising one already ran: replay from it on.
                if not self._drop_ran_held_calls(run_id, child_index, action_count):
                    held = False
                    self._discard_held_actions(
                        run_id,
                        "Not executed again: part of the turn already ran, so the model repeats the action"
                        + self._ran_before_text(child_index),
                    )
            if not held and callable(add_hints):
                add_hints(
                    [f"The person approved the page's {kind} dialog: {message!r}. "
                     "Repeat the action that opened it (only that action, not the ones before it); "
                     "the dialog will be accepted."],
                    source="dialog_answer",
                    run_id=run_id,
                )
            return
        self._discard_held_actions(
            run_id,
            f"Not executed: the person declined the page's {kind} dialog and it was cancelled"
            + self._ran_before_text(child_index),
        )
        if callable(add_hints):
            add_hints(
                [f"The person declined the page's {kind} dialog {message!r}; it was cancelled. "
                 "Do not repeat that action unless the person asks."],
                source="dialog_answer",
                run_id=run_id,
            )

    @staticmethod
    def _ran_before_text(child_index: int | None) -> str:
        """What ran before the dialog-raising action of a stopped batch, so the model does not redo it."""
        if not child_index:
            return "."
        return (
            f"; the first {child_index} action(s) of the turn DID run and must not be repeated, the next one ran "
            "but its dialog was cancelled, and the rest did not run."
        )

    def _drop_ran_held_calls(self, run_id: str, index: int, action_count: int) -> bool:
        for service in {id(s): s for s in (self.policy_service, self.perception_service)}.values():
            drop = getattr(service, "drop_held_calls_before", None)
            try:
                if callable(drop) and drop(run_id, index, action_count) is True:
                    return True
            except Exception as exc:  # noqa: BLE001 - falls back to the model repeating the action
                self._recovered(run_id, "drop_held_calls", exc)
        return False

    def check_resume_environment(self, record) -> None:
        """Fail before anything is changed when the run's original provider is not this one.

        A run this process created needs no check (it runs where it started); a run loaded
        from disk does, whatever its stop reason. Compared with the provider this process
        actually built when there is one: settings can change after it is built."""
        if record.run_id in self._created_runs:
            return
        from operon.core.runtime import current_runtime
        from operon.store.checkpoints import check_resume_environment

        root = getattr(self.run_store, "root_dir", None)
        provider = current_runtime().desktop_provider
        check_resume_environment(
            record, Path(root) if root is not None else None, getattr(provider, "name", None)
        )
        self._created_runs.add(record.run_id)

    async def _prepare_interrupted(self, run_id: str, record) -> RunStatus | None:
        """Rebuild this process's per-run state for a run a restart interrupted (KAN-80).

        The page and the model conversation are gone: re-register the run on the executor,
        drop any backend conversation, and hand the model a "resuming" hint listing what the
        saved state says was done. Returns what the run was doing when it died."""
        # Claimed first, across processes: another live process resuming this run is refused
        # before any state changes or the executor is configured (raises RunOwnedError).
        acquire = getattr(self.run_store, "acquire_run", None)
        if callable(acquire):
            acquire(run_id)  # this process now steps the run: no scan may call it interrupted
        # Cleared before the first await: a second resume of the same run then fails fast
        # instead of resetting the backend or adding a second hint.
        origin, record.interrupted_from = record.interrupted_from, None
        record.stop_reason = None
        clear_failure(record)
        async with self._executor_lock:
            if hasattr(self.executor, "set_current_run_id"):
                self.executor.set_current_run_id(run_id)
            if hasattr(self.executor, "configure_run"):
                try:
                    self.executor.configure_run(run_id, headless=record.headless, mode=record.mode)
                except Exception as exc:
                    self._recovered(run_id, "configure_run", exc)
        self._release_backend_run_state(run_id)
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            add_hints([build_resume_hint(record)], source=RESUME_HINT_SOURCE, run_id=run_id)
        return origin

    async def resume_interrupted(self, run_id: str) -> RunResponse:
        """Continue a run a restart interrupted, from the current screen (KAN-80).

        A run that was waiting for the person when the process died stays waiting: the
        gate it paused on is forgotten, so the person approves again before anything runs."""
        record = await self.run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"Run {run_id!r} not found")
        if record.stop_reason is not StopReason.INTERRUPTED or record.status is not RunStatus.WAITING_FOR_USER:
            raise ValueError(f"Run {run_id!r} is not interrupted")
        self.check_resume_environment(record)
        origin = await self._prepare_interrupted(run_id, record)
        if origin is RunStatus.WAITING_FOR_USER:
            record.stop_reason = StopReason.WAITING_FOR_USER
            record.pause_reason = "Resumed after a restart. Approve to continue from the current screen."
            await self.run_store.save_state(record)
            return RunResponse(
                run_id=run_id, status=record.status, intent=record.intent, step_count=record.step_count
            )
        return await self.resume_run(run_id)

    async def step_run(self, request: StepRequest) -> RunResponse:
        lock = self._run_locks.get(request.run_id)
        if lock is None:
            lock = self._run_locks[request.run_id] = asyncio.Lock()
        async with lock:
            # The executor has one "current run" that capture and execute read, so
            # steps of different runs must not interleave on it: each step selects
            # its run first and holds the executor until it ends.
            async with self._executor_lock:
                if hasattr(self.executor, "set_current_run_id"):
                    self.executor.set_current_run_id(request.run_id)
                try:
                    return await self._step_run_locked(request)
                finally:
                    await self._flush_recovered_errors(request.run_id)

    def _recovered(
        self, run_id: str | None, site: str, exc: BaseException, *, level: int = logging.WARNING
    ) -> None:
        """Record that the loop carried on after ``exc`` at ``site`` (KAN-158).

        Logs it with its traceback and counts it for the run, so recoveries show
        up in the run's state (``recovered_errors``) and in benchmark reports
        instead of passing silently. In strict mode (``OPERON_STRICT_ERRORS``) a
        bug (``_BUG_EXCEPTIONS``) is re-raised instead. Call it from the except block.
        """
        if _strict_errors() and isinstance(exc, _BUG_EXCEPTIONS):
            raise exc
        logger.log(
            level,
            "recovered at %s (run %s): %s: %s",
            site,
            (run_id or "-")[:8],
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        if run_id:
            counts = self._recovered_errors.setdefault(run_id, {})
            counts[site] = counts.get(site, 0) + 1

    async def _flush_recovered_errors(self, run_id: str) -> None:
        """Save the run's pending recovery counts into its state (``recovered_errors``)."""
        pending = self._recovered_errors.pop(run_id, None)
        if not pending:
            return
        record = await self.run_store.get_run(run_id)
        if record is None:
            return
        merged = dict(record.recovered_errors)
        for site, count in pending.items():
            merged[site] = merged.get(site, 0) + count
        record.recovered_errors = merged
        await self.run_store.save_state(record)

    async def _step_run_locked(self, request: StepRequest) -> RunResponse:
        """Run one step as a sequence of stages over a typed ``StepContext`` (KAN-155).

        capture -> perceive -> decide -> gate -> execute/verify (with in-step
        retries) -> recover -> record -> finish. A stage that ends the step early
        (run not steppable, perception failed, run cancelled, action gated for a
        human) returns the step's ``RunResponse``; otherwise it returns ``None``.
        """
        begun = await self._begin_step(request)
        if isinstance(begun, RunResponse):
            return begun
        ctx = begun
        await self._warmup_stage(ctx)
        await self._consume_force_fresh_perception(ctx.record)
        await self._capture_stage(ctx)
        self._consume_override_hint(ctx)
        self._add_task_note(ctx)
        self._add_secrets_hint(ctx)
        if (early := await self._perceive_stage(ctx)) is not None:
            return early
        await self._decide_stage(ctx)
        if (early := await self._gate_stage(ctx)) is not None:
            return early
        await self._execute_and_verify_stage(ctx)
        await self._recover_stage(ctx)
        self._record_step_stage(ctx)
        response = await self._finish_step_stage(ctx)
        return await self._pause_for_cancelled_dialog(ctx, response)

    async def _begin_step(self, request: StepRequest) -> StepContext | RunResponse:
        """Load the run; refuse a run that may not step; build the step's context."""
        record = await self.run_store.get_run(request.run_id)
        if record is None:
            raise ValueError(f"Run {request.run_id!r} not found")
        if record.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            # Paused (by hand or by a gate), cancelled or finished: a step must not run.
            # A gate pause is approved only through resume_run, never by stepping.
            return RunResponse(
                run_id=record.run_id,
                status=record.status,
                intent=record.intent,
                step_count=record.step_count,
            )
        # A run loaded from disk may have started in another environment provider: refuse
        # before this step changes anything (a no-op for browser runs and runs created here).
        self.check_resume_environment(record)
        if record.step_count >= record.max_steps:
            updated = await self.fail_for_max_steps(record)
            try:
                await self._cleanup_completed_run(record.run_id)
            except Exception as exc:
                self._recovered(record.run_id, "cleanup_after_max_steps", exc)
            await self._save_efficiency(record)
            return RunResponse(
                run_id=updated.run_id,
                status=updated.status,
                intent=updated.intent,
                step_count=updated.step_count,
            )

        step_index = record.step_count + 1
        _trace("─" * 60)
        _trace("STEP", f"step={step_index}  run_id={record.run_id}")

        ctx = StepContext(
            record=record,
            step_index=step_index,
            before_artifact_path=self._before_artifact_path(record.run_id, step_index),
            after_artifact_path=self._after_artifact_path(record.run_id, step_index),
        )
        self._prepare_step_artifacts(record.run_id, step_index, ctx.before_artifact_path, ctx.after_artifact_path)
        return ctx

    async def _warmup_stage(self, ctx: StepContext) -> None:
        """First step only: let the OS and browser window settle (skipped in test safe mode)."""
        if ctx.step_index != 1 or self._test_safe_mode_enabled():
            return
        logger.info("Bootstrap warmup: Allowing OS and Browser window to hydrate...")
        if (
            self.environment is UnifiedEnvironment.BROWSER
            and not ctx.record.start_url
            and hasattr(self.executor, "execute")
        ):
            warmup = AgentAction(action_type=ActionType.NAVIGATE, url="https://www.google.com")
            # The warmup is an agent-initiated navigation like any other, so it must
            # honor the trust gate (e.g. OPERON_TRUST_ALLOW_DOMAINS) instead of bypassing it.
            if self._trust_gate.evaluate(warmup).verdict is TrustVerdict.ALLOW:
                logger.info("Bootstrap warmup: no start URL — navigating to google.com")
                await self.executor.execute(warmup)
            else:
                logger.info("Bootstrap warmup: trust gate blocks google.com — skipping warmup navigation")
        await asyncio.sleep(2.5)

    async def _capture_stage(self, ctx: StepContext) -> None:
        _trace("  1 CAPTURE", "taking screenshot via CaptureService")
        _t0 = time.perf_counter()
        ctx.frame = await self.capture_service.capture(ctx.record)
        _trace("  1 CAPTURE OK", f"{ctx.record_stage('capture', _t0):.2f}s  frame={type(ctx.frame).__name__}")

    def _consume_override_hint(self, ctx: StepContext) -> None:
        """Hand the model any correction the user sent during a HITL pause.

        Runs before perceive(): combined backends (Gemini) choose the action inside
        perceive(), so a hint added after it would only reach the model a step late.
        """
        override_hint = run_control.consume_override_hint(ctx.run_id)
        if not override_hint:
            return
        ctx.override_text = (
            f"[USER OVERRIDE HINT] The user provided this correction at step {ctx.step_index}: "
            f"'{override_hint}'. Incorporate it into your next action decision."
        )
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            add_hints([ctx.override_text], source="user_override", run_id=ctx.run_id)
            ctx.override_sent = True
        logger.info("Consumed user override hint for step %d: %r", ctx.step_index, override_hint)

    async def _perceive_stage(self, ctx: StepContext) -> RunResponse | None:
        """Hints, then perceive the captured frame and store the observation."""
        record = ctx.record
        _trace("  2 PERCEIVE", f"sending screenshot to Gemini  backend={type(self.perception_service).__name__}")
        _t0 = time.perf_counter()
        try:
            perception = await self._perceive_with_liveness_retry(
                record, ctx.frame, on_recapture=lambda fresh: setattr(ctx, "frame", fresh)
            )
        except Exception as exc:
            # The run fails with the error recorded (pre-step failure log); a bug
            # still re-raises in strict mode.
            if _strict_errors() and isinstance(exc, _BUG_EXCEPTIONS):
                raise
            _trace("  2 PERCEIVE FAIL", str(exc))
            return await self._record_pre_step_perception_failure(
                record=record,
                step_index=ctx.step_index,
                before_artifact_path=ctx.before_artifact_path,
                error=exc,
            )
        _trace("  2 PERCEIVE OK", f"{ctx.record_stage('perceive', _t0):.2f}s  page_hint={perception.page_hint.value!r}  elements={len(perception.visible_elements)}")
        perception = self._infer_focused_element(record, perception)
        ctx.perception = self._with_capture_origin(perception, ctx.frame)
        ctx.perception_debug = self._resolve_model_debug_artifacts(ctx.run_id, ctx.step_index, "perception", self.perception_service)
        ctx.state = await self.run_store.update_state(ctx.run_id, ctx.perception)
        self._sync_progress_state_with_perception(ctx.state, ctx.perception)

        if ctx.override_text and not ctx.override_sent:
            # A backend without advisory hints: the policy reads it from the rule trace.
            ctx.state.last_rule_trace = ctx.override_text
        return None

    async def _decide_stage(self, ctx: StepContext) -> None:
        """Choose the step's action: the model's decision, passed through."""
        _trace("  3 POLICY", "ThinPolicy -> the backend's decision, unchanged")
        _t0 = time.perf_counter()
        decision = await self.policy_service.choose_action(ctx.state, ctx.perception)
        _enriched = self._attach_target_context(decision.action, ctx.perception, ctx.state.benchmark)
        if _enriched is not decision.action:
            decision = decision.model_copy(update={"action": _enriched})
        ctx.decision = decision
        _trace("  3 POLICY OK", f"{ctx.record_stage('policy', _t0):.2f}s  action={decision.action.action_type.value!r}  target={decision.action.target_element_id!r}  rationale={decision.rationale[:80]!r}")
        ctx.policy_debug = self._resolve_model_debug_artifacts(ctx.run_id, ctx.step_index, "policy", self.policy_service)
        ctx.state.current_subgoal = decision.active_subgoal

    async def _gate_stage(self, ctx: StepContext) -> RunResponse | None:
        """Stop if the run was cancelled; pause if the action needs a human."""
        if await self._run_cancelled(ctx.run_id):
            # Stopped while this step captured and asked the model: run nothing.
            _trace("  3 POLICY -> CANCELLED", "run was stopped during this step; not executing")
            await self._save_efficiency(ctx.record)
            return self._cancelled_response(ctx.record)

        # Human-in-the-loop, coordinate thrash and pre-execution gates: a blocked
        # action never runs; the run pauses for a human instead.
        pause_decision = await self._gate_pause_decision(ctx.record, ctx.state, ctx.decision, frame=ctx.frame)
        if pause_decision is None:
            # Reading the click text (KAN-163) waits on OCR: a stop that came in
            # meanwhile must still keep this action from running.
            if getattr(self, "_click_reader", None) is not None and await self._run_cancelled(ctx.run_id):
                _trace("  3 POLICY -> CANCELLED", "run was stopped while the gate read the click text")
                return self._cancelled_response(ctx.record)
            return None
        self._disarm_dialog_accept(ctx.run_id)  # a gate pause never leaves an accept armed
        return await self._pause_for_user(
            record=ctx.record,
            state=ctx.state,
            perception=ctx.perception,
            decision=pause_decision,
            perception_debug=ctx.perception_debug,
            policy_debug=ctx.policy_debug,
            step_index=ctx.step_index,
            before_artifact_path=ctx.before_artifact_path,
            step_ctx=ctx,
        )

    async def _execute_and_verify_stage(self, ctx: StepContext) -> None:
        """Execute the action once and verify it. A failure is the model's to see next step."""
        # Set run context on the executor so launched processes are tracked per run
        if hasattr(self.executor, "set_current_run_id"):
            self.executor.set_current_run_id(ctx.run_id)
        self._drop_stale_dialog(ctx)
        self._set_dialog_policy(ctx)
        try:
            await self._execute_attempt(ctx)
        finally:
            # A person's yes covers the one action it was given for, whatever happened to it.
            self._dialog_approved.pop(ctx.run_id, None)
            self._disarm_dialog_accept(ctx.run_id)
        self._absorb_dialog_message(ctx)
        self._note_cancelled_dialog(ctx)
        self._pass_on_change_note(ctx)
        await self._verify_attempt(ctx)
        self._check_expected_change(ctx)

    async def _execute_attempt(self, ctx: StepContext) -> None:
        decision = ctx.decision
        _trace("  4 EXECUTE", f"action={decision.action.action_type.value!r}  executor={type(self.executor).__name__}")
        _t0 = time.perf_counter()
        self._install_batch_child_check(ctx.run_id, decision.action)
        try:
            ctx.state, executed_action = await self._execute_action(
                record=ctx.record,
                state=ctx.state,
                decision_action=decision.action,
                step_index=ctx.step_index,
                use_unified_executor=True,
            )
        finally:
            if hasattr(self.executor, "batch_child_check"):
                self.executor.batch_child_check = None
        _trace("  4 EXECUTE OK" if executed_action.success else "  4 EXECUTE FAIL", f"{ctx.record_stage('execute', _t0):.2f}s  success={executed_action.success}  detail={executed_action.detail!r}")
        ctx.executed_action = self._relocate_after_artifact(executed_action, ctx.after_artifact_path)

    def _add_secrets_hint(self, ctx: StepContext) -> None:
        """Tell the model which secret NAMES it may type (OPERON_SECRETS, KAN-103).

        Names and the placeholder syntax only; a value is never read here. Allowed in the
        thin harness: it changes what the model sees, never its action.
        """
        names = available_names()
        if not names:
            return
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            add_hints([secrets_hint(names)], source="secrets", run_id=ctx.run_id)

    def _add_task_note(self, ctx: StepContext) -> None:
        """Send the running task note with this step (OPERON_TASK_NOTE, KAN-102).

        Built from the run's own history, no model call. Allowed in the thin harness: it
        changes what the model sees, never its action. Saved as ``task_note.txt``.
        """
        settings = get_settings()
        if not settings.task_note:
            return
        state = ctx.record
        note = build_task_note(
            state.intent,
            state.step_count,
            state.action_history,
            state.verification_history,
            plan_steps=self._saved_plan_steps(ctx.run_id),
            max_items=settings.task_note_max_items,
        )
        if not note:
            return
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            add_hints([note], source="task_note", run_id=ctx.run_id)
        try:
            step_dir = Path(self.run_store.root_dir) / ctx.run_id / f"step_{ctx.step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "task_note.txt").write_text(note, encoding="utf-8")
        except OSError as exc:
            self._recovered(ctx.run_id, "task_note", exc)

    def _saved_plan_steps(self, run_id: str) -> list[str]:
        """The planner's steps from ``plan.json`` (empty when no plan was written)."""
        try:
            data = json.loads((Path(self.run_store.root_dir) / run_id / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        steps = data.get("steps") if isinstance(data, dict) else None
        return [s for s in steps if isinstance(s, str)] if isinstance(steps, list) else []

    def _pass_on_change_note(self, ctx: StepContext) -> None:
        """Hand the model the executor's between-step change note, as text (KAN-111).

        The executor wrote it while the screen settled (OPERON_CHANGE_NOTE). It goes out
        as an advisory hint with the next step; no extra image, no model call.
        """
        pop = getattr(self.executor, "pop_change_note", None)
        if not callable(pop):
            return
        note = pop()
        if not isinstance(note, str) or not note:
            return
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            add_hints([note], source="change_note", run_id=ctx.run_id)

    def _send_expect_instruction(self, run_id: str, step_index: int = 0) -> None:
        """Ask the model to write an ``Expect:`` line before each action (OPERON_EXPECT_CHECK)."""
        if not get_settings().expect_check:
            return
        try:
            if self._backend_reports_expected_change():
                return
            add_hints = getattr(self.perception_service, "add_advisory_hints", None)
            if callable(add_hints):
                add_hints([EXPECT_INSTRUCTION], source="expect_instruction", run_id=run_id)
                self._expect_reminded[run_id] = step_index
        except Exception as exc:  # noqa: BLE001 - an advisory hint must never fail a run
            self._recovered(run_id, "expect_instruction", exc)

    def _backend_reports_expected_change(self) -> bool:
        """Whether the backend's own answer carries expected_change (set by a backend that reports it)."""
        return getattr(self.perception_service, "reports_expected_change", False) is True

    def _check_expected_change(self, ctx: StepContext) -> None:
        """Compare the model's expectation with the screen change; hint on a mismatch (KAN-116).

        Advisory only: never touches the verification result, retries or the action.
        """
        if not get_settings().expect_check:
            return
        try:
            executed = ctx.executed_action
            if executed is None or not executed.success or executed.action.action_type in NO_SCREEN_EFFECT_ACTIONS:
                return
            if ctx.dialog_captured:
                return
            ratio = ctx.screen_change_ratio
            if ratio is None:
                return
            # A missing screenshot reads as "no change"; that is not evidence.
            after_path = executed.artifact_path
            if not (ctx.before_artifact_path and after_path):
                return
            if not (Path(ctx.before_artifact_path).exists() and Path(after_path).exists()):
                return
            decision = ctx.decision
            # A backend's NONE is a real "none" only when the model actually supplied the field.
            reliable = self._backend_reports_expected_change() and "expected_change" in getattr(
                decision, "model_fields_set", ()
            )
            expectation = parse_expectation(
                decision.rationale, decision.expected_change, field_is_reliable=reliable
            )
            if expectation is None:
                # The model did not say what it expects: remind it now and then.
                if ctx.step_index - self._expect_reminded.get(ctx.run_id, 0) >= EXPECT_REMINDER_EVERY:
                    self._send_expect_instruction(ctx.run_id, ctx.step_index)
                return
            observation = compare(expectation, ratio)
            note = mismatch_note(expectation, observation) if observation else None
            if note:
                add_hints = getattr(self.perception_service, "add_advisory_hints", None)
                if callable(add_hints):
                    add_hints([note], source="expect_check", run_id=ctx.run_id)
            step_dir = Path(self._artifacts.before_path(ctx.run_id, ctx.step_index)).resolve().parent
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "expect_check.json").write_text(
                json.dumps(
                    {
                        "step_index": ctx.step_index,
                        "kind": expectation.kind.value,
                        "text": expectation.text,
                        "source": expectation.source,
                        "change_ratio": round(ratio, 5),
                        "mismatch": note is not None,
                        "note": note,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - an advisory check must never fail a step
            self._recovered(ctx.run_id, "expect_check", exc)

    def _set_dialog_policy(self, ctx: StepContext) -> None:
        """Tell the executor how to answer the page's confirm/prompt/beforeunload dialogs (KAN-172).

        Mirrors the trust gate's confirmations: ``auto`` permission mode (or the gate
        turned off) accepts them, anything else cancels them and asks the person."""
        # The agent acts now: nothing it does runs with the person's "I am driving" rule on.
        self._set_person_driving(ctx.run_id, False)
        set_policy = getattr(self.executor, "set_dialog_policy", None)
        if not callable(set_policy):
            return
        mode = resolve_permission_mode(getattr(ctx.state, "permission_mode", None))
        # With the gate off only ``ask`` still asks (apply_permission_mode, docs/api.md).
        set_policy(
            ctx.run_id,
            accept=mode is PermissionMode.AUTO or (not self._trust_gate.enabled and mode is not PermissionMode.ASK),
        )
        # The person approved a dialog: accept that one (type and message), once, in this step only.
        approved = self._dialog_approved.get(ctx.run_id)
        arm = getattr(self.executor, "arm_accept_next_dialog", None)
        if approved is not None and callable(arm):
            arm(ctx.run_id, approved[0], approved[1])

    def _drop_stale_dialog(self, ctx: StepContext) -> None:
        """Forget a dialog that fired after the previous action returned: it is not this action's."""
        pop = getattr(self.executor, "pop_pending_dialog", None)
        stale = pop(ctx.run_id) if callable(pop) else None
        if isinstance(stale, tuple) and len(stale) == 3:
            logger.info("Cancelled a page dialog that fired after the previous action returned")
            ctx.cancelled_dialog = stale
            self._hint_cancelled_dialog(ctx, asked=False)
            ctx.cancelled_dialog = None

    def _disarm_dialog_accept(self, run_id: str) -> None:
        disarm = getattr(self.executor, "disarm_accept_next_dialog", None)
        if callable(disarm):
            disarm(run_id)

    def _note_cancelled_dialog(self, ctx: StepContext) -> None:
        """Pick up a dialog the executor cancelled and tell the model about it (advisory hint).

        The pause for the person comes at the end of the step (``_pause_for_cancelled_dialog``)."""
        pop = getattr(self.executor, "pop_pending_dialog", None)
        if not callable(pop):
            return
        pending = pop(ctx.run_id)
        if not (isinstance(pending, tuple) and len(pending) == 3):
            return
        ctx.cancelled_dialog = pending
        ctx.dialog_captured = True

    def _hint_cancelled_dialog(self, ctx: StepContext, *, asked: bool) -> None:
        """Tell the model (advisory hint) about a dialog Operon cancelled, and whether the person was asked."""
        if ctx.cancelled_dialog is None:
            return
        kind, message = ctx.cancelled_dialog[0], ctx.cancelled_dialog[1]
        add_hints = getattr(self.perception_service, "add_advisory_hints", None)
        if callable(add_hints):
            tail = "and asked the person" if asked else "(the person was not asked)"
            add_hints(
                [f"The page showed a {kind} dialog: {message!r}. Operon cancelled it {tail}."],
                source="dialog",
                run_id=ctx.run_id,
            )

    async def _pause_for_cancelled_dialog(self, ctx: StepContext, response: RunResponse) -> RunResponse:
        """Pause the run for the person after the step cancelled a page dialog (KAN-172).

        The same pause the trust gate uses (WAITING_FOR_USER, approved by resuming). The backend
        holds the action that raised the dialog (``_note_not_executed``): approving replays it
        once with the dialog accepted, declining drops it. Nothing here picks or replaces an
        action; the only re-run is the one the person approved."""
        if ctx.cancelled_dialog is None:
            return response
        if response.status is not RunStatus.RUNNING:
            self._hint_cancelled_dialog(ctx, asked=False)  # ended or paused for another reason
            return response
        self._hint_cancelled_dialog(ctx, asked=True)
        kind, message, child_index = ctx.cancelled_dialog
        shown = message if len(message) <= 300 else message[:300] + "..."
        question = f'The page asked: "{shown}" Accept?' if shown else f"The page showed a {kind} dialog. Accept?"
        gate_text = f"{question} Resume to accept it and run the action again, or correct/stop."
        self._note_not_executed(
            ctx.run_id, f"the page's {kind} dialog was cancelled and the person was asked", ran=child_index or 0
        )
        children = ctx.decision.action.actions if ctx.decision.action.action_type is ActionType.BATCH else None
        self._dialog_pauses[ctx.run_id] = _DialogPause(
            kind, message, self._has_held_actions(ctx.run_id), child_index, len(children) if children else 1
        )
        pause = ctx.decision.model_copy(
            update={"action": AgentAction(action_type=ActionType.WAIT_FOR_USER, text=gate_text)}
        )
        _trace("  8 DIALOG -> HITL", f"{kind} dialog cancelled; asking the person")
        return await self._pause_for_user(
            record=ctx.record,
            state=ctx.state,
            perception=ctx.perception,
            decision=pause,
            perception_debug=ctx.perception_debug,
            policy_debug=ctx.policy_debug,
            step_index=ctx.step_index,
            before_artifact_path=ctx.before_artifact_path,
            log_step=False,
            step_ctx=ctx,
        )

    def _absorb_dialog_message(self, ctx: StepContext) -> None:
        """Fold a JS dialog message the executor accepted into this step's perception summary.

        Marks the step (``ctx.dialog_captured``) so the expect-check does not read the
        screen change against the model's expectation. Only updates in-memory state,
        avoiding a second run_store.update_state call.
        """
        if not hasattr(self.executor, "pop_last_dialog_message"):
            return
        dialog_msg = self.executor.pop_last_dialog_message(ctx.run_id)
        if not dialog_msg:
            return
        ctx.dialog_captured = True
        logger.info("JS dialog captured: %r — injecting into perception summary", dialog_msg)
        ctx.perception = ctx.perception.model_copy(
            update={"summary": (ctx.perception.summary or "") + f" {dialog_msg}"}
        )
        if ctx.state.observation_history:
            # In place: ``ctx.state`` is the stored record, and a copy would drop
            # this step's stop reason and recovery flags.
            ctx.state.observation_history[-1] = ctx.perception

    async def _verify_attempt(self, ctx: StepContext) -> None:
        executed_action = ctx.last_attempt
        _trace("  5 VERIFY", "OwnDoneVerifier: the model's own done ends the run")
        _t0 = time.perf_counter()
        verification = await self.verifier_service.verify(ctx.state, ctx.decision, executed_action)
        ctx.verification = verification
        _trace("  5 VERIFY OK", f"{ctx.record_stage('verify', _t0):.2f}s  status={verification.status.value!r}  stop={verification.stop_condition_met}  stop_reason={verification.stop_reason!r}")
        # A verifier with no model call (OwnDoneVerifier) has no debug artifacts: log none.
        ctx.verification_debug = self._latest_debug_artifacts(self.verifier_service)

        # Compute screen change ratio for progress tracking.
        from operon.agent.perception.screen_diff import (
            compute_screen_change_ratio as _scr,
        )
        _after_path = executed_action.artifact_path
        ctx.screen_change_ratio = (
            _scr(ctx.before_artifact_path, _after_path)
            if ctx.before_artifact_path and _after_path
            else None
        )

    async def _recover_stage(self, ctx: StepContext) -> None:
        """Pick the recovery, update progress tracking, persist the progress trace."""
        executed_action = ctx.last_attempt
        verification = ctx.verification
        _trace("  7 RECOVER", f"ThinRecovery  verification={verification.status.value!r}")
        _t0 = time.perf_counter()
        recovery: RecoveryDecision = await self.recovery_manager.recover(
            ctx.state,
            ctx.decision,
            executed_action,
            verification,
        )
        # Benchmark Integrity Check: reject recovery decisions that bypass verification
        # or claim success without visual confirmation.
        recovery = validate_benchmark_integrity(recovery, verification)
        _trace("  7 RECOVER OK", f"{ctx.record_stage('recover', _t0):.2f}s  strategy={recovery.strategy.value!r}  stop_reason={recovery.stop_reason!r}")
        progress_trace = self._update_progress_state(
            state=ctx.state,
            decision=ctx.decision,
            executed_action=executed_action,
            verification=verification,
            recovery=recovery,
            step_index=ctx.step_index,
            before_artifact_path=ctx.before_artifact_path,
            screen_change_ratio=ctx.screen_change_ratio,
        )
        progress_trace = progress_trace.model_copy(
            update={
                "final_failure_category": progress_trace.final_failure_category
                or recovery.failure_category
                or verification.failure_category
                or executed_action.failure_category,
                "final_stop_reason": progress_trace.final_stop_reason or recovery.stop_reason,
            }
        )
        ctx.recovery = recovery
        ctx.progress_trace_artifact_path = self._persist_progress_trace(ctx.run_id, ctx.step_index, progress_trace)

    def _record_step_stage(self, ctx: StepContext) -> None:
        """Write the step log and fold the step into the run's history and memory."""
        state = ctx.state
        decision = ctx.decision
        executed_action = ctx.last_attempt
        verification = ctx.verification
        recovery = ctx.recovery
        perception_debug = ctx.perception_debug
        policy_debug = ctx.policy_debug
        verification_debug = ctx.verification_debug

        failure = self._build_failure_record(state, decision, executed_action, verification, recovery)

        step_log = StepLog(
            run_id=ctx.run_id,
            step_id=f"step_{ctx.step_index}",
            step_index=ctx.step_index,
            before_artifact_path=ctx.before_artifact_path,
            after_artifact_path=ctx.after_artifact_path,
            perception_debug=perception_debug,
            policy_debug=policy_debug,
            verification_debug=verification_debug,
            perception=ctx.perception,
            policy_decision=decision,
            executed_action=executed_action,
            verification_result=verification,
            recovery_decision=recovery,
            progress_state=state.progress_state,
            progress_trace_artifact_path=ctx.progress_trace_artifact_path,
            failure=failure,
            decision_source=self._decision_source(decision, policy_debug, perception_debug),
            visual_variance=executed_action.visual_variance,
            **ctx.timing_fields(),
        )
        append_step_log(self._run_log_path(ctx.run_id), step_log)

        state.last_rule_trace = None

        if self.memory_store is not None:
            self.memory_store.record_step(
                state=state,
                perception=ctx.perception,
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
                ctx.before_artifact_path,
                ctx.after_artifact_path,
                perception_debug.prompt_artifact_path,
                perception_debug.raw_response_artifact_path,
                perception_debug.parsed_artifact_path,
                *( [perception_debug.usage_artifact_path] if perception_debug.usage_artifact_path else [] ),
                *( [perception_debug.diagnostics_artifact_path] if perception_debug.diagnostics_artifact_path else [] ),
                policy_debug.prompt_artifact_path,
                policy_debug.raw_response_artifact_path,
                policy_debug.parsed_artifact_path,
                *( [policy_debug.usage_artifact_path] if policy_debug.usage_artifact_path else [] ),
                *(
                    [
                        verification_debug.prompt_artifact_path,
                        verification_debug.raw_response_artifact_path,
                        verification_debug.parsed_artifact_path,
                        *( [verification_debug.usage_artifact_path] if verification_debug.usage_artifact_path else [] ),
                        *( [verification_debug.diagnostics_artifact_path] if verification_debug.diagnostics_artifact_path else [] ),
                    ]
                    if verification_debug is not None
                    else []
                ),
                *( [executed_action.execution_trace_artifact_path] if executed_action.execution_trace_artifact_path else [] ),
                ctx.progress_trace_artifact_path,
            ]
        )

    async def _finish_step_stage(self, ctx: StepContext) -> RunResponse:
        """Settle the run's status, apply recovery actions, clean up a finished run."""
        state = ctx.state
        verification = ctx.verification
        recovery = ctx.recovery
        run_id = ctx.run_id

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

        if await self._run_cancelled(run_id):
            # Stopped while this step ran: the stop wins over whatever the step found.
            final_status = RunStatus.CANCELLED
            state.stop_reason = None
        if final_status is RunStatus.RUNNING and run_id in self._pause_requests:
            # Paused by hand while this step ran: don't overwrite the pause.
            final_status = RunStatus.WAITING_FOR_USER
        if final_status is not RunStatus.WAITING_FOR_USER:
            self._pause_requests.discard(run_id)
        if final_status is RunStatus.FAILED:
            note_failure(state, detail=recovery.message)  # KAN-175: cause from the stop reason
        if final_status is RunStatus.RUNNING:
            await self._apply_recovery_actions(state, recovery, ctx.step_index)
        ctx.final_status = final_status

        _trace("  8 LOG", f"StepLog -> run.jsonl  final_status={final_status.value!r}")
        updated = await self.run_store.set_status(run_id, final_status)
        if final_status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
            try:
                await self._cleanup_completed_run(run_id)
            except Exception as exc:
                self._recovered(run_id, f"cleanup_after_{final_status.value}", exc)

        # Persist per-stage durations for tail-latency analysis. Best-effort —
        # the step has already completed by this point.
        self._persist_step_timing(run_id, ctx.step_index, ctx.stage_timings)
        if final_status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
            await self._save_efficiency(state)

        _trace("STEP DONE", f"step={ctx.step_index}  status={final_status.value!r}")
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    async def _save_efficiency(self, state) -> None:
        """Score the finished run (KAN-85) into ``efficiency.json`` and the saved state."""
        root = getattr(self.run_store, "root_dir", None)
        if root is None:
            return
        try:
            await bg_writer.flush()  # the run log and step timings are written in the background
            efficiency = await asyncio.to_thread(run_efficiency, Path(root) / state.run_id)
            (Path(root) / state.run_id / "efficiency.json").write_text(
                json.dumps(efficiency, indent=2) + "\n", encoding="utf-8"
            )
            state.efficiency = efficiency
            await self.run_store.save_state(state)
        except Exception as exc:
            self._recovered(state.run_id, "save_efficiency", exc, level=logging.DEBUG)

    async def _click_text_for_gate(
        self,
        action: AgentAction,
        frame: CaptureFrame | None,
        *,
        run_id: str = "",
        permission_mode: str | None = None,
    ) -> str | None:
        """The text the screenshot shows under a click (KAN-163), or None.

        Only for a pointer action at a point with no element name, with the gate on and
        ``OPERON_CLICK_TEXT`` set: that is the case where the gate otherwise cannot tell
        what is pressed. Skipped when its answer can't matter: the action was already
        approved, or the run's ``auto`` mode skips confirmations. A click inside a batch
        is not read (the batch's own point is None). OCR runs in a worker thread so the
        event loop keeps serving.
        """
        reader = getattr(self, "_click_reader", None)
        if (
            reader is None
            or frame is None
            or not self._trust_gate.enabled
            or action.x is None
            or action.y is None
            or not names_no_target(action)
            or "trust" in self._approved_gates(run_id, action)
            or resolve_permission_mode(permission_mode) is PermissionMode.AUTO
        ):
            return None
        # The action is already in virtual-desktop coordinates (_apply_monitor_origin);
        # the screenshot is the one monitor, so take its origin back off.
        text = await asyncio.to_thread(
            read_click_text,
            frame.artifact_path,
            action.x - frame.monitor_left,
            action.y - frame.monitor_top,
            (frame.width, frame.height),
            reader=reader,
        )
        if text:
            _trace("  3 POLICY -> CLICK_TEXT", repr(text))
        return text

    async def _current_url_for_trust_gate(
        self, run_id: str, *, log: bool = False, secret: bool = False
    ) -> str | None:
        """The browser's current URL, only when the trust gate has a domain allowlist
        to judge it against (or ``secret``: the action types a stored secret, whose
        approval must name the site, KAN-103); None otherwise or when the executor
        has no page.

        ``log`` (the pre-execution check) appends it to the run's
        ``trust_gate_urls.jsonl``, so a replay (evals/replay.py) can show the gate
        the same pages."""
        if not self._trust_gate.enabled or not (self._trust_gate.allow_domains or secret):
            return None
        if not hasattr(self.executor, "current_url_for_run"):
            return None
        try:
            url = await self.executor.current_url_for_run(run_id)
        except Exception as exc:
            self._recovered(run_id, "trust_gate_current_url", exc, level=logging.DEBUG)
            url = None
        if log:
            self._log_trust_gate_url(run_id, url)
        return url

    def _secret_host_changed(self, run_id: str, leaf: AgentAction, current_url: str | None) -> bool:
        """True when ``leaf`` types a secret in the browser, the gate would still ask for it, and
        the page's host is not the one the person approved (None counts as a host: "unknown").
        Desktop (no page URL) is never judged."""
        if not (self._trust_gate.enabled and action_uses_secret(leaf)):
            return False
        if not hasattr(self.executor, "current_url_for_run"):
            return False
        result = self._trust_gate.evaluate(leaf, current_url=current_url)
        if not (result.verdict is TrustVerdict.CONFIRM and result.always):
            return False
        host = (urlparse(current_url).hostname or None) if current_url else None
        return host != self._secret_site_approved.get(run_id)

    def _secret_site_suffix(
        self, leaf: AgentAction, current_url: str | None, whole: AgentAction | None = None
    ) -> str:
        """`` on <host>`` for an approval that types a stored secret in the browser, so the
        person sees where the password goes (batch screenshots have no address bar);
        `` on an unknown site`` when the URL cannot be read. When earlier leaves of the
        batch run first and may change the page, the host is only the page now, and the
        executor re-checks it right before typing. Desktop: unchanged."""
        if not action_uses_secret(leaf) or not hasattr(self.executor, "current_url_for_run"):
            return ""
        host = (urlparse(current_url).hostname or "") if current_url else ""
        text = f" on {host}" if host else " on an unknown site"
        leaves = _leaf_actions(whole) if whole is not None else [leaf]
        if leaves and leaves[0] is not leaf:
            text += " (current page; re-checked before typing, the batch stops if the site changes)"
        return text

    def _log_trust_gate_url(self, run_id: str, url: str | None) -> None:
        root = getattr(self.run_store, "root_dir", None)
        if root is None:
            return
        try:
            with (Path(root) / run_id / "trust_gate_urls.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"url": url_origin_for_log(url)}) + "\n")
        except OSError as exc:
            self._recovered(run_id, "trust_gate_url_log", exc, level=logging.DEBUG)

    def _pre_execution_gate_text(
        self,
        decision: PolicyDecision,
        *,
        intent: str,
        run_id: str,
        current_url: str | None = None,
        permission_mode: str | None = None,
        click_text: str | None = None,
    ) -> str | None:
        """Why ``decision`` must pause for a human before executing, or None to proceed.

        Gates run in order and the first block wins:
        1. Confidence: the operator's threshold (UI slider) is above the decision's
           confidence. Skips WAIT_FOR_USER and STOP, which already pause or end the run.
        2. Trust gate (RFC 0001 Move 5): deny/confirm policy; a no-op unless
           OPERON_TRUST_GATE is on. DENY = blocked, CONFIRM = high-risk. The run's
           permission mode (KAN-89) then adjusts it: ``ask`` confirms every action,
           ``auto`` skips confirmations; neither relaxes a DENY.
        3. File guard: a read_text / upload_file* path the user never named, outside
           the run dir and OPERON_ALLOWED_FILE_ROOTS.

        Gates 1-2 (and the coordinate-thrash pause in step) can be approved by
        resuming (see ``resume_run``); the file guard cannot, since its fix is
        naming the path in the task or allowing its folder.
        """
        action = decision.action
        fingerprint = self._gate_fingerprint(action)
        approved = self._approved_gates(run_id, action)
        self._trust_approved[run_id] = "trust" in approved
        threshold = run_control.get_confidence_threshold()
        if (
            "confidence" not in approved
            and threshold > 0.0
            and action.action_type is not ActionType.WAIT_FOR_USER
            and action.action_type is not ActionType.STOP
            and decision.confidence < threshold
        ):
            _trace("  3 POLICY -> CONFIDENCE_GATE", f"confidence={decision.confidence:.2f} < threshold={threshold:.2f}")
            self._pending_gate_approvals[run_id] = (fingerprint, "confidence", "", "")
            return (
                f"Confidence {decision.confidence:.0%} is below your threshold {threshold:.0%}. "
                f"Planned action: {action.action_type.value}. "
                "Click Proceed to approve or enter a correction hint."
            )

        # A batch runs each of its actions straight through the executor, so the
        # trust gate and file guard judge every one of them, not the container.
        leaves = _leaf_actions(action)
        if "trust" not in approved:
            # A host approved earlier must not authorise this action's secret (KAN-103).
            self._secret_site_approved.pop(run_id, None)
            verdicts = [
                (
                    leaf,
                    apply_permission_mode(
                        self._trust_gate.evaluate(
                            leaf,
                            current_url=current_url,
                            stated_reason=decision.rationale,
                            click_text=click_text if leaf is action else None,
                            intent=intent,
                        ),
                        leaf,
                        permission_mode,
                    ),
                )
                for leaf in leaves
            ]
            blocked = [pair for pair in verdicts if pair[1].verdict is not TrustVerdict.ALLOW]
            # A DENY anywhere in the batch outranks a CONFIRM, and a leaf that types a
            # secret outranks the other CONFIRMs, so the approval always shows the secret
            # (KAN-103); sort is stable, so the rest keep their order.
            blocked.sort(
                key=lambda pair: (pair[1].verdict is not TrustVerdict.DENY, not action_uses_secret(pair[0]))
            )
            if blocked:
                leaf, trust = blocked[0]
                _trace("  3 POLICY -> TRUST_GATE", f"{trust.verdict.value} (matched {trust.matched!r})")
                self._pending_gate_approvals[run_id] = (
                    fingerprint, "trust", trust.category, trust.verdict.value
                )
                if action_uses_secret(action):
                    # What the person approves: the secret may only be typed on this host.
                    self._secret_site_approved[run_id] = (urlparse(current_url).hostname or None) if current_url else None
                if trust.verdict is TrustVerdict.DENY:
                    secret_leaf = next((item for item in leaves if action_uses_secret(item)), None)
                    secret_note = (
                        f" It also types {_describe_for_approval(secret_leaf)}"
                        f"{self._secret_site_suffix(secret_leaf, current_url, action)}."
                        if secret_leaf is not None
                        else ""
                    )
                    return (
                        f"Blocked by policy: {leaf.action_type.value} action "
                        f"({trust.reason}: '{trust.matched}').{secret_note} It will not run autonomously — "
                        "review and resume to override, or stop the run."
                    )
                site = self._secret_site_suffix(leaf, current_url, action)
                if resolve_permission_mode(permission_mode).value == "ask" and trust.reason.startswith("ask mode"):
                    return (
                        f"Approval needed (ask mode): {_describe_for_approval(leaf)}{site}. "
                        "Approve to run it, or correct/stop."
                    )
                if trust.category:
                    return (
                        f"{trust.category.capitalize()}: approve {_describe_for_approval(leaf)}"
                        f"{site}? "
                        f"(matched '{trust.matched}'). Resume to approve, or correct/stop."
                    )
                return (
                    f"High-risk action needs approval: {leaf.action_type.value} "
                    f"(matched '{trust.matched}'). Resume to approve, or correct/stop."
                )

        elif leaves and self._secret_host_changed(run_id, leaves[0], current_url):
            # Approved on one site, but the page is on another now (a redirect while the run
            # waited): ask again, naming the site it is on now (KAN-103).
            self._approved_gate_actions.pop(run_id, None)
            self._trust_approved[run_id] = False
            self._secret_site_approved[run_id] = (urlparse(current_url).hostname or None) if current_url else None
            self._pending_gate_approvals[run_id] = (fingerprint, "trust", "account", TrustVerdict.CONFIRM.value)
            return (
                f"Account: approve {_describe_for_approval(leaves[0])}"
                f"{self._secret_site_suffix(leaves[0], current_url)}? "
                "(the page changed since you approved). Resume to approve, or correct/stop."
            )

        for leaf in leaves:
            if leaf.action_type not in FILE_PATH_ACTIONS:
                continue
            blocked_path = blocked_file_path(
                leaf, intent=intent, run_dir=Path(self.run_store.root_dir) / run_id
            )
            if blocked_path is not None:
                _trace("  3 POLICY -> FILE_GUARD", f"blocked path {blocked_path!r}")
                return (
                    f"Blocked {leaf.action_type.value}: the path {blocked_path!r} is not "
                    "named in the task and is outside the run directory and OPERON_ALLOWED_FILE_ROOTS. "
                    "Restart with the path in the task, allow its folder, or stop the run."
                )
        self._approved_gate_actions.pop(run_id, None)  # approval used: the action runs now
        return None

    def _approved_gates(self, run_id: str, action: AgentAction) -> frozenset[str]:
        """Gates a human already approved for this exact action (see resume_run)."""
        prior = self._approved_gate_actions.get(run_id)
        if prior is None:
            return frozenset()
        if prior[0] != self._gate_fingerprint(action):
            del self._approved_gate_actions[run_id]  # the policy moved on
            return frozenset()
        return prior[1]

    def _gate_fingerprint(self, action: AgentAction) -> str:
        """What a human approves at a gate pause: the action's type plus the text, URL
        and target name the trust gate judges — not coordinates, which shift between
        perceptions of the same screen."""
        if action.action_type is ActionType.BATCH:
            # Approving one batch must not approve a different one.
            # JSON keeps child boundaries unambiguous whatever text the children carry.
            return "batch" + json.dumps([self._gate_fingerprint(child) for child in action.actions or []])
        return f"{action.action_type.value}|{self._trust_gate.describe(action)}"

    async def _gate_pause_decision(
        self, record, state, decision: PolicyDecision, *, frame: CaptureFrame | None = None
    ) -> PolicyDecision | None:
        """The WAIT_FOR_USER decision to pause on before ``decision`` runs, or None
        when it may run: the policy asked for a human, the target thrashed, or a
        pre-execution gate (confidence, trust, file guard) blocked it."""
        if decision.action.action_type is ActionType.WAIT_FOR_USER:
            _trace("  3 POLICY -> WAIT_FOR_USER", "pausing run")
            return decision

        # Retry Exhaustion / Coordinate Thrash guard: if this action targets a (x, y)
        # that has already failed 3 times, escalate to HITL instead of attempting again.
        if (
            "coordinate_thrash" not in self._approved_gates(record.run_id, decision.action)
            and self._coordinate_thrash_detected(state, decision.action)
        ):
            self._pending_gate_approvals[record.run_id] = (self._gate_fingerprint(decision.action), "coordinate_thrash", "", "")
            _trace("  3 POLICY -> COORDINATE_THRASH_HITL", "3 coord failures at same point — escalating to HITL")
            thrash_x, thrash_y = decision.action.x, decision.action.y
            self._note_not_executed(record.run_id, "repeated failures at the same point")
            return decision.model_copy(update={
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

        current_url = await self._current_url_for_trust_gate(
            record.run_id, log=True, secret=action_uses_secret(decision.action)
        )
        gate_text = self._pre_execution_gate_text(
            decision,
            intent=state.intent,
            run_id=record.run_id,
            current_url=current_url,
            permission_mode=getattr(state, "permission_mode", None),
            click_text=await self._click_text_for_gate(
                decision.action, frame, run_id=record.run_id, permission_mode=getattr(state, "permission_mode", None)
            ),
        )
        if gate_text is None:
            return None
        self._note_not_executed(record.run_id, gate_text)
        return decision.model_copy(update={
            "action": AgentAction(action_type=ActionType.WAIT_FOR_USER, text=gate_text)
        })

    def _install_batch_child_check(self, run_id: str, action: AgentAction) -> None:
        """Have the executor check before each of a batch's later children: that the
        run wasn't stopped meanwhile, and, with a domain allowlist, the live page
        (an earlier child may have opened one the pre-execution check never saw)."""
        if action.action_type is not ActionType.BATCH or not hasattr(self.executor, "batch_child_check"):
            return
        check_domains = bool(
            self._trust_gate.enabled and self._trust_gate.allow_domains and not self._trust_approved.get(run_id)
        )

        async def check(child: AgentAction) -> str | None:
            if await self._run_cancelled(run_id):
                return "the run was stopped"
            # A nested batch runs its first leaf next (its later children get their
            # own check), so judge that leaf, not the container.
            leaves = _leaf_actions(child)
            if not leaves:
                return None
            leaf = leaves[0]
            secret = self._trust_gate.enabled and action_uses_secret(leaf)
            if not (check_domains or secret):
                return None
            current_url = await self._current_url_for_trust_gate(run_id, secret=secret)
            result = self._trust_gate.evaluate(leaf, current_url=current_url)
            if result.verdict is TrustVerdict.DENY:
                return f"blocked by policy ({result.reason}: '{result.matched}')"
            if secret and hasattr(self.executor, "current_url_for_run"):
                # A secret goes only to the site the person approved (KAN-103); an earlier
                # step of the batch may have moved the page. The step then fails before
                # anything is typed and the secret type is gated again on its own.
                host = (urlparse(current_url).hostname or None) if current_url else None
                if (
                    result.verdict is TrustVerdict.CONFIRM
                    and result.always
                    and (host is None or host != self._secret_site_approved.get(run_id))
                ):
                    return "the page is not the site approved for typing a secret; nothing was typed"
            return None

        self.executor.batch_child_check = check

    def _note_not_executed(self, run_id: str, reason: str, ran: int = 0) -> None:
        """Tell the policy a gate stopped its decision before it ran, so a backend
        that answers the model's tool calls does not report it as done."""
        note = getattr(self.policy_service, "note_not_executed", None)
        if note is None:
            return
        try:
            # ``ran``: the first actions of a stopped batch really ran; the backend says so per call.
            note(run_id, reason, ran=ran) if ran else note(run_id, reason)
        except Exception as exc:  # noqa: BLE001 - a courtesy to the backend, never a blocker
            self._recovered(run_id, "note_not_executed", exc)

    def _has_held_actions(self, run_id: str) -> bool:
        for service in {id(s): s for s in (self.policy_service, self.perception_service)}.values():
            held = getattr(service, "has_held_actions", None)
            try:
                if held is not None and held(run_id):
                    return True
            except Exception as exc:  # noqa: BLE001 - only feeds a counter
                self._recovered(run_id, "has_held_actions", exc)
        return False

    def _discard_held_actions(self, run_id: str, reason: str) -> None:
        for service in {id(s): s for s in (self.policy_service, self.perception_service)}.values():
            discard = getattr(service, "discard_held_actions", None)
            if discard is None:
                continue
            try:
                discard(run_id, reason)
            except Exception as exc:  # noqa: BLE001 - never block a resume on it
                self._recovered(run_id, "discard_held_actions", exc)

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
        log_step: bool = True,
        step_ctx: StepContext | None = None,
    ) -> RunResponse:
        """Pause the run, generate an LLM explanation, notify the human, and start the escalation timer.

        ``log_step=False`` when the step was already logged with the action it ran."""
        from operon.agent.hitl import (
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
        if hasattr(state, "pause_reason"):
            # What the person is asked to approve or fix, in the loop's own words.
            state.pause_reason = getattr(decision.action, "text", None) or hitl_message

        if log_step:
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
                    # Nothing ran before the pause, so no after screenshot was taken.
                    after_artifact_path=None,
                    perception_debug=perception_debug,
                    policy_debug=policy_debug,
                    perception=perception,
                    policy_decision=decision,
                    executed_action=executed_action,
                    verification_result=verification,
                    recovery_decision=recovery,
                    decision_source=self._decision_source(decision, policy_debug, perception_debug),
                    **(step_ctx.timing_fields() if step_ctx is not None else {}),
                ),
            )
        state.stop_reason = StopReason.WAITING_FOR_USER
        state.step_count = step_index
        await self.run_store.save_state(state)
        updated = await self.run_store.set_status(record.run_id, RunStatus.WAITING_FOR_USER)
        if updated.status is RunStatus.WAITING_FOR_USER:
            category = self.pending_gate(record.run_id)
            if (
                category is None
                and step_ctx is not None
                and step_ctx.decision.action.action_type is ActionType.WAIT_FOR_USER
                and self._has_held_actions(record.run_id)
            ):
                category = "model_confirmation"
            if category:
                self._resolve_approval_audit(record.run_id, "cancelled")
                event = request_event(record.run_id, step_index, category)
                self._pending_audit[record.run_id] = event
                append_event(self._approval_audit_path(record.run_id), event)

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

    async def _perceive_with_liveness_retry(
        self,
        state,
        initial_frame: CaptureFrame,
        *,
        on_recapture: Callable[[CaptureFrame], None] | None = None,
    ):
        """Perceive with flag-based liveness retries for transient zero-element frames.

        When perception returns is_empty_frame=True (blank/loading screen), re-capture
        and retry up to _LIVENESS_RETRY_MAX times before raising PerceptionLowQualityError.
        Other quality failures (unlabeled elements, etc.) propagate immediately.
        The liveness_retries count is stamped onto the returned ScreenPerception so it
        appears in run.jsonl for observability. ``on_recapture`` gets each new frame,
        so a caller can keep the screenshot the returned perception came from (the
        trust gate reads the click text from it, KAN-163).
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

            # CDP/observable mode: a brand-new tab starts at about:blank with zero
            # elements. Retrying is pointless — nothing navigates the page until the
            # policy acts. Return a blank perception so the policy can issue NAVIGATE.
            if attempt == 1 and hasattr(self.executor, "current_url_for_run"):
                try:
                    current_url = await self.executor.current_url_for_run(state.run_id)
                    if current_url in ("about:blank", "chrome://newtab/", "", None):
                        logger.info(
                            "liveness_retry: page is %r — passing blank perception to policy for navigation",
                            current_url,
                        )
                        return ScreenPerception(
                            summary="",
                            page_hint="unknown",
                            visible_elements=[],
                            capture_artifact_path=frame.artifact_path,
                            confidence=0.0,
                            is_empty_frame=False,
                        )
                except Exception as exc:
                    self._recovered(state.run_id, "liveness_current_url", exc, level=logging.DEBUG)

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
            if on_recapture is not None:
                on_recapture(frame)
        raise PerceptionLowQualityError("no visible elements after all liveness retries")  # pragma: no cover

    async def fail_for_max_steps(self, record: AgentState) -> AgentState:
        """End a run whose step budget is used up: FAILED, cause ``max_steps``, a ``run_end`` row (KAN-175)."""
        record.stop_reason = StopReason.MAX_STEP_LIMIT_REACHED
        message = f"Step limit reached after {record.step_count} steps."
        note_failure(record, cause=RunFailureCause.MAX_STEPS, detail=message)
        append_run_end(
            self._run_log_path(record.run_id),
            record,
            status=RunStatus.FAILED,
            category=FailureCategory.MAX_STEP_LIMIT_REACHED,
            message=message,
        )
        return await self.run_store.set_status(record.run_id, RunStatus.FAILED)

    async def _record_pre_step_perception_failure(
        self,
        *,
        record,
        step_index: int,
        before_artifact_path: str,
        error: Exception,
    ) -> RunResponse:
        # Only what the backend reported for this call: a backend that failed before it
        # finished reports nothing, and the fallback names would be files never written.
        perception_debug = self._latest_debug_artifacts(self.perception_service)
        failure_category = FailureCategory.PRE_STEP_PERCEPTION_FAILED
        stop_reason = StopReason.PRE_STEP_PERCEPTION_FAILED
        if isinstance(error, PerceptionLowQualityError):
            failure_category = error.failure_category
            stop_reason = error.stop_reason
        elif isinstance(error, ModelBlockedError):
            failure_category = FailureCategory.MODEL_BLOCKED
            stop_reason = StopReason.MODEL_BLOCKED

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
                logged_at=datetime.now(UTC),
            ),
        )
        record.artifact_paths.extend(await self._pre_step_failure_files(before_artifact_path, perception_debug))
        record.stop_reason = stop_reason
        # KAN-175: an error in the perceive stage is a perception failure unless it is a
        # model API error or a provider block; the error text goes to state.json.
        cause = cause_of_error(error)
        note_failure(
            record,
            cause=RunFailureCause.PERCEPTION_FAILED if cause is RunFailureCause.RUN_CRASHED else cause,
            detail=error_text(error),
        )
        updated = await self.run_store.set_status(record.run_id, RunStatus.FAILED)
        try:
            await self._cleanup_completed_run(record.run_id)
        except Exception as exc:
            self._recovered(record.run_id, "cleanup_after_pre_step_failure", exc)
        await self._save_efficiency(record)
        return RunResponse(
            run_id=updated.run_id,
            status=updated.status,
            intent=updated.intent,
            step_count=updated.step_count,
        )

    @staticmethod
    async def _pre_step_failure_files(before_artifact_path: str, debug: ModelDebugArtifacts | None) -> list[str]:
        """The files a failed perceive step actually left: the reported debug files and
        whatever the backend wrote in the step folder before it failed, if they exist."""
        await bg_writer.flush()  # the backend's queued writes (its prompt) land first
        named = [before_artifact_path]
        if debug is not None:
            named += [
                path
                for key, path in debug.model_dump().items()
                if key.endswith("_artifact_path") and isinstance(path, str) and path
            ]
        step_dir = Path(before_artifact_path).parent
        if step_dir.is_dir():
            named += sorted(str(path) for path in step_dir.iterdir() if path.is_file())
        files: list[str] = []
        for path in named:
            if path not in files and Path(path).is_file():
                files.append(path)
        return files

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

    async def _apply_recovery_actions(self, state, recovery: RecoveryDecision, step_index: int | None = None) -> None:
        """Execute concrete side effects for CONTEXT_RESET and SESSION_RESET tiers.

        Runs after the step is logged so the log accurately reflects the failure,
        and before set_status so the next step's capture sees the cleaned state.
        Failures are swallowed — a broken reset must never terminate a live run.
        These are actions the model never chose, so the executor never clicks blind, the
        start URL goes through the trust gate, and what was done is written to the step's
        ``recovery_reset.json``.
        """
        strategy = recovery.strategy
        url_allowed: bool | None = None
        if strategy is RecoveryStrategy.CONTEXT_RESET:
            _trace("  7b CONTEXT_RESET", "escape×2 + blank-region click (skipped if none) + scroll-top")
            try:
                await self.executor.context_reset()
            except Exception as exc:
                self._recovered(state.run_id, "context_reset", exc)
        elif strategy is RecoveryStrategy.SESSION_RESET:
            start_url = state.start_url
            if start_url and self.environment is UnifiedEnvironment.BROWSER:
                url_allowed = self._reset_url_allowed(start_url, getattr(state, "permission_mode", None))
                if not url_allowed:
                    logger.info("session_reset: the trust gate does not allow %r; reloading instead", url_origin_for_log(start_url))
                    start_url = None
            _trace("  7b SESSION_RESET", f"origin={url_origin_for_log(start_url)!r}")
            try:
                await self.executor.session_reset(start_url)
            except Exception as exc:
                self._recovered(state.run_id, "session_reset", exc)
        else:
            return
        actions = list(getattr(self.executor, "last_reset_actions", None) or [])
        if url_allowed is False:
            actions.insert(0, "start URL not allowed by the trust gate: did not navigate to it")
        if step_index is not None and actions:
            try:
                path = self._artifacts.persist_recovery_reset(
                    state.run_id, step_index, strategy.value, actions, url_allowed
                )
                state.artifact_paths.append(path)
            except Exception as exc:
                self._recovered(state.run_id, "persist_recovery_reset", exc)

    def _reset_url_allowed(self, url: str, permission_mode: str | None = None) -> bool:
        """True when the agent may navigate to ``url`` unasked: the trust gate allows it and
        the run's permission mode would not hold it for approval (a reset cannot ask a person)."""
        probe = AgentAction(action_type=ActionType.NAVIGATE, url=url)
        result = apply_permission_mode(self._trust_gate.evaluate(probe), probe, permission_mode)
        return result.verdict is TrustVerdict.ALLOW

    def _approval_audit_path(self, run_id: str) -> Path:
        return Path(self._run_log_path(run_id)).with_name("approval_audit.jsonl")

    def _resolve_approval_audit(self, run_id: str, outcome: str) -> None:
        event = self._pending_audit.pop(run_id, None)
        if event is not None:
            append_event(self._approval_audit_path(run_id), {**event, "outcome": outcome})

    async def _cleanup_completed_run(self, run_id: str) -> None:
        self._resolve_approval_audit(run_id, "cancelled")
        self._release_backend_run_state(run_id)
        if hasattr(self.executor, "stop_run_recording"):
            try:
                await self.executor.stop_run_recording(run_id)
            except Exception as exc:
                self._recovered(run_id, "stop_run_recording", exc)
        if not hasattr(self.executor, "aclose_run"):
            await self._persist_cleanup_artifacts(run_id)
            return
        try:
            await self.executor.aclose_run(run_id)
        except Exception as exc:
            self._recovered(run_id, "aclose_run", exc)
        await self._persist_cleanup_artifacts(run_id)

    def _release_backend_run_state(self, run_id: str) -> None:
        """Drop the finished run's conversation and last screenshot from the backends,
        which otherwise keep them for the life of the server."""
        self._expect_reminded.pop(run_id, None)
        self._secret_site_approved.pop(run_id, None)
        self._dialog_pauses.pop(run_id, None)
        self._dialog_approved.pop(run_id, None)
        services = {id(s): s for s in (self.perception_service, self.policy_service)}.values()
        for service in services:
            for target in (service, getattr(service, "delegate", None)):
                release = getattr(target, "release_run", None)
                if release is None:
                    continue
                try:
                    release(run_id)
                except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                    self._recovered(run_id, "release_run", exc)

    async def _persist_cleanup_artifacts(self, run_id: str) -> None:
        if not hasattr(self.executor, "recorded_video_path_for_run"):
            return
        try:
            video_path = self.executor.recorded_video_path_for_run(run_id)
        except Exception as exc:
            self._recovered(run_id, "recorded_video_path", exc)
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
        debug_artifacts = self._latest_debug_artifacts(service)
        if debug_artifacts is not None:
            return debug_artifacts

        # The service wrote no artifacts of its own: name the files in this run's real
        # step directory (where before.png goes), not a path relative to the cwd.
        step_dir = Path(self._before_artifact_path(run_id, step_index)).parent
        return ModelDebugArtifacts(
            prompt_artifact_path=str(step_dir / f"{stage_name}_prompt.txt"),
            raw_response_artifact_path=str(step_dir / f"{stage_name}_raw.txt"),
            parsed_artifact_path=str(step_dir / ("policy_decision.json" if stage_name == "policy" else "verification_result.json" if stage_name == "verification" else "perception_parsed.json")),
            diagnostics_artifact_path=str(step_dir / f"{stage_name}_diagnostics.json") if stage_name in {"perception", "verification"} else None,
        )

    @staticmethod
    def _latest_debug_artifacts(service) -> ModelDebugArtifacts | None:
        """The debug artifacts the service wrote for its latest call, or None."""
        getter = getattr(service, "latest_debug_artifacts", None)
        debug = getter() if callable(getter) else None
        if debug is not None and debug.usage is None and debug.usage_artifact_path:
            # Backends write the usage file only when the call reported usage: don't
            # point the step log at a file that was never written.
            debug = debug.model_copy(update={"usage_artifact_path": None})
        return debug

    def _model_label(self) -> str | None:
        """The name of the model behind the policy, when the policy or its backend says."""
        for candidate in (self.policy_service, getattr(self.policy_service, "delegate", None)):
            label = getattr(candidate, "model_label", None)
            if isinstance(label, str) and label:
                return label
        return None

    def _decision_source(
        self,
        decision: PolicyDecision,
        policy_debug: ModelDebugArtifacts | None,
        perception_debug: ModelDebugArtifacts | None = None,
    ) -> str:
        """Who chose the step's action: ``[RULE] name`` or ``[LLM] provider:model``.

        The model comes from the call's reported usage, else the backend's own label;
        ``[LLM] unknown`` when neither says. Readers match the ``[LLM] `` prefix."""
        if decision.rule_name:
            return f"[RULE] {decision.rule_name}"
        for debug in (policy_debug, perception_debug):
            usage = debug.usage if debug is not None else None
            if usage is not None:
                return f"[LLM] {usage.provider}:{usage.model}"
        label = self._model_label()
        return f"[LLM] {label}" if label else "[LLM] unknown"

    async def _execute_action(
        self,
        *,
        record,
        state,
        decision_action,
        step_index: int,
        use_unified_executor: bool = False,
    ):
        """Run the model's action once, exactly as chosen."""
        executor_runner = self.executor_adapter if use_unified_executor else self.executor
        executed_action = await executor_runner.execute(decision_action)
        executed_action = self._apply_no_progress_detection(state, executed_action)
        return state, self._persist_execution_trace(record.run_id, step_index, executed_action)

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
    def _with_capture_origin(perception, frame):
        """Stamp the captured monitor's virtual-desktop origin on the perception.

        The desktop executor captures the foreground monitor, so the model's
        coordinates are relative to that monitor's image; without its origin a click
        on a secondary monitor lands on the primary one."""
        origin = (int(getattr(frame, "monitor_left", 0) or 0), int(getattr(frame, "monitor_top", 0) or 0))
        current = tuple(getattr(perception, "monitor_origin", (0, 0)) or (0, 0))
        if origin == (0, 0) or current != (0, 0) or not hasattr(perception, "model_copy"):
            return perception
        return perception.model_copy(update={"monitor_origin": origin})

    @staticmethod
    def _apply_monitor_origin(action: AgentAction, perception) -> AgentAction:
        """Translate monitor-local coords to virtual-desktop coords by adding the
        monitor origin stored in perception. No-op when origin is (0, 0).

        Covers x/y (start point), x_end/y_end (drag/screenshot_region end point) and
        each child of a BATCH.
        """
        origin = getattr(perception, "monitor_origin", (0, 0))
        ox, oy = origin
        if ox == 0 and oy == 0:
            return action
        updates: dict = {}
        if action.action_type is ActionType.BATCH and action.actions:
            updates["actions"] = [AgentLoop._apply_monitor_origin(child, perception) for child in action.actions]
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
            ActionType.UPLOAD_FILE,
            ActionType.UPLOAD_FILE_NATIVE,
            ActionType.TYPE,
        }:
            center_x, center_y = element_center(target)
            return action.model_copy(update={"x": center_x, "y": center_y})
        return action

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
            expected_section=None,
        )

    def _persist_execution_trace(self, run_id: str, step_index: int, executed_action):
        return self._artifacts.persist_execution_trace(run_id, step_index, executed_action)

    def _persist_progress_trace(self, run_id: str, step_index: int, progress_trace: ProgressTrace) -> str:
        return self._artifacts.persist_progress_trace(run_id, step_index, progress_trace)

    def _persist_step_timing(self, run_id: str, step_index: int, stage_timings: dict[str, float]) -> None:
        self._artifacts.persist_step_timing(run_id, step_index, stage_timings)

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
    # Progress-tracking and signatures (delegated to operon.agent.policy.progress_tracker)
    # ------------------------------------------------------------------

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

    def _mark_completed_progress(self, state, decision, executed_action, verification) -> None:
        self._progress._mark_completed_progress(state, decision, executed_action, verification)

    @classmethod
    def _page_signature(cls, perception) -> str:
        return page_signature(perception)

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
