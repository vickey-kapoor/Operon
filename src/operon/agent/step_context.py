"""Typed per-step context passed between the stages of one agent-loop step (KAN-155).

``AgentLoop._step_run_locked`` runs a step as a sequence of stage methods
(capture, perceive, decide, gate, execute/verify, recover, record, finish).
Everything one stage hands to a later one lives here instead of in locals of
one long function, so a stage can be read, tested and changed on its own.

Fields set by a stage are declared ``field(init=False)``: reading one before
the stage that sets it ran raises ``AttributeError`` (a loop bug), and the
type stays non-optional for the stages after it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypedDict

from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.recovery import RecoveryDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationResult


class StepTiming(TypedDict):
    """The timing fields of one step's log record (``StepLog``).

    ``model_latency_ms`` is the time spent in the perceive and policy stages (the stages that
    wait on the model, local work in them included), capped at ``step_latency_ms``."""

    started_at: datetime
    finished_at: datetime
    step_latency_ms: float
    model_latency_ms: float


@dataclass
class StepContext:
    """State of one step while it moves through the loop's stages."""

    # Set when the step begins.
    record: AgentState
    step_index: int
    before_artifact_path: str
    after_artifact_path: str
    # True once an executor really wrote an after screenshot; the log names the path only then.
    after_captured: bool = False
    # Per-stage durations in milliseconds, written to step_timing.json at the end.
    stage_timings: dict[str, float] = field(default_factory=dict)
    # When the step began: wall-clock UTC for the log, and a monotonic clock for its duration.
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_perf: float = field(default_factory=time.perf_counter)

    # Capture and the user's correction (consumed before perceive).
    frame: CaptureFrame = field(init=False)
    override_text: str | None = None
    override_sent: bool = False

    # Perceive.
    perception: ScreenPerception = field(init=False)
    perception_debug: ModelDebugArtifacts = field(init=False)
    state: AgentState = field(init=False)

    # Decide.
    decision: PolicyDecision = field(init=False)
    policy_debug: ModelDebugArtifacts = field(init=False)

    # Gates a person approved for this action, captured before the gate stage uses the approval up (KAN-207).
    approved_gates: frozenset[str] = frozenset()

    # Fresh look (KAN-207), taken in the gate stage: whether a screenshot was grabbed, and the failed
    # result that replaces the action when its target spot changed.
    fresh_look_ran: bool = False
    fresh_look_stale: ExecutedAction | None = None

    # Execute / verify.
    executed_action: ExecutedAction | None = None
    verification: VerificationResult = field(init=False)
    # None when the verifier makes no model call (OwnDoneVerifier).
    verification_debug: ModelDebugArtifacts | None = None
    screen_change_ratio: float | None = None
    # A JS dialog was auto-accepted this step: the screen change says nothing about the
    # model's expectation then (KAN-116).
    dialog_captured: bool = False
    # A confirm/prompt/beforeunload dialog the executor cancelled this step: (type, message, batch child index).
    # The loop pauses for the person at the end of the step (KAN-172).
    cancelled_dialog: tuple[str, str, int | None] | None = None

    # Recover and finish.
    recovery: RecoveryDecision = field(init=False)
    progress_trace_artifact_path: str = field(init=False)
    final_status: RunStatus = RunStatus.RUNNING

    @property
    def run_id(self) -> str:
        return self.record.run_id

    def record_stage(self, name: str, t0: float) -> float:
        """Add the time since ``t0`` to stage ``name``; return it in seconds."""
        elapsed = time.perf_counter() - t0
        self.stage_timings[name] = self.stage_timings.get(name, 0.0) + elapsed * 1000.0
        return elapsed

    # Stages that wait on the model; their sum is the step's model latency.
    MODEL_STAGES = ("perceive", "policy")

    def timing_fields(self) -> StepTiming:
        """Start/end time, step latency and model latency (the perceive + policy stages) for this step's log record."""
        step_latency_ms = (time.perf_counter() - self.started_perf) * 1000.0
        model_latency_ms = sum(self.stage_timings.get(name, 0.0) for name in self.MODEL_STAGES)
        return {
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC),
            "step_latency_ms": round(step_latency_ms, 3),
            "model_latency_ms": round(min(model_latency_ms, step_latency_ms), 3),
        }

    @property
    def last_attempt(self) -> ExecutedAction:
        """The action the step ran; set once it ran."""
        if self.executed_action is None:
            raise AttributeError("no attempt has run in this step yet")
        return self.executed_action
