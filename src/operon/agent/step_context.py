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

from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.recovery import RecoveryDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationResult


@dataclass
class StepContext:
    """State of one step while it moves through the loop's stages."""

    # Set when the step begins.
    record: AgentState
    step_index: int
    before_artifact_path: str
    after_artifact_path: str
    # Per-stage durations in milliseconds, written to step_timing.json at the end.
    stage_timings: dict[str, float] = field(default_factory=dict)

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

    # Execute / verify, including in-step adaptation retries.
    executed_action: ExecutedAction | None = None
    earlier_attempts: list[ExecutedAction] = field(default_factory=list)
    adaptation_trace: list[str] = field(default_factory=list)
    deferred_pause: PolicyDecision | None = None
    retries_used: int = 0
    attempt_index: int = 1
    verification: VerificationResult = field(init=False)
    verification_debug: ModelDebugArtifacts = field(init=False)
    screen_change_ratio: float | None = None

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

    @property
    def last_attempt(self) -> ExecutedAction:
        """The attempt the step ends with; set once the first attempt ran."""
        if self.executed_action is None:
            raise AttributeError("no attempt has run in this step yet")
        return self.executed_action
