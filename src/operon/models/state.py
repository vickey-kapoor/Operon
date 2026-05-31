"""Schemas for persisted in-process agent state."""

from __future__ import annotations

from pydantic import Field

from operon.models.common import RunStatus, StopReason, StrictModel
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception
from operon.models.progress import ProgressState
from operon.models.verification import VerificationResult


class AgentState(StrictModel):
    """Local typed state stored for a single agent run."""

    run_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    start_url: str | None = Field(default=None, min_length=1)
    headless: bool | None = None
    benchmark: str | None = Field(default=None, min_length=1)
    status: RunStatus
    current_subgoal: str | None = Field(default=None, min_length=1)
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=25, ge=1, le=200)
    observation_history: list[ScreenPerception] = Field(default_factory=list)
    action_history: list[ExecutedAction] = Field(default_factory=list)
    verification_history: list[VerificationResult] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    target_failure_counts: dict[str, int] = Field(default_factory=dict)
    progress_state: ProgressState = Field(default_factory=ProgressState)
    artifact_paths: list[str] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    hitl_message: str | None = Field(default=None, description="LLM-generated message shown to the human when a run pauses for intervention")
    force_fresh_perception: bool = Field(default=False, description="When True the loop waits an extra settle delay before the next capture, then resets. Set by the no-progress recovery rule after a visual perturbation.")
    # Carries the outcome of the most recent deterministic rule that fired.
    # Injected into the next LLM prompt so the planner knows what the rule tried and whether it worked.
    last_rule_trace: str | None = None
    # Set to True by the loop when a native desktop menu was explicitly opened
    # (e.g. after a successful right-click or menu-bar click). Guards the
    # _dropdown_menu_select_rule from misfiring on desktop app toolbars.
    menu_is_active: bool = False
    # Signals whether this run originated from the Command Center frontend
    # ("observable") or was launched programmatically/via benchmark ("batch").
    # The frontend uses this to decide whether to auto-attach the live CDP view.
    # Not related to whether the debug port is open — port 9222 is always
    # exposed for all local browser runs regardless of this value.
    mode: str = Field(default="batch", pattern=r"^(observable|batch)$")
