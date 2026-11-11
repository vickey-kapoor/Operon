"""A real AgentLoop over a file-backed run store, with scripted perception/policy and
a recording executor: enough to drive the front door (KAN-89) offline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.trust_gate import TrustGate
from operon.models.capture import CaptureFrame
from operon.models.execution import ExecutedAction
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.verification import VerificationResult, VerificationStatus
from operon.store.run_store import FileBackedRunStore

CLICK_SAVE = AgentAction(action_type=ActionType.CLICK, x=10, y=20, target_element_id="save")
PURCHASE = AgentAction(action_type=ActionType.TYPE, text="purchase now", target_element_id="buy")


def _debug() -> ModelDebugArtifacts:
    return ModelDebugArtifacts(prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="d")


class ScriptedPolicy:
    """Returns the scripted actions in order, then repeats the last one."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self.actions = list(actions)
        self.calls = 0

    async def choose_action(self, *_args, **_kwargs) -> PolicyDecision:
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return PolicyDecision(action=action, rationale=f"because {action.action_type.value}", confidence=0.9, active_subgoal="s")

    def latest_debug_artifacts(self) -> ModelDebugArtifacts:
        return _debug()


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[AgentAction] = []
        self.user_inputs: list[dict] = []
        self.viewport_size = (100, 100)

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.executed.append(action)
        return ExecutedAction(action=action, success=True, detail="ok")

    async def user_input(self, run_id: str, **kwargs) -> bool:
        self.user_inputs.append({"run_id": run_id, **kwargs})
        return True


def build_loop(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    actions: list[AgentAction],
    *,
    gate: TrustGate | None = None,
) -> tuple[AgentLoop, RecordingExecutor]:
    from operon.agent import control, hitl

    monkeypatch.setattr(hitl, "notify_desktop", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "post_hitl_webhook", AsyncMock())
    monkeypatch.setattr(hitl, "start_escalation_timer", AsyncMock())
    monkeypatch.setattr(control, "_confidence_threshold", 0.0)

    executor = RecordingExecutor()
    frame = CaptureFrame(artifact_path=str(root / "f.png"), width=10, height=10, mime_type="image/png")
    loop = AgentLoop(
        capture_service=SimpleNamespace(capture=AsyncMock(return_value=frame)),
        perception_service=SimpleNamespace(
            perceive=AsyncMock(
                return_value=ScreenPerception(
                    summary="s", page_hint="unknown", capture_artifact_path=str(root / "f.png"), visible_elements=[]
                )
            ),
            latest_debug_artifacts=_debug,
        ),
        run_store=FileBackedRunStore(root),
        policy_service=ScriptedPolicy(actions),
        executor=executor,
        verifier_service=SimpleNamespace(
            verify=AsyncMock(
                return_value=VerificationResult(
                    status=VerificationStatus.UNCERTAIN, expected_outcome_met=False, stop_condition_met=False, reason="c"
                )
            )
        ),
        recovery_manager=SimpleNamespace(
            recover=AsyncMock(
                return_value=RecoveryDecision(strategy=RecoveryStrategy.WAIT_AND_RETRY, message="m", retry_after_ms=10)
            )
        ),
    )
    loop._trust_gate = gate if gate is not None else TrustGate(enabled=True, confirm=("purchase",))
    return loop, executor
