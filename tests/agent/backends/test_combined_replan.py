"""Re-plan path for CombinedPerceptionPolicyService.

When PolicyCoordinator triggers a re-plan (e.g., _reject_premature_stop) it
calls choose_action() a second time within the same step. The combined service
caches the decision in perceive() and pops it on first choose_action(); a
second call hits an empty cache and historically raised PolicyError, crashing
the run mid-suite. The fix re-issues a fresh combined call against the same
screenshot so the LLM can incorporate the just-appended advisory hint.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.policy import PolicyError
from operon.models.policy import ActionType, AgentAction, PolicyDecision


def _make_service(prompt_template: str = "intent={intent} subgoal={current_subgoal} step={step_count} prev={previous_summary} retries={retry_counts} hints={advisory_hints}") -> tuple[CombinedPerceptionPolicyService, MagicMock]:
    gemini = MagicMock()
    gemini.generate_perception = AsyncMock()
    gemini.latest_perception_scale_ratio.return_value = 1.0
    svc = CombinedPerceptionPolicyService.__new__(CombinedPerceptionPolicyService)
    svc.gemini_client = gemini
    svc.prompt_path = Path("test_prompt.txt")
    svc._prompt_template = prompt_template
    svc._cached_decision = {}
    svc._last_debug_artifacts = None
    svc._advisory_hints = {}
    svc._perception_only = False
    svc._image_size = {}
    svc._replans = {}
    return svc, gemini


def _make_state(run_id: str = "abc123") -> MagicMock:
    state = MagicMock()
    state.run_id = run_id
    state.intent = "do the thing"
    state.current_subgoal = None
    state.step_count = 1
    state.observation_history = []
    state.action_history = []
    state.retry_counts = {}
    return state


# Re-plans save their artifacts next to the screenshot: keep them out of the repo.
_STEP_DIR = Path(tempfile.mkdtemp(prefix="combined-replan-")) / "step_1"


def _make_perception(screenshot_path: str = str(_STEP_DIR / "before.png")) -> MagicMock:
    p = MagicMock()
    p.capture_artifact_path = screenshot_path
    return p


def _click_decision(rationale: str = "click submit") -> PolicyDecision:
    return PolicyDecision(
        action=AgentAction(action_type=ActionType.CLICK, target_element_id="submit", x=10, y=20),
        rationale=rationale,
        confidence=0.9,
        active_subgoal="step",
    )


@pytest.mark.asyncio
async def test_choose_action_returns_cached_decision_on_first_call() -> None:
    svc, _ = _make_service()
    state = _make_state()
    cached = _click_decision()
    svc._cached_decision[state.run_id] = cached

    out = await svc.choose_action(state, _make_perception())

    assert out is cached
    assert state.run_id not in svc._cached_decision  # popped


@pytest.mark.asyncio
async def test_choose_action_replans_via_fresh_combined_call_when_cache_empty() -> None:
    """Re-plan path: cache empty (consumed by an earlier choose_action this step).

    The service must re-issue a combined call against the same screenshot and
    return the freshly-parsed decision rather than raising PolicyError. Any
    advisory hint the coordinator just appended (e.g., the premature-stop
    correction) is rendered into the new prompt automatically.
    """
    svc, gemini = _make_service()
    state = _make_state()
    perception = _make_perception(screenshot_path=str(_STEP_DIR / "before.png"))

    # Simulate the coordinator adding a correction hint between the original
    # choose_action and the re-plan call.
    svc.add_advisory_hints(["CORRECTION: don't STOP yet"], source="validation", run_id=state.run_id)

    # The fresh combined call returns a non-STOP action.
    raw = json.dumps({
        "perception": {
            "page_hint": "form_page",
            "visible_elements": [],
            "capture_artifact_path": str(_STEP_DIR / "before.png"),
        },
        "action": {
            "action_type": "click",
            "target_element_id": "submit",
            "x": 100,
            "y": 200,
        },
        "rationale": "post-correction click",
        "confidence": 0.9,
        "active_subgoal": "submit form",
    })
    gemini.generate_perception.return_value = raw

    out = await svc.choose_action(state, perception)

    # A fresh combined call was made
    assert gemini.generate_perception.await_count == 1
    sent_prompt, sent_screenshot = gemini.generate_perception.call_args.args
    assert sent_screenshot == str(_STEP_DIR / "before.png")
    # Advisory hint was rendered into the prompt
    assert "CORRECTION: don't STOP yet" in sent_prompt
    # The re-planned decision came back parsed
    assert out.action.action_type is ActionType.CLICK
    assert out.action.target_element_id == "submit"
    assert out.rationale == "post-correction click"


@pytest.mark.asyncio
async def test_choose_action_raises_when_cache_empty_and_no_screenshot_path() -> None:
    """Defensive path: no cached decision AND no screenshot to re-plan from."""
    svc, _ = _make_service()
    state = _make_state()
    perception = MagicMock()
    perception.capture_artifact_path = ""  # falsy

    with pytest.raises(PolicyError, match="no screenshot path"):
        await svc.choose_action(state, perception)


@pytest.mark.asyncio
async def test_a_replan_records_its_own_usage(tmp_path: Path) -> None:
    from operon.models.usage import ModelUsage
    from operon.store.background_writer import bg_writer

    svc, gemini = _make_service()
    usage = ModelUsage(provider="gemini", model="gemini-test", request_kind="perception", input_tokens=10, output_tokens=5, total_tokens=15)
    gemini.latest_usage.return_value = usage
    gemini.generate_perception.side_effect = [
        json.dumps({
            "perception": {"summary": "s", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
            "action": {"action_type": "wait", "wait_ms": 500},
            "rationale": "r", "confidence": 0.5, "active_subgoal": "s",
        })
    ] * 2
    step_dir = tmp_path / "step_3"
    step_dir.mkdir()

    await svc.choose_action(_make_state(), _make_perception(str(step_dir / "before.png")))
    await svc.choose_action(_make_state(), _make_perception(str(step_dir / "before.png")))
    await bg_writer.flush()

    debug = svc.latest_debug_artifacts()
    assert debug.usage == usage
    assert debug.usage_artifact_path == str(step_dir / "combined_replan_2_usage.json")
    assert sorted(p.name for p in step_dir.glob("*_usage.json")) == ["combined_replan_2_usage.json", "combined_replan_usage.json"]
