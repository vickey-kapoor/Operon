"""Unit tests for the episodic task memory feature."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agent.reflector import PostRunReflector
from src.models.common import RunStatus
from src.models.episode import Episode, EpisodeReplayState, EpisodeStep
from src.models.perception import PageHint, ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.store.memory import FileBackedMemoryStore, normalize_intent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_index: int,
    page_hint: str,
    action_type: str,
    target_id: str | None = None,
    target_name: str | None = None,
    text: str | None = None,
    key: str | None = None,
    subgoal: str = "test subgoal",
    failed: bool = False,
) -> dict:
    """Build a step dict matching the run.jsonl structure."""
    visible_elements = []
    if target_id and target_name:
        visible_elements.append({
            "element_id": target_id,
            "primary_name": target_name,
            "element_type": "input",
        })

    return {
        "step_index": step_index,
        "perception": {
            "page_hint": page_hint,
            "visible_elements": visible_elements,
        },
        "policy_decision": {
            "action": {
                "action_type": action_type,
                "target_element_id": target_id,
                "text": text,
                "key": key,
            },
            "active_subgoal": subgoal,
        },
        "executed_action": {
            "failure_category": "execution_target_not_found" if failed else None,
        },
    }


def _make_episode(
    *,
    normalized_intent: str,
    benchmark: str,
    episode_id: str = "run-test",
    success_count: int = 1,
    page_hint: PageHint = PageHint.FORM_PAGE,
) -> Episode:
    """Build a minimal valid Episode."""
    return Episode(
        episode_id=episode_id,
        normalized_intent=normalized_intent,
        benchmark=benchmark,
        source_run_id=episode_id,
        steps=[
            EpisodeStep(
                step_index=1,
                page_hint=page_hint,
                action_type=ActionType.CLICK,
                target_description="Submit button",
                subgoal="submit the form",
            ),
            EpisodeStep(
                step_index=2,
                page_hint=page_hint,
                action_type=ActionType.TYPE,
                text="hello",
                subgoal="fill the name field",
            ),
        ],
        success_count=success_count,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_perception(page_hint: PageHint = PageHint.FORM_PAGE) -> ScreenPerception:
    return ScreenPerception(
        summary="A form is visible.",
        page_hint=page_hint,
        visible_elements=[],
        capture_artifact_path="runs/test/step_1/before.png",
        confidence=0.9,
    )


def _make_state(intent: str = "fill out the contact form", step_count: int = 1) -> AgentState:
    return AgentState(
        run_id="run-test",
        intent=intent,
        status=RunStatus.RUNNING,
        step_count=step_count,
    )


# ---------------------------------------------------------------------------
# 1. normalize_intent
# ---------------------------------------------------------------------------


def test_normalize_intent_lowercases() -> None:
    assert normalize_intent("Fill Out The Form") == "fill out the form"


def test_normalize_intent_strips_leading_and_trailing_whitespace() -> None:
    assert normalize_intent("  fill the form  ") == "fill the form"


def test_normalize_intent_collapses_internal_spaces() -> None:
    assert normalize_intent("fill   out   the   form") == "fill out the form"


@pytest.mark.parametrize("punct", [".", ",", ";", ":", "!", "?"])
def test_normalize_intent_removes_trailing_punctuation(punct: str) -> None:
    result = normalize_intent(f"fill the form{punct}")
    assert result == "fill the form"


def test_normalize_intent_does_not_strip_internal_punctuation() -> None:
    result = normalize_intent("fill the form, please")
    assert result == "fill the form, please"


def test_normalize_intent_handles_already_normalized_string() -> None:
    assert normalize_intent("fill the form") == "fill the form"


def test_normalize_intent_handles_empty_string() -> None:
    assert normalize_intent("") == ""


def test_normalize_intent_combined_operations() -> None:
    # lowercase + strip outer whitespace + collapse internal spaces + strip trailing punct
    result = normalize_intent("  FILL   OUT  the   Form!  ")
    assert result == "fill out the form"


# ---------------------------------------------------------------------------
# 2. Episode extraction (_extract_episode)
# ---------------------------------------------------------------------------


def test_extract_episode_returns_episode_for_two_qualifying_steps() -> None:
    steps = [
        _make_step(1, "form_page", "click", target_id="el_1", target_name="Name input"),
        _make_step(2, "form_page", "type", text="Alice"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert len(episode.steps) == 2


def test_extract_episode_returns_none_when_fewer_than_two_qualifying_steps() -> None:
    steps = [
        _make_step(1, "form_page", "click"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is None


def test_extract_episode_returns_none_when_no_steps() -> None:
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", [])
    assert episode is None


def test_extract_episode_skips_stop_actions() -> None:
    steps = [
        _make_step(1, "form_page", "click"),
        _make_step(2, "form_page", "stop"),
        _make_step(3, "form_page", "type", text="done"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    action_types = [s.action_type for s in episode.steps]
    assert ActionType.STOP not in action_types
    assert len(episode.steps) == 2


def test_extract_episode_skips_wait_actions() -> None:
    steps = [
        _make_step(1, "form_page", "click"),
        _make_step(2, "form_page", "wait"),
        _make_step(3, "form_page", "type", text="done"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    action_types = [s.action_type for s in episode.steps]
    assert ActionType.WAIT not in action_types


def test_extract_episode_skips_failed_steps() -> None:
    steps = [
        _make_step(1, "form_page", "click", failed=True),
        _make_step(2, "form_page", "type", text="Alice"),
        _make_step(3, "form_page", "click"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert len(episode.steps) == 2
    # The failed click at step 1 must not appear
    assert episode.steps[0].step_index == 2


def test_extract_episode_returns_none_when_all_steps_fail_or_are_skipped() -> None:
    steps = [
        _make_step(1, "form_page", "stop"),
        _make_step(2, "form_page", "click", failed=True),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is None


def test_extract_episode_resolves_target_description_from_visible_elements() -> None:
    steps = [
        _make_step(1, "form_page", "click", target_id="el_name", target_name="Name Input"),
        _make_step(2, "form_page", "type", text="Alice"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert episode.steps[0].target_description == "Name Input"


def test_extract_episode_target_description_is_none_when_element_not_found() -> None:
    steps = [
        _make_step(1, "form_page", "click", target_id="el_missing"),
        _make_step(2, "form_page", "type", text="Alice"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert episode.steps[0].target_description is None


def test_extract_episode_normalizes_intent_on_episode() -> None:
    steps = [
        _make_step(1, "form_page", "click"),
        _make_step(2, "form_page", "type", text="Alice"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "  Fill The Form! ", "auth_free_form", steps)
    assert episode is not None
    assert episode.normalized_intent == "fill the form"


def test_extract_episode_sets_correct_benchmark_and_run_id() -> None:
    steps = [
        _make_step(1, "form_page", "click"),
        _make_step(2, "form_page", "type", text="Alice"),
    ]
    episode = PostRunReflector._extract_episode("run-42", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert episode.source_run_id == "run-42"
    assert episode.benchmark == "auth_free_form"


def test_extract_episode_preserves_text_and_key_fields() -> None:
    steps = [
        _make_step(1, "form_page", "type", text="Alice"),
        _make_step(2, "form_page", "press_key", key="Tab"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill the form", "auth_free_form", steps)
    assert episode is not None
    assert episode.steps[0].text == "Alice"
    assert episode.steps[1].key == "Tab"


# ---------------------------------------------------------------------------
# 3. Episode storage (FileBackedMemoryStore)
# ---------------------------------------------------------------------------


def test_save_and_get_episode_exact_match(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the contact form", benchmark="auth_free_form")
    store.save_episode(ep)

    result = store.get_episode("fill the contact form", "auth_free_form")
    assert result is not None
    assert result.normalized_intent == "fill the contact form"
    assert result.benchmark == "auth_free_form"


def test_get_episode_returns_none_for_no_match(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the contact form", benchmark="auth_free_form")
    store.save_episode(ep)

    result = store.get_episode("send an email draft", "auth_free_form")
    assert result is None


def test_get_episode_returns_none_for_wrong_benchmark(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the contact form", benchmark="auth_free_form")
    store.save_episode(ep)

    result = store.get_episode("fill the contact form", "gmail_draft_authenticated")
    assert result is None


def test_get_episode_containment_match_when_no_exact(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(
        normalized_intent="fill the contact form on the practice site",
        benchmark="auth_free_form",
    )
    store.save_episode(ep)

    # "fill the contact form" is a substring of the stored intent
    result = store.get_episode("fill the contact form", "auth_free_form")
    assert result is not None
    assert "fill the contact form" in result.normalized_intent


def test_get_episode_exact_match_preferred_over_containment(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    exact = _make_episode(
        normalized_intent="fill the form",
        benchmark="auth_free_form",
        episode_id="run-exact",
    )
    broader = _make_episode(
        normalized_intent="fill the form and submit",
        benchmark="auth_free_form",
        episode_id="run-broader",
    )
    store.save_episode(exact)
    store.save_episode(broader)

    result = store.get_episode("fill the form", "auth_free_form")
    assert result is not None
    assert result.episode_id == "run-exact"


def test_save_episode_increments_success_count_on_same_intent_and_benchmark(
    tmp_path: pytest.TempPathFactory,
) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the form", benchmark="auth_free_form")
    store.save_episode(ep)
    store.save_episode(ep)

    result = store.get_episode("fill the form", "auth_free_form")
    assert result is not None
    assert result.success_count == 2


def test_save_episode_does_not_increment_across_different_benchmarks(
    tmp_path: pytest.TempPathFactory,
) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep_form = _make_episode(normalized_intent="fill the form", benchmark="auth_free_form")
    ep_gmail = _make_episode(normalized_intent="fill the form", benchmark="gmail_draft_authenticated")
    store.save_episode(ep_form)
    store.save_episode(ep_gmail)

    result = store.get_episode("fill the form", "auth_free_form")
    assert result is not None
    assert result.success_count == 1


def test_episode_store_is_isolated_per_tmp_path(
    tmp_path: pytest.TempPathFactory,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    store_a = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the form", benchmark="auth_free_form")
    store_a.save_episode(ep)

    store_b = FileBackedMemoryStore(root_dir=tmp_path_factory.mktemp("other"))
    result = store_b.get_episode("fill the form", "auth_free_form")
    assert result is None


def test_episode_round_trip_preserves_steps(tmp_path: pytest.TempPathFactory) -> None:
    store = FileBackedMemoryStore(root_dir=tmp_path)
    ep = _make_episode(normalized_intent="fill the form", benchmark="auth_free_form")
    store.save_episode(ep)

    result = store.get_episode("fill the form", "auth_free_form")
    assert result is not None
    assert len(result.steps) == len(ep.steps)
    assert result.steps[0].action_type == ActionType.CLICK
    assert result.steps[1].action_type == ActionType.TYPE


# ---------------------------------------------------------------------------
# 4. EpisodeReplayState deviation logic
# ---------------------------------------------------------------------------


def test_episode_replay_state_defaults() -> None:
    state = EpisodeReplayState(episode_id="ep-1")
    assert state.active is True
    assert state.deviations == 0
    assert state.max_deviations == 2
    assert state.current_step_index == 0


def test_episode_replay_state_active_remains_true_below_max_deviations() -> None:
    state = EpisodeReplayState(episode_id="ep-1", deviations=1, max_deviations=2)
    assert state.active is True


def test_episode_replay_state_can_be_deactivated() -> None:
    state = EpisodeReplayState(episode_id="ep-1")
    state.active = False
    assert state.active is False


def test_episode_replay_deactivates_after_max_deviations_page_hint_mismatches(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """
    Verify that after max_deviations page_hint mismatches the PolicyCoordinator
    sets replay_state.active = False.
    """
    from src.agent.policy_coordinator import PolicyCoordinator

    episode = _make_episode(
        normalized_intent="fill the form",
        benchmark="auth_free_form",
        page_hint=PageHint.FORM_PAGE,
    )

    mock_store = MagicMock()
    mock_store.get_hints.return_value = []
    mock_store.get_episode.return_value = episode

    mock_delegate = MagicMock()
    mock_delegate._reset_advisory_hints_for_test = MagicMock()
    mock_delegate._advisory_hints = []

    coordinator = PolicyCoordinator(delegate=mock_delegate, memory_store=mock_store)
    state = _make_state(step_count=1)

    # First call triggers episode load; page_hint differs from episode (GMAIL_INBOX vs FORM_PAGE)
    wrong_hint_perception = _make_perception(page_hint=PageHint("gmail_inbox"))
    coordinator._try_episode_hint(state, wrong_hint_perception)
    assert coordinator._replay_state is not None
    assert coordinator._replay_state.deviations == 1
    assert coordinator._replay_state.active is True

    # Second mismatch should reach max_deviations and deactivate
    coordinator._replay_state.current_step_index = 0  # reset so we still look at step 0
    coordinator._try_episode_hint(state, wrong_hint_perception)
    assert coordinator._replay_state.active is False


# ---------------------------------------------------------------------------
# 5. Episode hint injection in PolicyCoordinator
# ---------------------------------------------------------------------------


class _MockPolicyDelegate:
    """Minimal PolicyService stand-in that records advisory hints and returns a fixed decision."""

    def __init__(self) -> None:
        self.advisory_hints_received: list[list[str]] = []
        self._advisory_hints: list[tuple[str, str]] = []

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        self._advisory_hints = list(hints)
        self.advisory_hints_received.append(list(hints))

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self._advisory_hints.extend(h for h in hints if h)
        self.advisory_hints_received.append(list(hints))

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        self._advisory_hints = []  # real delegates clear hints after consuming them
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.CLICK, target_element_id="el_1"),
            rationale="delegate chose click",
            confidence=0.9,
            active_subgoal="test",
        )

    def latest_debug_artifacts(self) -> None:
        return None


@pytest.mark.asyncio
async def test_policy_coordinator_injects_episode_hint_when_episode_matches() -> None:
    episode = _make_episode(
        normalized_intent="fill the form",
        benchmark="auth_free_form",
        page_hint=PageHint.FORM_PAGE,
    )

    mock_store = MagicMock()
    mock_store.get_hints.return_value = []
    mock_store.get_episode.return_value = episode

    delegate = _MockPolicyDelegate()
    from src.agent.policy_coordinator import PolicyCoordinator

    coordinator = PolicyCoordinator(delegate=delegate, memory_store=mock_store)
    state = _make_state(intent="fill the form", step_count=1)
    perception = _make_perception(page_hint=PageHint.FORM_PAGE)

    await coordinator.choose_action(state, perception)

    # Delegate must have received at least one add_advisory_hints call containing an episode hint
    all_hints = [h for call in delegate.advisory_hints_received for h in call]
    episode_hints = [h for h in all_hints if "Episode replay" in h]
    assert episode_hints, "Expected at least one episode replay advisory hint to be injected"


@pytest.mark.asyncio
async def test_policy_coordinator_does_not_inject_episode_hint_when_no_match() -> None:
    mock_store = MagicMock()
    mock_store.get_hints.return_value = []
    mock_store.get_episode.return_value = None

    delegate = _MockPolicyDelegate()
    from src.agent.policy_coordinator import PolicyCoordinator

    coordinator = PolicyCoordinator(delegate=delegate, memory_store=mock_store)
    state = _make_state(intent="fill the form", step_count=1)
    perception = _make_perception(page_hint=PageHint.FORM_PAGE)

    await coordinator.choose_action(state, perception)

    all_hints = [h for call in delegate.advisory_hints_received for h in call]
    episode_hints = [h for h in all_hints if "Episode replay" in h]
    assert not episode_hints


@pytest.mark.asyncio
async def test_policy_coordinator_stops_episode_hints_after_episode_exhausted() -> None:
    """After all episode steps have been consumed, no more episode hints are injected."""
    episode = _make_episode(
        normalized_intent="fill the form",
        benchmark="auth_free_form",
        page_hint=PageHint.FORM_PAGE,
    )
    # Episode has 2 steps

    mock_store = MagicMock()
    mock_store.get_hints.return_value = []
    mock_store.get_episode.return_value = episode

    delegate = _MockPolicyDelegate()
    from src.agent.policy_coordinator import PolicyCoordinator

    coordinator = PolicyCoordinator(delegate=delegate, memory_store=mock_store)
    perception = _make_perception(page_hint=PageHint.FORM_PAGE)

    # Step 1 + 2 consume the episode
    await coordinator.choose_action(_make_state(step_count=1), perception)
    await coordinator.choose_action(_make_state(step_count=2), perception)

    # Clear recorded calls so we can check step 3 in isolation
    delegate.advisory_hints_received.clear()
    await coordinator.choose_action(_make_state(step_count=3), perception)

    all_hints = [h for call in delegate.advisory_hints_received for h in call]
    episode_hints = [h for h in all_hints if "Episode replay" in h]
    assert not episode_hints


@pytest.mark.asyncio
async def test_policy_coordinator_deactivates_episode_after_max_deviations() -> None:
    episode = _make_episode(
        normalized_intent="fill the form",
        benchmark="auth_free_form",
        page_hint=PageHint.FORM_PAGE,
    )

    mock_store = MagicMock()
    mock_store.get_hints.return_value = []
    mock_store.get_episode.return_value = episode

    delegate = _MockPolicyDelegate()
    from src.agent.policy_coordinator import PolicyCoordinator

    coordinator = PolicyCoordinator(delegate=delegate, memory_store=mock_store)
    # Use a page hint that mismatches the episode's FORM_PAGE
    wrong_perception = _make_perception(page_hint=PageHint("gmail_inbox"))

    # First step initializes replay; wrong page_hint → first deviation
    await coordinator.choose_action(_make_state(step_count=1), wrong_perception)
    assert coordinator._replay_state is not None
    assert coordinator._replay_state.deviations == 1

    # Manually reset current_step_index so the second call still checks step 0
    coordinator._replay_state.current_step_index = 0
    # Second deviation reaches max (2) → deactivates
    await coordinator.choose_action(_make_state(step_count=2), wrong_perception)
    assert coordinator._replay_state.active is False
