"""Thin coordinator that runs explicit rules before LLM-backed policy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.policy import PolicyService
from src.agent.policy_rules import PolicyRuleEngine
from src.models.common import FailureCategory
from src.models.episode import Episode, EpisodeReplayState
from src.models.logs import ModelDebugArtifacts
from src.models.memory import MemoryHint
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.selector import SelectorTrace
from src.models.state import AgentState
from src.store.background_writer import bg_writer
from src.store.memory import MemoryStore, benchmark_name_for_intent, normalize_intent

logger = logging.getLogger(__name__)


class PolicyCoordinator(PolicyService):
    """Run explicit rules first, then fall back to the existing policy service."""

    def __init__(
        self,
        *,
        delegate: PolicyService,
        memory_store: MemoryStore,
        rule_engine: PolicyRuleEngine | None = None,
    ) -> None:
        self.delegate = delegate
        self.memory_store = memory_store
        self.rule_engine = rule_engine or PolicyRuleEngine()
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._active_episode: Episode | None = None
        self._replay_state: EpisodeReplayState | None = None

    def prepare_hints(self, state: AgentState, perception: ScreenPerception) -> None:
        """Fetch memory hints and inject them into the delegate BEFORE perceive().

        In combined mode the Gemini call happens during perceive(), so hints
        must be set before that call.  In separate mode this is a harmless no-op
        because choose_action() will re-fetch and set hints anyway.
        """
        memory_hints = self.memory_store.get_hints(
            benchmark=benchmark_name_for_intent(state.intent),
            page_hint=perception.page_hint if perception else None,
            subgoal=state.current_subgoal,
            recent_failure_category=self._recent_failure_category(state),
        )
        if hasattr(self.delegate, "set_advisory_hints"):
            existing = getattr(self.delegate, "_advisory_hints", []) or []
            self.delegate.set_advisory_hints(existing + [hint.hint for hint in memory_hints])

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        # Inject episode advisory hint if a matching trajectory exists
        self._try_episode_hint(state, perception)

        memory_hints = self.memory_store.get_hints(
            benchmark=benchmark_name_for_intent(state.intent),
            page_hint=perception.page_hint,
            subgoal=state.current_subgoal,
            recent_failure_category=self._recent_failure_category(state),
        )

        decision = self.rule_engine.choose_action(state, perception, memory_hints, benchmark_name=benchmark_name_for_intent(state.intent))
        selector_traces = self.rule_engine.latest_selector_traces()
        if decision is not None:
            self._last_debug_artifacts = self._write_rule_debug_artifacts(state, perception, memory_hints, decision, selector_traces)
            return decision

        if hasattr(self.delegate, "set_advisory_hints"):
            existing = getattr(self.delegate, "_advisory_hints", []) or []
            self.delegate.set_advisory_hints(existing + [hint.hint for hint in memory_hints])

        decision = await self.delegate.choose_action(state, perception)
        if hasattr(self.delegate, "latest_debug_artifacts"):
            self._last_debug_artifacts = self.delegate.latest_debug_artifacts()
        self._last_debug_artifacts = self._attach_selector_trace(perception, self._last_debug_artifacts, selector_traces)
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    @staticmethod
    def _recent_failure_category(state: AgentState) -> FailureCategory | None:
        if state.verification_history:
            failure = state.verification_history[-1].failure_category
            if failure is not None:
                return failure
        if state.action_history:
            return state.action_history[-1].failure_category
        return None

    def _write_rule_debug_artifacts(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
        decision: PolicyDecision,
        selector_traces: list[SelectorTrace],
    ) -> ModelDebugArtifacts:
        step_dir = Path(perception.capture_artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "policy_prompt.txt"
        raw_path = step_dir / "policy_raw.txt"
        parsed_path = step_dir / "policy_decision.json"
        selector_trace_path = self._write_selector_trace(step_dir, selector_traces)
        bg_writer.enqueue(prompt_path, self._render_rule_context(state, perception, memory_hints))
        bg_writer.enqueue(raw_path, json.dumps({
            "source": "rule",
            "decision_rationale": decision.rationale,
            "subgoal": decision.active_subgoal,
        }))
        bg_writer.enqueue(parsed_path, decision.model_dump_json())
        return ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(parsed_path),
            selector_trace_artifact_path=str(selector_trace_path) if selector_trace_path is not None else None,
        )

    @staticmethod
    def _render_rule_context(
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> str:
        lines = [
            "Policy coordinator selected a rule-based action.",
            f"intent: {state.intent}",
            f"subgoal: {state.current_subgoal or 'not set'}",
            f"page_hint: {perception.page_hint.value}",
            "memory_hints:",
        ]
        if not memory_hints:
            lines.append("- none")
        else:
            lines.extend(f"- {hint.key}: {hint.hint}" for hint in memory_hints)
        return "\n".join(lines)

    @staticmethod
    def _write_selector_trace(step_dir: Path, selector_traces: list[SelectorTrace]) -> Path | None:
        if not selector_traces:
            return None
        path = step_dir / "selector_trace.json"
        payload = [trace.model_dump(mode="json") for trace in selector_traces]
        bg_writer.enqueue(path, json.dumps(payload))
        return path

    def _attach_selector_trace(
        self,
        perception: ScreenPerception,
        debug_artifacts: ModelDebugArtifacts | None,
        selector_traces: list[SelectorTrace],
    ) -> ModelDebugArtifacts | None:
        if debug_artifacts is None:
            return None
        selector_trace_path = self._write_selector_trace(Path(perception.capture_artifact_path).resolve().parent, selector_traces)
        if selector_trace_path is None:
            return debug_artifacts
        return debug_artifacts.model_copy(update={"selector_trace_artifact_path": str(selector_trace_path)})

    # ------------------------------------------------------------------
    # Episode replay (advisory hints)
    # ------------------------------------------------------------------

    def _try_episode_hint(self, state: AgentState, perception: ScreenPerception) -> None:
        """Inject an advisory hint from a matching episode trajectory."""
        # Initialize replay on first step
        if self._replay_state is None and state.step_count <= 1:
            benchmark = benchmark_name_for_intent(state.intent)
            norm = normalize_intent(state.intent)
            episode = self.memory_store.get_episode(norm, benchmark)
            if episode is not None:
                self._active_episode = episode
                self._replay_state = EpisodeReplayState(episode_id=episode.episode_id)
                logger.info(
                    "Episode replay activated: %s (%d steps, %dx success)",
                    episode.episode_id[:8],
                    len(episode.steps),
                    episode.success_count,
                )

        if self._replay_state is None or not self._replay_state.active:
            return

        episode = self._active_episode
        if episode is None:
            return

        idx = self._replay_state.current_step_index
        if idx >= len(episode.steps):
            self._replay_state.active = False
            return

        expected = episode.steps[idx]

        # Divergence check: page_hint mismatch
        if perception.page_hint != expected.page_hint:
            self._replay_state.deviations += 1
            if self._replay_state.deviations >= self._replay_state.max_deviations:
                self._replay_state.active = False
                logger.info("Episode replay deactivated: too many divergences")
                return

        # Build advisory hint
        total = len(episode.steps)
        parts = [f"Episode replay (step {idx + 1}/{total}):"]
        parts.append(f"perform {expected.action_type.value}")
        if expected.target_description:
            parts.append(f"on '{expected.target_description}'")
        if expected.text:
            parts.append(f"with text '{expected.text}'")
        if expected.key:
            parts.append(f"key '{expected.key}'")
        parts.append(f"[subgoal: {expected.subgoal}]")
        parts.append("Follow this if the screen matches.")
        hint = " ".join(parts)

        if hasattr(self.delegate, "set_advisory_hints"):
            existing = getattr(self.delegate, "_advisory_hints", []) or []
            self.delegate.set_advisory_hints(list(existing) + [hint])

        self._replay_state.current_step_index += 1
