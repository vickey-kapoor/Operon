"""Anthropic-backed planner service for next-action selection."""

from __future__ import annotations

from pathlib import Path

from src.agent.policy import GeminiPolicyService
from src.clients.anthropic import AnthropicClientError, AnthropicHttpClient
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.state import AgentState


class AnthropicPolicyService(GeminiPolicyService):
    """Planner service that reuses the strict policy parser with Anthropic output."""

    def __init__(
        self,
        anthropic_client: AnthropicHttpClient,
        prompt_path: Path | None = None,
    ) -> None:
        self.anthropic_client = anthropic_client
        super().__init__(gemini_client=anthropic_client, prompt_path=prompt_path)

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        prompt = self._render_prompt(state, perception)
        step_dir = Path(perception.capture_artifact_path).resolve().parent
        debug_artifacts = self._artifact_paths(step_dir)
        from src.store.background_writer import bg_writer

        bg_writer.enqueue(debug_artifacts.prompt_artifact_path, prompt)
        try:
            raw_output = await self.anthropic_client.generate_policy(prompt)
        except AnthropicClientError:
            raise
        bg_writer.enqueue(debug_artifacts.raw_response_artifact_path, raw_output)
        decision = self._apply_focus_first_guardrail(state, perception, self._parse_output(raw_output))
        bg_writer.enqueue(debug_artifacts.parsed_artifact_path, decision.model_dump_json())
        from src.models.logs import ModelDebugArtifacts

        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(debug_artifacts.prompt_artifact_path),
            raw_response_artifact_path=str(debug_artifacts.raw_response_artifact_path),
            parsed_artifact_path=str(debug_artifacts.parsed_artifact_path),
        )
        return decision

    @staticmethod
    def _parse_output(raw_output: str) -> PolicyDecision:
        from src.agent.policy import parse_policy_output

        return parse_policy_output(raw_output)
