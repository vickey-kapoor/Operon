"""Prompt-driven policy delegate for coordinator tests."""

from __future__ import annotations

import json
from pathlib import Path

from operon.agent.policy import PolicyService, parse_policy_output
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.state import AgentState


class PromptPolicyStub(PolicyService):
    """Renders a prompt template, sends it to a stub client and parses the JSON reply.

    Advisory hints are appended to the prompt the way the JSON backends do, so
    tests can assert that the coordinator passed them on.
    """

    def __init__(self, gemini_client, prompt_path: Path) -> None:
        self.gemini_client = gemini_client
        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        self._advisory_hints: dict[str, list[str]] = {}

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self._advisory_hints.setdefault(run_id, []).extend(h for h in hints if h)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self._advisory_hints.pop(run_id, None)

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        prompt = self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            perception_json=perception.model_dump_json(),
        )
        hints = self._advisory_hints.pop(state.run_id, None)
        if hints:
            prompt = f"{prompt}\n\nAdvisory memory hints:\n" + "\n".join(f"- {h}" for h in hints)
        return parse_policy_output(await self.gemini_client.generate_policy(prompt))
