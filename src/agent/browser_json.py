from __future__ import annotations

from pathlib import Path

from src.agent.combined import CombinedPerceptionPolicyService
from src.clients.gemini import GeminiClient


class BrowserJsonBackend(CombinedPerceptionPolicyService):
    """Browser backend using the existing combined JSON perception/policy flow."""

    def __init__(self, *, gemini_client: GeminiClient, prompt_path: Path) -> None:
        super().__init__(gemini_client=gemini_client, prompt_path=prompt_path)
