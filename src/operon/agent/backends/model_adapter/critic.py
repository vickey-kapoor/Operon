"""The full harness's step critic on the run's own model (``OPERON_CRITIC=same``, KAN-68).

The critic (``DeterministicVerifierService``) calls ``generate_verification(prompt,
screenshot_path)`` and reads ``latest_usage()``. Claude and Gemini profiles reuse
Operon's own clients for that; this module gives OpenAI profiles the same two methods,
so "GPT full" is judged by GPT rather than by a Gemini critic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operon.agent.backends.model_adapter.images import prepare_image
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports import (
    ModelTransportError,
    OpenAICompatClient,
    OpenAIResponsesClient,
)
from operon.models.usage import ModelUsage


class ProfileCriticClient:
    """A critic client for an ``openai_native`` or ``openai_compat`` registry profile."""

    def __init__(self, profile: ModelProfile, client: Any | None = None) -> None:
        if profile.adapter not in {"openai_native", "openai_compat"}:
            raise ValueError(f"profile {profile.name}: no critic client for adapter {profile.adapter}")
        self.profile = profile
        self.client = client or (
            OpenAIResponsesClient(profile) if profile.adapter == "openai_native" else OpenAICompatClient(profile)
        )

    def latest_usage(self) -> ModelUsage | None:
        return self.client.latest_usage()

    async def generate_verification(self, prompt: str, screenshot_path: str) -> str:
        # The profile's own size and format caps, as the acting backend sends them.
        image = prepare_image(Path(screenshot_path), self.profile.image).data_url
        if self.profile.adapter == "openai_native":
            body = await self.client.create(
                input_items=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image},
                ]}],
                tools=[],
                previous_response_id=None,
            )
            text = "".join(
                part.get("text") or ""
                for item in body.get("output") or []
                if item.get("type") == "message"
                for part in item.get("content") or []
                if part.get("type") == "output_text"
            )
        else:
            body = await self.client.chat(
                [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image}},
                ]}],
                tools=None,
            )
            text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not text.strip():
            raise ModelTransportError(f"{self.profile.name}: the critic reply has no text")
        return text
