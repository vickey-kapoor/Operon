from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.action_translation import (
    build_policy_decision,
    normalize_computer_use_actions,
)
from src.agent.backend import AgentBackend
from src.agent.fallback_backend import BackendCompatibilityError
from src.clients.gemini_computer_use import (
    GeminiComputerUseClient,
    GeminiComputerUseError,
)
from src.models.capture import CaptureFrame
from src.models.logs import ModelDebugArtifacts
from src.models.perception import PageHint, ScreenPerception
from src.models.policy import PolicyDecision
from src.models.state import AgentState
from src.store.background_writer import bg_writer


@dataclass(slots=True)
class _ConversationState:
    contents: list[dict] = field(default_factory=list)
    pending_function_names: list[str] = field(default_factory=list)
    acknowledged_actions: int = 0


class BrowserComputerUseBackend(AgentBackend):
    """Browser backend for Google Computer Use models.

    The request/response protocol is intentionally isolated from the existing
    JSON backend. Until the transport is fully implemented, protocol-level
    issues are surfaced as compatibility errors so the route-configured JSON
    fallback can take over.
    """

    def __init__(
        self,
        *,
        client: GeminiComputerUseClient,
        prompt_path: Path,
        browser_runtime=None,
    ) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.browser_runtime = browser_runtime
        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        self._retry_suffix = (
            "\n\nRetry guidance:\n"
            "- A text-only response is not enough when a harmless browser tool action is available.\n"
            "- Prefer one concrete tool call that reveals more page structure.\n"
            "- If uncertain, choose a low-risk action like hover_at or scroll_at instead of waiting.\n"
        )
        self._cached_decision: PolicyDecision | None = None
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._advisory_hints: list[str] = []
        self._conversations: dict[str, _ConversationState] = {}

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "computer_use_prompt.txt"
        raw_path = step_dir / "computer_use_raw.json"
        parsed_path = step_dir / "computer_use_parsed.json"

        prompt = self._render_prompt(state)
        bg_writer.enqueue(prompt_path, prompt)

        response_payload = await self._run_with_retry(prompt=prompt, screenshot=screenshot, state=state)

        bg_writer.enqueue(raw_path, json.dumps(response_payload, indent=2))

        perception_payload = response_payload.get("perception") or {}
        perception = self._build_perception(
            payload=perception_payload,
            screenshot_path=screenshot.artifact_path,
        )
        if "function_call" in response_payload:
            response_payload = {
                **response_payload,
                "action": normalize_computer_use_actions(
                    response_payload.get("function_calls") or [response_payload["function_call"]],
                    screen_width=screenshot.width,
                    screen_height=screenshot.height,
                ),
            }
        self._cached_decision = build_policy_decision(response_payload)
        bg_writer.enqueue(
            parsed_path,
            json.dumps(
                {
                    "perception": perception.model_dump(mode="json"),
                    "decision": self._cached_decision.model_dump(mode="json"),
                }
            ),
        )
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(parsed_path),
        )
        return perception

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        if self._cached_decision is None:
            raise RuntimeError("No cached computer-use decision available")
        decision = self._cached_decision
        self._cached_decision = None
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def set_advisory_hints(self, hints: list[str]) -> None:
        self._advisory_hints = [hint for hint in hints if hint]

    async def _run_with_retry(self, *, prompt: str, screenshot: CaptureFrame, state: AgentState) -> dict:
        try:
            response_payload = await self._run_turn(prompt=prompt, screenshot=screenshot, state=state)
        except (GeminiComputerUseError, BackendCompatibilityError) as exc:
            raise BackendCompatibilityError(str(exc)) from exc

        if not self._needs_stronger_retry(response_payload):
            return response_payload

        if state.action_history:
            return response_payload

        retry_prompt = prompt + self._retry_suffix
        try:
            retry_payload = await self.client.run_step(
                prompt=retry_prompt,
                screenshot_path=screenshot.artifact_path,
            )
        except (GeminiComputerUseError, BackendCompatibilityError):
            return response_payload
        if not self._needs_stronger_retry(retry_payload):
            conversation = _ConversationState()
            conversation.contents.append(
                self._build_user_content(
                    prompt=retry_prompt,
                    image_path=Path(screenshot.artifact_path),
                )
            )
            model_content = retry_payload.get("model_content")
            if isinstance(model_content, dict):
                conversation.contents.append({"role": "model", "parts": model_content.get("parts", [])})
            if retry_payload.get("function_calls"):
                conversation.pending_function_names = [
                    function_call.get("name")
                    for function_call in retry_payload["function_calls"]
                    if function_call.get("name")
                ]
            self._conversations[state.run_id] = conversation
        return retry_payload if not self._needs_stronger_retry(retry_payload) else response_payload

    async def _run_turn(self, *, prompt: str, screenshot: CaptureFrame, state: AgentState) -> dict:
        conversation = self._conversations.setdefault(state.run_id, _ConversationState())
        if conversation.pending_function_names and len(state.action_history) > conversation.acknowledged_actions:
            current_url = None
            if self.browser_runtime is not None and hasattr(self.browser_runtime, "get_current_url"):
                current_url = await self.browser_runtime.get_current_url()
            error = None
            last_action = state.action_history[-1]
            if not last_action.success:
                error = last_action.detail
            function_response = self._build_function_response_content(
                function_names=conversation.pending_function_names,
                screenshot_path=screenshot.artifact_path,
                current_url=current_url,
                error=error,
            )
            conversation.contents.append(function_response)
            conversation.acknowledged_actions = len(state.action_history)
            conversation.pending_function_names = []
        elif not conversation.contents:
            conversation.contents.append(
                self._build_user_content(
                    prompt=prompt,
                    image_path=Path(screenshot.artifact_path),
                )
            )

        response_payload = await self.client.generate_turn(contents=conversation.contents)
        model_content = response_payload.get("model_content")
        if isinstance(model_content, dict):
            conversation.contents.append({"role": "model", "parts": model_content.get("parts", [])})
        if response_payload.get("function_calls"):
            conversation.pending_function_names = [
                function_call.get("name")
                for function_call in response_payload["function_calls"]
                if function_call.get("name")
            ]
        return response_payload

    def _render_prompt(self, state: AgentState) -> str:
        hint_block = "\n".join(f"- {hint}" for hint in self._advisory_hints) if self._advisory_hints else "none"
        self._advisory_hints = []
        return self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            previous_summary=state.observation_history[-1].summary if state.observation_history else "none",
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            advisory_hints=hint_block,
        )

    @staticmethod
    def _needs_stronger_retry(response_payload: dict) -> bool:
        if "function_call" in response_payload:
            return False
        action = response_payload.get("action", {})
        if action.get("action_type") != "wait":
            return False
        confidence = float(response_payload.get("confidence", 0.0))
        summary = str(response_payload.get("perception", {}).get("summary", "")).lower()
        return confidence <= 0.4 or "computer use step evaluated" in summary

    @staticmethod
    def _build_perception(*, payload: dict, screenshot_path: str) -> ScreenPerception:
        return ScreenPerception(
            summary=payload.get("summary", "Browser screen observed"),
            page_hint=payload.get("page_hint", PageHint.UNKNOWN),
            visible_elements=payload.get("visible_elements", []),
            focused_element_id=payload.get("focused_element_id"),
            capture_artifact_path=screenshot_path,
            confidence=float(payload.get("confidence", 0.5)),
        )

    @staticmethod
    def _build_user_content(*, prompt: str, image_path: Path) -> dict:
        image_bytes = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        return {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ],
        }

    def _build_function_response_content(
        self,
        *,
        function_names: list[str],
        screenshot_path: str,
        current_url: str | None,
        error: str | None = None,
    ) -> dict:
        first = self.client.build_function_response_content(
            function_name=function_names[0],
            screenshot_path=screenshot_path,
            current_url=current_url,
            error=error,
        )
        parts = [first["parts"][0]]
        for function_name in function_names[1:]:
            response_payload: dict[str, str] = {}
            if current_url:
                response_payload["url"] = current_url
            if error:
                response_payload["error"] = error
            parts.append(
                {
                    "function_response": {
                        "name": function_name,
                        "response": response_payload,
                    }
                }
            )
        parts.append(first["parts"][1])
        return {"role": "user", "parts": parts}
