"""Browser backend for Claude's computer use toolset (``OPERON_BROWSER_BACKEND=claude_computer_use``)."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from operon.agent._agent_utils import collect_latest_usage
from operon.agent.actions.translation import (
    build_policy_decision,
    normalize_claude_computer_calls,
)
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.clients.anthropic import AnthropicClientError
from operon.clients.anthropic_computer_use import (
    TOOLSET_NAME,
    AnthropicComputerUseHttpClient,
)
from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, PolicyDecision
from operon.models.state import AgentState
from operon.store.background_writer import bg_writer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are a browser automation agent. You see the browser through screenshots and act with the tools."

# Screenshots are downscaled so their long edge fits this; Claude's coordinates are
# scaled back up. Smaller images cost fewer tokens and the docs advise web
# resolutions around 1280x800.
_MAX_LONG_EDGE = 1280

_NOT_EXECUTED = "Not executed: an earlier computer action in this turn failed."
_CONTINUE = "Continue with the task using the tools. If it is already done, say so in one sentence."


@dataclass(slots=True)
class _ConversationState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    # tool_use blocks of the last assistant turn that still need a tool_result.
    pending_calls: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_actions: int = 0
    # Action types Claude's open calls translate to (a batch and its members). A
    # deterministic rule can run something else in their place; Claude must hear
    # that its calls did not run.
    proposed_action_types: frozenset[ActionType] = frozenset()


def encode_screenshot(path: Path, max_long_edge: int = _MAX_LONG_EDGE) -> tuple[str, str, int, int]:
    """Return (base64 data, media type, width, height) of the image Claude is sent."""
    raw = path.read_bytes()
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            ratio = min(1.0, max_long_edge / max(width, height))
            if ratio < 1.0:
                width, height = max(1, round(width * ratio)), max(1, round(height * ratio))
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg", width, height
    except Exception:  # noqa: BLE001 - an unreadable image is sent as-is
        logger.debug("Could not decode screenshot %s; sending raw bytes", path)
        return base64.b64encode(raw).decode("ascii"), "image/png", 0, 0


class BrowserClaudeComputerUseBackend(AgentBackend):
    """Drives the browser with Claude's ``computer_toolset_20260801``.

    One Messages API conversation per run: each step answers the previous turn's
    tool calls with the new screenshot and asks for the next ones. API failures
    and refusals surface as compatibility errors so the configured fallback
    backend can take over.
    """

    system_prompt = _SYSTEM_PROMPT
    # Custom tools answered alongside the computer toolset's members.
    custom_tool_names: frozenset[str] = frozenset({"navigate"})
    screen_name = "Browser"

    def __init__(
        self,
        *,
        client: AnthropicComputerUseHttpClient,
        prompt_path: Path,
        browser_runtime=None,
    ) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.browser_runtime = browser_runtime
        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        self._cached_decision: PolicyDecision | None = None
        self._last_frames: dict[str, CaptureFrame] = {}
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._advisory_hints: dict[str, list[tuple[str, str]]] = {}
        self._conversations: dict[str, _ConversationState] = {}

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        self._last_frames[state.run_id] = screenshot
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "computer_use_prompt.txt"
        raw_path = step_dir / "computer_use_raw.json"
        parsed_path = step_dir / "computer_use_parsed.json"

        conversation = self._conversations.setdefault(state.run_id, _ConversationState())
        open_calls, acknowledged = list(conversation.pending_calls), conversation.acknowledged_actions
        image, media_type, sent_width, sent_height = encode_screenshot(Path(screenshot.artifact_path))
        user_message = await self._next_user_message(
            conversation, state=state, image=image, media_type=media_type
        )
        bg_writer.enqueue(prompt_path, json.dumps(_without_images(user_message), indent=2))
        conversation.messages.append(user_message)

        try:
            response = await self.client.generate_turn(system=self.system_prompt, messages=conversation.messages)
        except AnthropicClientError as exc:
            # Leave the conversation as it was so a later turn can answer the same calls.
            conversation.messages.pop()
            conversation.pending_calls, conversation.acknowledged_actions = open_calls, acknowledged
            raise BackendCompatibilityError(str(exc)) from exc
        bg_writer.enqueue(raw_path, json.dumps(response, indent=2))

        if response.get("stop_reason") == "refusal":
            conversation.messages.pop()
            conversation.pending_calls, conversation.acknowledged_actions = open_calls, acknowledged
            details = response.get("stop_details") or {}
            raise BackendCompatibilityError(f"Claude declined the turn ({details.get('category') or 'refusal'}).")

        content = response["content"]
        # Pass every block back unchanged, thinking blocks included.
        conversation.messages.append({"role": "assistant", "content": content})
        calls = [
            block for block in content
            if block.get("type") == "tool_use"
            and (block.get("toolset_name") == TOOLSET_NAME or block.get("name") in self.custom_tool_names)
        ]
        conversation.pending_calls = calls
        text = " ".join(
            block["text"].strip() for block in content if block.get("type") == "text" and block.get("text", "").strip()
        )

        perception = ScreenPerception(
            summary=text or self._describe(calls) or f"{self.screen_name} screen observed",
            page_hint=PageHint.UNKNOWN,
            visible_elements=[],
            capture_artifact_path=screenshot.artifact_path,
            confidence=0.5,
            elements_reported=False,
        )
        if calls:
            scale_x = screenshot.width / sent_width if sent_width else 1.0
            scale_y = screenshot.height / sent_height if sent_height else 1.0
            action = normalize_claude_computer_calls(
                [{"name": call.get("name"), "input": call.get("input") or {}} for call in calls],
                scale_x=scale_x,
                scale_y=scale_y,
                screen_width=screenshot.width,
                screen_height=screenshot.height,
            )
            decision_payload = {
                "action": action,
                "rationale": text or "computer_use",
                "confidence": 0.7,
                "active_subgoal": self._describe(calls),
            }
        elif self._has_acted(state):
            # A reply without tool calls after acting is Claude saying the task is
            # done. Surface it as STOP so the verifier judges the claim.
            decision_payload = {
                "action": {"action_type": ActionType.STOP.value},
                "rationale": text or "Task complete.",
                "confidence": 0.7,
                "active_subgoal": "complete task",
            }
        else:
            decision_payload = {
                "action": {"action_type": "wait", "wait_ms": 1000},
                "rationale": text or "No action proposed.",
                "confidence": 0.4,
                "active_subgoal": f"gather more {self.screen_name.lower()} context",
            }
        self._cached_decision = build_policy_decision(decision_payload)
        proposed = self._cached_decision.action
        conversation.proposed_action_types = (
            frozenset({proposed.action_type, *(child.action_type for child in proposed.actions or [])})
            if calls
            else frozenset()
        )
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
            usage_artifact_path=str(step_dir / "computer_use_usage.json"),
            usage=collect_latest_usage(self.client, step_dir / "computer_use_usage.json"),
        )
        return perception

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        if self._cached_decision is None:
            # perceive() was skipped or a second decision is asked for this step:
            # take a fresh turn on the current screenshot.
            frame = self._last_frames.get(state.run_id)
            if frame is None or not perception.capture_artifact_path:
                raise RuntimeError("No cached computer-use decision available")
            await self.perceive(frame.model_copy(update={"artifact_path": perception.capture_artifact_path}), state)
        decision = self._cached_decision
        self._cached_decision = None
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def supports_resampling(self, run_id: str = "") -> bool:
        # Each decision is a turn in one stateful conversation; there is no second sample.
        return False

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state. Test use only."""
        self._advisory_hints = {"": [(h, "") for h in hints if h]}

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self._advisory_hints.setdefault(run_id, []).extend((h, source) for h in hints if h)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self._advisory_hints.pop(run_id, None)

    async def _next_user_message(
        self,
        conversation: _ConversationState,
        *,
        state: AgentState,
        image: str,
        media_type: str,
    ) -> dict[str, Any]:
        image_block = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image}}
        hints = self._take_hints(state.run_id)
        if not conversation.messages:
            prompt = self._prompt_template.format(
                intent=state.intent,
                current_subgoal=state.current_subgoal or "not set",
                step_count=state.step_count,
                previous_summary=state.observation_history[-1].summary if state.observation_history else "none",
                retry_counts=json.dumps(state.retry_counts, sort_keys=True),
                advisory_hints="\n".join(f"- {h}" for h in hints) if hints else "none",
            )
            return {"role": "user", "content": [{"type": "text", "text": prompt}, image_block]}

        if not conversation.pending_calls:
            # The last turn had no tool call: show the current screen and nudge on.
            text = _CONTINUE + _hint_text(hints)
            return {"role": "user", "content": [{"type": "text", "text": text}, image_block]}

        # Answer every open call. A mid-step re-look (an execution or adaptation
        # retry) arrives before the action is recorded, so it reports success
        # with the current screen.
        error = None
        if len(state.action_history) > conversation.acknowledged_actions:
            executed = state.action_history[-1]
            ran = executed.action.action_type
            if conversation.proposed_action_types and ran not in conversation.proposed_action_types:
                detail = f" ({executed.detail})" if executed.detail else ""
                error = f"Not executed: the agent ran {ran.value}{detail} instead. Check the screen before going on."
            elif not executed.success:
                error = executed.detail or "The action failed."
        conversation.acknowledged_actions = len(state.action_history)

        results: list[dict[str, Any]] = []
        for index, call in enumerate(conversation.pending_calls):
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": call["id"]}
            if call.get("toolset_name") == TOOLSET_NAME:
                result["toolset_name"] = TOOLSET_NAME
            if error is not None:
                # The batch ran as one action; report the failure on its first call
                # and mark the rest as skipped, as the toolset expects.
                result["is_error"] = True
                result["content"] = f"Error: {error}" if index == 0 else _NOT_EXECUTED
            elif call.get("name") == "screenshot":
                result["content"] = [image_block]
            else:
                result["content"] = [{"type": "text", "text": "OK"}]
            results.append(result)
        conversation.pending_calls = []

        extra: list[dict[str, Any]] = []
        answered_with_image = error is None and any(r.get("content") == [image_block] for r in results)
        if not answered_with_image:
            if error is None:
                results[-1]["content"] = [*results[-1]["content"], image_block]
            else:
                extra.append(image_block)
        current_url = await self._current_url()
        note = (f"Current URL: {current_url}" if current_url else "") + _hint_text(hints)
        if note.strip():
            extra.append({"type": "text", "text": note.strip()})
        return {"role": "user", "content": [*results, *extra]}

    async def _current_url(self) -> str | None:
        if self.browser_runtime is not None and hasattr(self.browser_runtime, "get_current_url"):
            try:
                return await self.browser_runtime.get_current_url()
            except Exception:  # noqa: BLE001 - the URL is a courtesy, never a blocker
                return None
        return None

    def _take_hints(self, run_id: str) -> list[str]:
        hints = self._advisory_hints.pop(run_id, None) or self._advisory_hints.pop("", None) or []
        return [h for h, _ in hints]

    @staticmethod
    def _has_acted(state: AgentState) -> bool:
        # A step reported as failed still counts: it may have done its job (a
        # submit that navigated away mid-batch), and Claude, who sees the screen,
        # is the one saying the task is done. The verifier judges that claim;
        # treating it as a wait instead loops on the "say so" nudge until max_steps.
        return any(
            executed.action.action_type not in {ActionType.WAIT, ActionType.STOP, ActionType.WAIT_FOR_USER}
            for executed in state.action_history
        )

    @staticmethod
    def _describe(calls: list[dict[str, Any]]) -> str:
        """Name the step by the call and its arguments, so two different steps on
        the same page never share a subgoal (see BrowserComputerUseBackend)."""
        described = []
        for call in calls:
            args = call.get("input") or {}
            details = " ".join(f"{key}={args[key]}" for key in sorted(args))
            described.append(f"{call.get('name', 'browser step')} {details}".strip())
        return "; ".join(described)


def _hint_text(hints: list[str]) -> str:
    if not hints:
        return ""
    return "\nAdvisory hints:\n" + "\n".join(f"- {h}" for h in hints)


def _without_images(message: dict[str, Any]) -> dict[str, Any]:
    """The user turn as written to disk, with image data elided."""

    def strip(block: Any) -> Any:
        if isinstance(block, dict):
            if block.get("type") == "image":
                return {"type": "image", "source": "<screenshot>"}
            return {key: strip(value) for key, value in block.items()}
        if isinstance(block, list):
            return [strip(item) for item in block]
        return block

    return strip(message)
