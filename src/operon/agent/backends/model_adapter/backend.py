"""``ModelAdapterBackend``: any profiled model as an ``AgentBackend``."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from operon.agent.actions.translation import build_policy_decision
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.agent.backends.model_adapter.actions import (
    CoordinateMapper,
    InvalidAction,
    to_payload,
)
from operon.agent.backends.model_adapter.formats import (
    ActionFormat,
    ParsedReply,
    PromptContext,
    Turn,
    get_format,
)
from operon.agent.backends.model_adapter.images import SentImage, prepare_image
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports import ModelTransportError
from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.models.usage import ModelUsage
from operon.store.background_writer import bg_writer

logger = logging.getLogger(__name__)

# Rationale prefix of a STOP the model chose with ``fail``; the own-done verifier
# reads it to end the run as given up rather than as a claimed success.
GAVE_UP_PREFIX = "Model gave up: "
# Rationale prefix of the WAIT that stands in for a reply the parser could not use.
INVALID_PREFIX = "Invalid model reply: "


@dataclass(slots=True)
class _Run:
    turns: list[Turn] = field(default_factory=list)
    acknowledged_actions: int = 0
    not_executed_reason: str | None = None
    # The full harness ran another action in place of the last turn's (KAN-68).
    replaced_by: str | None = None
    invalid_replies: int = 0
    replies: int = 0


class ModelAdapterBackend(AgentBackend):
    """Prompt and history builder + transport + action parser for one model profile.

    The history keeps every turn, with the screenshot for only the last
    ``history_screens`` of them (older ones become a short text placeholder). A reply
    the parser cannot use is recorded and answered with a WAIT, and the model is told
    why on the next turn; it never crashes the run.
    """

    def __init__(
        self,
        *,
        profile: ModelProfile,
        client: Any,
        environment: Literal["browser", "desktop"],
        task_prompt_path: Path,
        action_format: ActionFormat | None = None,
        browser_runtime: Any = None,
    ) -> None:
        self.profile = profile
        self.client = client
        self.environment = environment
        self.format = action_format or get_format(profile.parser)
        self.browser_runtime = browser_runtime
        self._task_template = task_prompt_path.read_text(encoding="utf-8")
        self._runs: dict[str, _Run] = {}
        self._cached_decision: PolicyDecision | None = None
        self._last_frames: dict[str, CaptureFrame] = {}
        self._last_debug: ModelDebugArtifacts | None = None
        self._hints: dict[str, list[str]] = {}

    # ── AgentBackend ────────────────────────────────────────────────────────
    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        self._last_frames[state.run_id] = screenshot
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path, raw_path = step_dir / "model_prompt.json", step_dir / "model_raw.json"
        parsed_path, usage_path = step_dir / "model_parsed.json", step_dir / "model_usage.json"

        run = self._runs.setdefault(state.run_id, _Run())
        self._settle_last_turn(run, state)
        image = prepare_image(Path(screenshot.artifact_path), self.profile.image)
        ctx = self._context(state, image, screenshot)
        note = await self._note(state.run_id)
        messages = self._messages(run, ctx, image, note)
        tools = self.format.tools(ctx)
        bg_writer.enqueue(prompt_path, json.dumps({"messages": _without_images(messages), "tools": tools}, indent=2))

        try:
            response = await self.client.chat(messages, tools=tools)
        except ModelTransportError as exc:
            raise BackendCompatibilityError(str(exc)) from exc
        bg_writer.enqueue(raw_path, json.dumps(response, indent=2))
        message = response["choices"][0]["message"]
        parsed = self.format.parse(message, ctx)
        decision = self._decide(parsed, ctx)
        run.replies += 1
        if parsed.invalid:
            run.invalid_replies += 1
            logger.info("run %s: invalid reply from %s: %s", state.run_id[:8], self.profile.name, parsed.invalid)
        run.turns.append(Turn(image=image, assistant=message, parsed=parsed, note=note))
        # Only the last history_screens images are ever sent again; free the older ones.
        for old_turn in run.turns[: max(0, len(run.turns) - self.profile.history_screens)]:
            old_turn.image = None
        self._cached_decision = decision

        perception = ScreenPerception(
            summary=parsed.thought[:2000] or f"{self.environment.title()} screen observed",
            page_hint=PageHint.UNKNOWN,
            visible_elements=[],
            capture_artifact_path=screenshot.artifact_path,
            confidence=0.5,
            elements_reported=False,
        )
        bg_writer.enqueue(
            parsed_path,
            json.dumps(
                {
                    "profile": self.profile.name,
                    "parser": self.format.name,
                    "invalid": parsed.invalid is not None,
                    "invalid_reason": parsed.invalid,
                    "invalid_replies": run.invalid_replies,
                    "replies": run.replies,
                    "actions": [{"name": a.name, "args": a.args} for a in parsed.actions],
                    "terminal": None if parsed.terminal is None else {"name": parsed.terminal.name, "args": parsed.terminal.args},
                    "sent_image": [image.width, image.height],
                    "screen": [screenshot.width, screenshot.height],
                    "decision": decision.model_dump(mode="json"),
                }
            ),
        )
        usage = self.client.latest_usage() if hasattr(self.client, "latest_usage") else None
        if isinstance(usage, ModelUsage):
            bg_writer.enqueue(usage_path, usage.model_dump_json(indent=2))
        self._last_debug = ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(parsed_path),
            usage_artifact_path=str(usage_path),
            usage=usage if isinstance(usage, ModelUsage) else None,
        )
        return perception

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        if self._cached_decision is None:
            frame = self._last_frames.get(state.run_id)
            if frame is None or not perception.capture_artifact_path:
                raise RuntimeError("No cached model decision available")
            await self.perceive(frame.model_copy(update={"artifact_path": perception.capture_artifact_path}), state)
        decision, self._cached_decision = self._cached_decision, None
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug

    def supports_resampling(self, run_id: str = "") -> bool:
        return False

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self._hints.setdefault(run_id, []).extend(h for h in hints if h)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self._hints.pop(run_id, None)

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        self._hints = {"": [h for h in hints if h]}

    def note_not_executed(self, run_id: str, reason: str) -> None:
        run = self._runs.get(run_id)
        if run is not None and run.turns:
            run.not_executed_reason = reason.strip()[:300] or "blocked by a gate"

    def note_replaced(self, run_id: str, action: str) -> None:
        run = self._runs.get(run_id)
        if run is not None and run.turns:
            run.replaced_by = action.strip()[:300] or "another action"

    def invalid_reply_counts(self, run_id: str) -> tuple[int, int]:
        """(invalid replies, all replies) so far in a run."""
        run = self._runs.get(run_id)
        return (run.invalid_replies, run.replies) if run else (0, 0)

    # ── internals ───────────────────────────────────────────────────────────
    def _context(self, state: AgentState, image: SentImage, screenshot: CaptureFrame) -> PromptContext:
        mapper = CoordinateMapper(
            space=self.profile.coords,
            sent_width=image.width,
            sent_height=image.height,
            screen_width=screenshot.width or image.source_width,
            screen_height=screenshot.height or image.source_height,
        )
        task_prompt = self._task_template.format(
            intent=state.intent, environment_rules=PromptContext.environment_rules(self.environment)
        )
        return PromptContext(intent=state.intent, environment=self.environment, mapper=mapper, task_prompt=task_prompt)

    def _settle_last_turn(self, run: _Run, state: AgentState) -> None:
        """Record what happened to the last turn's actions, as the model will hear it."""
        if not run.turns or run.turns[-1].outcome is not None:
            return
        last = run.turns[-1]
        if last.parsed.invalid:
            last.outcome = (
                f"Error: your reply could not be used ({last.parsed.invalid}), so nothing ran. "
                "Look at the screen and reply again in the required format."
            )
        elif run.not_executed_reason:
            last.outcome = (
                f"Not executed: the run paused for a human before this action ran ({run.not_executed_reason}). "
                "Check the screen and propose the next action again."
            )
            run.not_executed_reason = None
        elif run.replaced_by:
            last.outcome = _replaced_outcome(run.replaced_by, state, run.acknowledged_actions)
        elif len(state.action_history) > run.acknowledged_actions:
            executed = state.action_history[-1]
            last.outcome = "OK" if executed.success else f"Error: {executed.detail or 'the action failed'}"
        else:
            # A re-look before the action was recorded (a mid-step retry).
            last.outcome = "OK"
        run.acknowledged_actions = len(state.action_history)
        run.replaced_by = None

    def _messages(self, run: _Run, ctx: PromptContext, image: SentImage, note: str) -> list[dict[str, Any]]:
        keep_images_from = len(run.turns) - (self.profile.history_screens - 1)
        if self.format.system_as_user:
            # The format's prompt already carries the task.
            messages: list[dict[str, Any]] = [{"role": "user", "content": self.format.system_prompt(ctx)}]
        else:
            messages = [
                {"role": "system", "content": self.format.system_prompt(ctx)},
                {"role": "user", "content": ctx.task_prompt},
            ]
        carry = ""
        for index, turn in enumerate(run.turns):
            messages.append(
                _user_message(turn.image if index >= keep_images_from else None, _join(carry, turn.note))
            )
            replayed, carry = self.format.replay(turn)
            messages.extend(replayed)
        messages.append(_user_message(image, _join(carry, note)))
        return messages

    async def _note(self, run_id: str) -> str:
        parts = []
        if self.environment == "browser" and self.browser_runtime is not None and hasattr(self.browser_runtime, "get_current_url"):
            try:
                if url := await self.browser_runtime.get_current_url():
                    parts.append(f"Current URL: {url}")
            except Exception:  # noqa: BLE001 - the URL is a courtesy, never a blocker
                pass
        hints = self._hints.pop(run_id, None) or self._hints.pop("", None) or []
        if hints:
            parts.append("Advisory hints:\n" + "\n".join(f"- {h}" for h in hints))
        return "\n".join(parts)

    def _decide(self, parsed: ParsedReply, ctx: PromptContext) -> PolicyDecision:
        thought = parsed.thought.strip()
        if parsed.invalid is None and parsed.terminal is not None:
            args = parsed.terminal.args or {}
            if parsed.terminal.name == "fail":
                reason = str(args.get("reason") or thought or "no reason given")
                return _stop(f"{GAVE_UP_PREFIX}{reason}", None)
            answer = str(args.get("answer") or "").strip() or None
            return _stop(thought or answer or "Task complete.", answer)
        if parsed.invalid is None:
            try:
                payload = to_payload(parsed.actions, ctx.mapper)
            except InvalidAction as exc:
                parsed.invalid = str(exc)
            else:
                return build_policy_decision(
                    {
                        "action": payload,
                        "rationale": thought or self.profile.name,
                        "confidence": 0.7,
                        "active_subgoal": _describe(parsed),
                    }
                )
        return build_policy_decision(
            {
                "action": {"action_type": "wait", "wait_ms": 500},
                "rationale": f"{INVALID_PREFIX}{parsed.invalid}",
                "confidence": 0.3,
                "active_subgoal": "reply in the required format",
            }
        )


def _stop(rationale: str, answer: str | None) -> PolicyDecision:
    return PolicyDecision(
        action=AgentAction(action_type=ActionType.STOP, text=answer),
        rationale=rationale[:2000],
        confidence=0.7,
        active_subgoal="complete task",
    )


def _describe(parsed: ParsedReply) -> str:
    """Name the step by its actions and arguments, so two steps never share a subgoal."""
    parts = []
    for action in parsed.actions:
        args = " ".join(f"{k}={action.args[k]}" for k in sorted(action.args or {}))
        parts.append(f"{action.name} {args}".strip())
    return "; ".join(parts)[:500] or "model step"


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _user_message(image: SentImage | None, text: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    else:
        content.append({"type": "text", "text": "(earlier screenshot omitted)"})
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _without_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                return {"type": "image_url", "image_url": "<screenshot>"}
            if value.get("type") in {"input_image", "computer_screenshot"}:
                # Responses API shapes (OpenAI computer use): image_url is the data URL itself.
                return {**value, "image_url": "<screenshot>"}
            return {k: strip(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(messages)


def _replaced_outcome(action: str, state: AgentState, acknowledged: int) -> str:
    """What the model hears when the full harness ran something else in its place."""
    result = ""
    if len(state.action_history) > acknowledged:
        executed = state.action_history[-1]
        result = " It succeeded." if executed.success else f" It failed: {executed.detail or 'the action failed'}."
    return (
        f"Not executed: Operon's harness ran a different action instead ({action}).{result} "
        "Check the screen and propose the next action."
    )
