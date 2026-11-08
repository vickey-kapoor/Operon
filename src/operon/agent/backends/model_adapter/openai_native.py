"""OpenAI computer use (Responses API, ``{"type": "computer"}`` tool) as an ``AgentBackend``.

One stateful conversation per run, chained with ``previous_response_id``: each step
answers the previous turn's calls (a ``computer_call_output`` carrying the new
screenshot, a ``function_call_output`` for ``navigate`` / ``launch_app``) and reads
the next ones. A ``computer_call`` holds an ordered ``actions`` list (older replies
carry one ``action``); they map to neutral actions and run as one batch.

The model says it is done by replying with a message and no call. After it has
acted, that is a STOP (its own "done"); before, it is nudged to go on. A call with
pending safety checks pauses the run for a human; on resume they are acknowledged.
Experimental: used only through a registry profile with ``adapter: openai_native``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from operon.agent.actions.translation import build_policy_decision
from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.agent.backends.model_adapter.actions import (
    CoordinateMapper,
    InvalidAction,
    NeutralAction,
    to_payload,
)
from operon.agent.backends.model_adapter.backend import (
    INVALID_PREFIX,
    _stop,
    _without_images,
)
from operon.agent.backends.model_adapter.formats.base import PromptContext
from operon.agent.backends.model_adapter.images import SentImage, prepare_image
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports import ModelTransportError
from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, PolicyDecision
from operon.models.state import AgentState
from operon.models.usage import ModelUsage
from operon.store.background_writer import bg_writer

logger = logging.getLogger(__name__)

_CONTINUE = "Continue with the task. When it is fully done, reply with one sentence saying so and make no call."
_DONE_RULE = (
    "When the task is fully done, reply with one short sentence saying what the screen shows and make no "
    "further calls. Put any answer the task asks for in that sentence."
)


def _custom_tools(environment: str) -> list[dict[str, Any]]:
    if environment == "desktop":
        return [{
            "type": "function",
            "name": "launch_app",
            "description": "Open a desktop application by name, instead of using the Start menu.",
            "parameters": {
                "type": "object",
                "properties": {"app": {"type": "string", "enum": sorted(APP_LAUNCH_TARGETS)}},
                "required": ["app"],
                "additionalProperties": False,
            },
        }]
    return [{
        "type": "function",
        "name": "navigate",
        "description": "Open a URL in the current tab. The screenshots show no address bar.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    }]


def computer_actions(call: dict[str, Any]) -> list[NeutralAction]:
    """The neutral actions of one ``computer_call``, in order."""
    raw = call.get("actions")
    if raw is None and call.get("action") is not None:
        raw = [call["action"]]
    if not isinstance(raw, list) or not raw:
        raise InvalidAction("computer_call has no actions")
    actions = []
    for action in raw:
        if not isinstance(action, dict):
            raise InvalidAction("a computer action must be an object")
        kind = action.get("type")
        if action.get("keys") and kind in {"click", "double_click", "move", "scroll", "drag"}:
            raise InvalidAction(f"{kind} with held keys is not supported; press the keys as a keypress")
        point = {"x": action.get("x"), "y": action.get("y")}
        if kind == "click":
            button = str(action.get("button") or "left").lower()
            if button == "back":
                actions.append(NeutralAction("key", {"keys": "alt+Left"}))
            elif button == "forward":
                actions.append(NeutralAction("key", {"keys": "alt+Right"}))
            else:
                name = {"right": "right_click", "wheel": "middle_click", "middle": "middle_click"}.get(button, "click")
                actions.append(NeutralAction(name, point))
        elif kind == "double_click":
            actions.append(NeutralAction("double_click", point))
        elif kind == "move":
            actions.append(NeutralAction("hover", point))
        elif kind == "type":
            actions.append(NeutralAction("type", {"text": action.get("text")}))
        elif kind == "keypress":
            keys = action.get("keys") or []
            actions.append(NeutralAction("key", {"keys": [str(k) for k in keys] if isinstance(keys, list) else str(keys)}))
        elif kind == "scroll":
            dx, dy = float(action.get("scroll_x") or 0), float(action.get("scroll_y") or 0)
            if abs(dy) >= abs(dx):
                direction, amount = ("down" if dy > 0 else "up"), abs(dy)
            else:
                direction, amount = ("right" if dx > 0 else "left"), abs(dx)
            # OpenAI scrolls by pixels; neutral scroll counts 100-pixel wheel notches.
            actions.append(NeutralAction("scroll", {**point, "direction": direction, "amount": max(1, round(amount / 100))}))
        elif kind == "drag":
            path = action.get("path") or []
            if len(path) < 2:
                raise InvalidAction("drag needs a path of at least two points")
            (sx, sy), (ex, ey) = (_xy(path[0]), _xy(path[-1]))
            actions.append(NeutralAction("drag", {"start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}))
        elif kind == "wait":
            actions.append(NeutralAction("wait", {"seconds": 2}))
        elif kind == "screenshot":
            actions.append(NeutralAction("screenshot"))
        else:
            raise InvalidAction(f"unknown computer action {kind!r}")
    return actions


def _xy(point: Any) -> tuple[Any, Any]:
    if isinstance(point, dict):
        return point.get("x"), point.get("y")
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return point[0], point[1]
    raise InvalidAction("a drag path point must be {x, y} or [x, y]")


@dataclass(slots=True)
class _Conversation:
    previous_response_id: str | None = None
    # Calls of the last response that still need an output item.
    pending: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_actions: int = 0
    invalid: str | None = None
    not_executed_reason: str | None = None
    # Safety checks of a held call, acknowledged when its output is sent.
    held_checks: list[dict[str, Any]] = field(default_factory=list)
    # The held call's actions, run without a new request once the human resumes the run.
    held_payload: dict[str, Any] | None = None
    invalid_replies: int = 0
    replies: int = 0


class OpenAIComputerUseBackend(AgentBackend):
    def __init__(
        self,
        *,
        profile: ModelProfile,
        client: Any,
        environment: Literal["browser", "desktop"],
        task_prompt_path: Path,
        browser_runtime: Any = None,
    ) -> None:
        self.profile = profile
        self.client = client
        self.environment = environment
        self.browser_runtime = browser_runtime
        self._task_template = task_prompt_path.read_text(encoding="utf-8")
        self._runs: dict[str, _Conversation] = {}
        self._cached_decision: PolicyDecision | None = None
        self._last_frames: dict[str, CaptureFrame] = {}
        self._last_debug: ModelDebugArtifacts | None = None
        self._hints: dict[str, list[str]] = {}

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        self._last_frames[state.run_id] = screenshot
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path, raw_path = step_dir / "model_prompt.json", step_dir / "model_raw.json"
        parsed_path, usage_path = step_dir / "model_parsed.json", step_dir / "model_usage.json"

        conv = self._runs.setdefault(state.run_id, _Conversation())
        if conv.held_payload is not None:
            # The run paused on a safety check and a human resumed it: run the held
            # actions now; the next step acknowledges the checks with the outcome.
            payload, conv.held_payload = conv.held_payload, None
            conv.not_executed_reason = None
            self._cached_decision = build_policy_decision({
                "action": payload,
                "rationale": "Running the action the human confirmed.",
                "confidence": 0.7,
                "active_subgoal": "run the confirmed action",
            })
            conv.acknowledged_actions = len(state.action_history)
            self._last_debug = None
            return ScreenPerception(
                summary="Confirmed action resumed",
                page_hint=PageHint.UNKNOWN,
                visible_elements=[],
                capture_artifact_path=screenshot.artifact_path,
                confidence=0.5,
                elements_reported=False,
            )
        image = prepare_image(Path(screenshot.artifact_path), self.profile.image)
        mapper = CoordinateMapper(
            space="pixels",
            sent_width=image.width,
            sent_height=image.height,
            screen_width=screenshot.width or image.source_width,
            screen_height=screenshot.height or image.source_height,
        )
        input_items = await self._input(conv, state, image)
        tools = [{"type": "computer"}, *_custom_tools(self.environment)]
        instructions = (
            None if conv.previous_response_id else f"You operate a computer to complete a task. {_DONE_RULE}"
        )
        bg_writer.enqueue(prompt_path, json.dumps({"input": _without_images(input_items), "tools": tools}, indent=2))
        try:
            response = await self.client.create(
                input_items=input_items,
                tools=tools,
                previous_response_id=conv.previous_response_id,
                instructions=instructions,
            )
        except ModelTransportError as exc:
            # The conversation is left as it was, so a later step answers the same calls.
            raise BackendCompatibilityError(str(exc)) from exc
        conv.held_checks, conv.not_executed_reason = [], None
        bg_writer.enqueue(raw_path, json.dumps(response, indent=2))
        conv.previous_response_id = response["id"]
        conv.replies += 1
        conv.acknowledged_actions = len(state.action_history)

        output = response.get("output") or []
        calls = [item for item in output if item.get("type") in {"computer_call", "function_call"}]
        text = " ".join(
            part.get("text", "").strip()
            for item in output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") in {"output_text", "text"} and part.get("text", "").strip()
        )
        conv.pending, conv.invalid = calls, None
        actions: list[NeutralAction] = []
        decision: PolicyDecision | None = None
        try:
            for call in calls:
                if call.get("type") == "computer_call":
                    checks = call.get("pending_safety_checks") or []
                    if checks:
                        conv.held_checks = checks
                        conv.held_payload = to_payload(computer_actions(call), mapper)
                        reason = "; ".join(str(c.get("message") or c.get("code") or "safety check") for c in checks)
                        decision = build_policy_decision({
                            "action": {"action_type": "wait_for_user", "text": f"OpenAI safety check: {reason}"[:500]},
                            "rationale": text or "The model asked for confirmation.",
                            "confidence": 0.5,
                            "active_subgoal": "confirm a sensitive action",
                        })
                    actions.extend(computer_actions(call))
                else:
                    actions.append(_function_action(call))
            if decision is None and actions:
                decision = build_policy_decision({
                    "action": to_payload(actions, mapper),
                    "rationale": text or self.profile.name,
                    "confidence": 0.7,
                    "active_subgoal": "; ".join(f"{a.name} {a.args}" for a in actions)[:500],
                })
        except InvalidAction as exc:
            conv.invalid = str(exc)
            conv.invalid_replies += 1
            decision = build_policy_decision({
                "action": {"action_type": "wait", "wait_ms": 500},
                "rationale": f"{INVALID_PREFIX}{exc}",
                "confidence": 0.3,
                "active_subgoal": "reply with valid actions",
            })
        if decision is None:
            if _has_acted(state):
                decision = _stop(text or "Task complete.", None)
            else:
                decision = build_policy_decision({
                    "action": {"action_type": "wait", "wait_ms": 1000},
                    "rationale": text or "No action proposed.",
                    "confidence": 0.4,
                    "active_subgoal": "start the task",
                })
        self._cached_decision = decision

        perception = ScreenPerception(
            summary=text[:2000] or f"{self.environment.title()} screen observed",
            page_hint=PageHint.UNKNOWN,
            visible_elements=[],
            capture_artifact_path=screenshot.artifact_path,
            confidence=0.5,
            elements_reported=False,
        )
        bg_writer.enqueue(parsed_path, json.dumps({
            "profile": self.profile.name,
            "parser": "openai_native",
            "invalid": conv.invalid is not None,
            "invalid_reason": conv.invalid,
            "invalid_replies": conv.invalid_replies,
            "replies": conv.replies,
            "actions": [{"name": a.name, "args": a.args} for a in actions],
            "sent_image": [image.width, image.height],
            "screen": [screenshot.width, screenshot.height],
            "decision": decision.model_dump(mode="json"),
        }))
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
                raise RuntimeError("No cached OpenAI decision available")
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

    def note_not_executed(self, run_id: str, reason: str) -> None:
        conv = self._runs.get(run_id)
        if conv is not None and conv.pending:
            conv.not_executed_reason = reason.strip()[:300] or "blocked by a gate"

    def invalid_reply_counts(self, run_id: str) -> tuple[int, int]:
        conv = self._runs.get(run_id)
        return (conv.invalid_replies, conv.replies) if conv else (0, 0)

    async def _input(self, conv: _Conversation, state: AgentState, image: SentImage) -> list[dict[str, Any]]:
        screenshot_part = {"type": "input_image", "image_url": image.data_url}
        notes = await self._notes(state.run_id)
        if conv.previous_response_id is None:
            task = self._task_template.format(
                intent=state.intent, environment_rules=PromptContext.environment_rules(self.environment)
            )
            text = "\n\n".join(p for p in (task, notes) if p)
            return [{"role": "user", "content": [{"type": "input_text", "text": text}, screenshot_part]}]
        if not conv.pending:
            text = "\n\n".join(p for p in (_CONTINUE, notes) if p)
            return [{"role": "user", "content": [{"type": "input_text", "text": text}, screenshot_part]}]

        outcome = self._outcome(conv, state)
        items: list[dict[str, Any]] = []
        for call in conv.pending:
            if call.get("type") == "computer_call":
                item: dict[str, Any] = {
                    "type": "computer_call_output",
                    "call_id": call.get("call_id"),
                    "output": {"type": "computer_screenshot", "image_url": image.data_url},
                }
                if conv.held_checks and call.get("pending_safety_checks"):
                    item["acknowledged_safety_checks"] = conv.held_checks
                items.append(item)
            else:
                items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": outcome})
        extra = "\n\n".join(p for p in ((outcome if outcome != "OK" else ""), notes) if p)
        if extra:
            items.append({"role": "user", "content": [{"type": "input_text", "text": extra}]})
        return items

    def _outcome(self, conv: _Conversation, state: AgentState) -> str:
        if conv.invalid:
            return f"Error: those actions could not be run ({conv.invalid}), so nothing ran."
        if conv.not_executed_reason:
            reason = conv.not_executed_reason
            return (
                f"Not executed: the run paused for a human before this action ran ({reason}). "
                "Check the screen and propose the next action again."
            )
        if len(state.action_history) > conv.acknowledged_actions:
            executed = state.action_history[-1]
            if not executed.success:
                return f"Error: {executed.detail or 'the action failed'}"
        return "OK"

    async def _notes(self, run_id: str) -> str:
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


def _function_action(call: dict[str, Any]) -> NeutralAction:
    name = call.get("name")
    try:
        arguments = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise InvalidAction(f"the arguments of {name} are not valid JSON") from exc
    if name in {"navigate", "launch_app"} and isinstance(arguments, dict):
        return NeutralAction(name, arguments)
    raise InvalidAction(f"there is no {name!r} tool")


def _has_acted(state: AgentState) -> bool:
    return any(
        executed.action.action_type not in {ActionType.WAIT, ActionType.STOP, ActionType.WAIT_FOR_USER}
        for executed in state.action_history
    )
