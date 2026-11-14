from __future__ import annotations

import base64
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from operon.agent import tools as _tools
from operon.agent._agent_utils import collect_latest_usage
from operon.agent.actions import zoom as _zoom
from operon.agent.actions.translation import (
    KeyHold,
    build_policy_decision,
    normalize_computer_use_actions,
)
from operon.agent.backends.base import AgentBackend
from operon.agent.backends.fallback import BackendCompatibilityError
from operon.clients.gemini_computer_use import (
    GeminiComputerUseClient,
    GeminiComputerUseError,
)
from operon.models.capture import CaptureFrame
from operon.models.execution import ToolOutput
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, PolicyDecision
from operon.models.state import AgentState
from operon.store.background_writer import bg_writer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ConversationState:
    contents: list[dict] = field(default_factory=list)
    pending_function_names: list[str] = field(default_factory=list)
    acknowledged_actions: int = 0
    # Calls the model marked require_confirmation: the run paused on them, and the
    # next perceive (only reached after the user resumes) runs them as approved.
    held_calls: list[dict] = field(default_factory=list)
    acknowledge_safety: bool = False
    # Set when the user declined the held calls: the next answer to the model's
    # open calls reports this instead of an outcome, since they never ran.
    declined_reason: str | None = None
    # Keys the model is holding down (key_down without its key_up yet); they join
    # its next key press, which may come several turns later.
    key_hold: KeyHold = field(default_factory=KeyHold)
    # The hold as the open calls would leave it. It replaces ``key_hold`` only once
    # those calls ran: a batch that failed part-way, or calls dropped by a re-look,
    # must not leave a key held that the model never got to press.
    pending_hold: KeyHold | None = None
    # The calls of the model's latest turn, until they run or are set aside: a gate
    # pause holds them so the person's approval runs exactly what the model asked.
    last_calls: list[dict] = field(default_factory=list)
    # The model's stated reason for ``last_calls`` / ``held_calls``: replaying held
    # calls keeps it, so a later gate (the trust gate reads it) still sees it.
    last_rationale: str | None = None
    held_rationale: str | None = None
    # Why the open calls did not run when Operon set them aside without asking the
    # person (a rule ran something else, or a guard rejected the call); the next
    # answer reports it instead of an outcome.
    not_run_reason: str | None = None
    # Why the last attempt failed when it never reached the action history (an
    # in-step retry asks the model again before the step is recorded).
    attempt_error: str | None = None


def _requires_confirmation(call: dict) -> bool:
    decision = (call.get("args") or {}).get("safety_decision")
    return isinstance(decision, dict) and decision.get("decision") == "require_confirmation"


def _without_safety_decision(call: dict) -> dict:
    args = {key: value for key, value in (call.get("args") or {}).items() if key != "safety_decision"}
    return {**call, "args": args}


class BrowserComputerUseBackend(AgentBackend):
    """Browser backend for Google Computer Use models.

    The request/response protocol is intentionally isolated from the existing
    JSON backend. Until the transport is fully implemented, protocol-level
    issues are surfaced as compatibility errors so the route-configured JSON
    fallback can take over.
    """

    _retry_suffix = (
        "\n\nRetry guidance:\n"
        "- A text-only response is not enough when a harmless browser tool action is available.\n"
        "- Prefer one concrete tool call that reveals more page structure.\n"
        "- If uncertain, choose a low-risk action like hover_at or scroll_at instead of waiting.\n"
    )

    def __init__(
        self,
        *,
        client: GeminiComputerUseClient,
        prompt_path: Path,
        browser_runtime=None,
        tools: tuple[str, ...] = (),
    ) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.browser_runtime = browser_runtime
        # The tools the client declares (RFC 0002); their skill notes join the prompt.
        self.tools = tuple(tools)
        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        self._skill_text = _tools.skill_text(self.tools)
        self._cached_decisions: dict[str, PolicyDecision] = {}
        self._perception_only = False
        self._last_frames: dict[str, CaptureFrame] = {}
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._advisory_hints: dict[str, list[tuple[str, str]]] = {}
        self._conversations: dict[str, _ConversationState] = {}
        # Zoom before clicking on dense screens (KAN-82, OPERON_ZOOM; off by default).
        self.zoom = _zoom.zoom_settings()

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        self._last_frames[state.run_id] = screenshot
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "computer_use_prompt.txt"
        raw_path = step_dir / "computer_use_raw.json"
        parsed_path = step_dir / "computer_use_parsed.json"

        if self._perception_only:
            # A mid-step re-look (an execution retry, a page-load wait): the screen
            # is all the loop needs. A model turn here would answer the open call
            # before it was recorded and replace the step's decision.
            return self._build_perception(payload={}, screenshot_path=screenshot.artifact_path)
        zoom_usage = None

        conversation = self._conversations.get(state.run_id)
        if conversation is not None and conversation.held_calls:
            # The run paused before these calls ran (the model's require_confirmation,
            # or an Operon gate) and the person approved: run them now, with no
            # model turn. Queued hints stay queued for the next turn.
            bg_writer.enqueue(prompt_path, "No model call: running the calls the person approved.\n")
            held = conversation.held_calls
            function_calls = [_without_safety_decision(call) for call in held]
            conversation.held_calls = []
            conversation.last_calls = list(held)
            rationale = conversation.held_rationale or "Running the action the user confirmed."
            conversation.last_rationale = conversation.held_rationale
            conversation.held_rationale = None
            # The API rejects the answer to a require_confirmation call unless it
            # acknowledges the decision, and only those calls carry one.
            conversation.acknowledge_safety = any(_requires_confirmation(call) for call in held)
            response_payload = {
                "function_call": function_calls[0],
                "function_calls": function_calls,
                "rationale": rationale,
            }
        else:
            hints = self._take_hints(state)
            prompt = self._render_prompt(state, hints)
            bg_writer.enqueue(prompt_path, prompt)
            response_payload = await self._run_with_retry(
                prompt=prompt, screenshot=screenshot, state=state, hints=hints
            )
            if self.zoom.enabled:
                response_payload, zoom_usage = await self._zoom_first_click(response_payload, screenshot, state)
            function_calls = response_payload.get("function_calls") or []
            conversation = self._conversations[state.run_id]
            conversation.last_calls = list(function_calls)
            conversation.last_rationale = response_payload.get("rationale")
            if any(_requires_confirmation(call) for call in function_calls):
                conversation.held_calls = function_calls
                conversation.held_rationale = conversation.last_rationale

        bg_writer.enqueue(raw_path, json.dumps(response_payload, indent=2))

        perception_payload = response_payload.get("perception") or {}
        perception = self._build_perception(
            payload=perception_payload,
            screenshot_path=screenshot.artifact_path,
        )
        if "function_call" in response_payload:
            function_calls = response_payload.get("function_calls") or [response_payload["function_call"]]
            conversation = self._conversations.get(state.run_id)
            hold = conversation.key_hold.copy() if conversation is not None else KeyHold()
            response_payload = {
                **response_payload,
                "action": normalize_computer_use_actions(
                    function_calls,
                    screen_width=screenshot.width,
                    screen_height=screenshot.height,
                    key_hold=hold,
                ),
                "active_subgoal": self._subgoal_for_calls(function_calls),
            }
            if conversation is not None:
                conversation.pending_hold = hold
        elif self._is_completion_text(response_payload, state):
            # A text-only reply after the model has acted is how Computer Use says
            # the task is done. Surface it as STOP so the verifier judges the claim.
            response_payload = {
                **response_payload,
                "action": {"action_type": ActionType.STOP.value},
                "active_subgoal": "complete task",
            }
        decision = build_policy_decision(response_payload)
        self._cached_decisions[state.run_id] = decision
        bg_writer.enqueue(
            parsed_path,
            json.dumps(
                {
                    "perception": perception.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                }
            ),
        )
        usage_path = step_dir / "computer_use_usage.json"
        if zoom_usage is None:
            usage = collect_latest_usage(self.client, usage_path)
        else:
            # The client only remembers the zoom request; the step's turn is the one before it.
            usage = _zoom.merge_usage(zoom_usage[0], zoom_usage[1])
            if usage is not None:
                bg_writer.enqueue(usage_path, usage.model_dump_json(indent=2))
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(parsed_path),
            usage_artifact_path=str(usage_path),
            usage=usage,
        )
        return perception

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        if state.run_id not in self._cached_decisions:
            # perceive() was skipped (the loop reused a prior perception) or the
            # coordinator is asking for a second decision this step: take a fresh
            # turn on the current screenshot rather than failing the step.
            frame = self._last_frames.get(state.run_id)
            if frame is None or not perception.capture_artifact_path:
                raise RuntimeError("No cached computer-use decision available")
            await self.perceive(frame.model_copy(update={"artifact_path": perception.capture_artifact_path}), state)
        return self._cached_decisions.pop(state.run_id)

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def has_held_actions(self, run_id: str) -> bool:
        conversation = self._conversations.get(run_id)
        return bool(conversation is not None and conversation.held_calls)

    def discard_held_actions(self, run_id: str, reason: str) -> None:
        conversation = self._conversations.get(run_id)
        if conversation is None or not conversation.held_calls:
            return
        conversation.held_calls = []
        conversation.held_rationale = None
        conversation.last_calls = []
        conversation.acknowledge_safety = False
        conversation.declined_reason = reason

    def note_not_executed(self, run_id: str, reason: str) -> None:
        """An Operon gate paused the run before the model's calls ran. Hold them:
        the person's approval runs exactly these calls, and a correction declines
        them (``discard_held_actions``), so the model never hears them reported as
        done. Calls already set aside (a rule ran something else) are not held."""
        conversation = self._conversations.get(run_id)
        if conversation is None or not conversation.pending_function_names:
            return
        if conversation.held_calls:
            return  # Already held for the model's own confirmation request.
        if conversation.last_calls and conversation.not_run_reason is None:
            conversation.held_calls = list(conversation.last_calls)
            conversation.held_rationale = conversation.last_rationale
            return
        conversation.not_run_reason = f"Not executed: {reason}"

    def note_replaced(self, run_id: str, action: str) -> None:
        """Operon's harness ran ``action`` instead of the model's calls."""
        self._set_not_run(run_id, f"Not executed: Operon ran {action} instead.")

    def note_rejected(self, run_id: str, reason: str) -> None:
        """Operon's harness dropped the model's calls without running them."""
        self._set_not_run(run_id, f"Not executed: {reason}.")

    def note_attempt_failed(self, run_id: str, detail: str) -> None:
        """The step's attempt failed before it was recorded; the next answer to the
        model's calls reports it."""
        conversation = self._conversations.get(run_id)
        if conversation is not None and detail:
            conversation.attempt_error = detail

    def set_perception_only(self, value: bool) -> None:
        """When True, perceive() only wraps the screenshot: no model turn, and the
        conversation and the step's decision are left as they are."""
        self._perception_only = value

    def _set_not_run(self, run_id: str, reason: str) -> None:
        conversation = self._conversations.get(run_id)
        if conversation is None or not conversation.pending_function_names:
            return
        conversation.not_run_reason = reason
        conversation.last_calls = []
        conversation.pending_hold = None

    def supports_resampling(self, run_id: str = "") -> bool:
        # The decision is a function call in a stateful Computer Use conversation;
        # there is no second sample for the same turn.
        return False

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state. Test use only."""
        self._advisory_hints = {"": [(h, "") for h in hints if h]}

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        """Append hints without discarding hints set by other writers."""
        incoming = [(h, source) for h in hints if h]
        bucket = self._advisory_hints.setdefault(run_id, [])
        bucket.extend(incoming)
        logger.debug(
            "add_advisory_hints(%s): run=%r source=%r incoming=%d total=%d",
            self.__class__.__name__, run_id, source, len(incoming), len(bucket),
        )

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self._advisory_hints.pop(run_id, None)

    async def _zoom_first_click(
        self, response_payload: dict, screenshot: CaptureFrame, state: AgentState
    ) -> tuple[dict, tuple | None]:
        """Refine the turn's first click on a zoomed crop (OPERON_ZOOM). Returns the
        payload, with the click moved if zoom ran, and (turn usage, zoom usage) when a
        zoom request was made. The conversation keeps the model's original call."""
        function_calls = response_payload.get("function_calls") or []
        if not function_calls:
            return response_payload, None
        turn_usage = self.client.latest_usage() if hasattr(self.client, "latest_usage") else None
        result = await _zoom.refine_click(
            client=self.client,
            settings=self.zoom,
            function_calls=function_calls,
            screenshot_path=screenshot.artifact_path,
            width=screenshot.width,
            height=screenshot.height,
            intent=state.intent,
            rationale=str(response_payload.get("rationale") or ""),
        )
        if result.detail:
            logger.info("%s", result.detail)
        zoom_usage = result.usage
        if result.requested and zoom_usage is None and hasattr(self.client, "latest_usage"):
            # The zoom request failed after the client recorded its usage (for example a
            # reply it could not parse): count that, and never lose the turn's own usage.
            latest = self.client.latest_usage()
            zoom_usage = latest if latest is not turn_usage else None
        usages = (turn_usage, zoom_usage) if result.requested else None
        if not result.zoomed:
            return response_payload, usages
        return (
            {
                **response_payload,
                "function_call": result.function_calls[0],
                "function_calls": result.function_calls,
                "zoom": result.detail,
            },
            usages,
        )

    async def _run_with_retry(
        self, *, prompt: str, screenshot: CaptureFrame, state: AgentState, hints: list[tuple[str, str]] | None = None
    ) -> dict:
        try:
            response_payload = await self._run_turn(prompt=prompt, screenshot=screenshot, state=state, hints=hints)
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

    async def _run_turn(
        self, *, prompt: str, screenshot: CaptureFrame, state: AgentState, hints: list[tuple[str, str]] | None = None
    ) -> dict:
        conversation = self._conversations.setdefault(state.run_id, _ConversationState())
        if conversation.pending_function_names:
            # Always answer the model's open tool call before asking again: the API
            # rejects a request that ends with a model turn. A mid-step re-look (an
            # execution or adaptation retry) arrives before the action is recorded,
            # so the answer is just the current screen.
            current_url = None
            if self.browser_runtime is not None and hasattr(self.browser_runtime, "get_current_url"):
                current_url = await self.browser_runtime.get_current_url()
            error = None
            tool_outputs: list[ToolOutput] = []
            declined = conversation.declined_reason is not None
            not_run = conversation.not_run_reason
            conversation.not_run_reason = None
            attempt_error, conversation.attempt_error = conversation.attempt_error, None
            if declined:
                error = conversation.declined_reason
                conversation.declined_reason = None
            elif not_run is not None:
                error = not_run
            elif attempt_error is not None and len(state.action_history) <= conversation.acknowledged_actions:
                error = attempt_error
            elif len(state.action_history) > conversation.acknowledged_actions and not state.action_history[-1].success:
                error = state.action_history[-1].detail
            if not declined and not_run is None and len(state.action_history) > conversation.acknowledged_actions:
                tool_outputs = list(state.action_history[-1].tool_outputs or [])
            ran = (
                not declined
                and not_run is None
                and len(state.action_history) > conversation.acknowledged_actions
                and state.action_history[-1].success
            )
            hold, conversation.pending_hold = conversation.pending_hold, None
            if ran and hold is not None:
                conversation.key_hold = KeyHold(keys=hold.keys)
                if hold.skipped:
                    error = (
                        f"Not run: {'; '.join(hold.skipped)}. Clicking or scrolling with a key held "
                        "is not supported here; release the key (key_up) and use a hotkey or a plain click."
                    )
            function_response = self._build_function_response_content(
                function_names=conversation.pending_function_names,
                screenshot_path=screenshot.artifact_path,
                current_url=current_url,
                error=error,
                tool_outputs=tool_outputs,
            )
            if conversation.acknowledge_safety:
                # The API rejects the answer to a require_confirmation call unless
                # it acknowledges the decision the user just made.
                for part in function_response["parts"]:
                    if "function_response" in part:
                        part["function_response"]["response"]["safety_acknowledgement"] = "true"
                conversation.acknowledge_safety = False
            if declined:
                # The user resumed with a correction instead: send it with the
                # decline, since this turn is otherwise only the function response.
                function_response["parts"].append({"text": prompt})
            elif hints:
                # Operon's notes for this step (the person's override or hand-back
                # note, memory, a stagnation hint) ride along with the answer; the
                # full prompt went out on the first turn.
                function_response["parts"].append({"text": self._hint_text(hints)})
            conversation.contents.append(function_response)
            conversation.acknowledged_actions = len(state.action_history)
            conversation.pending_function_names = []
            conversation.last_calls = []
        elif not conversation.contents or (
            not conversation.pending_function_names and conversation.contents[-1].get("role") == "model"
        ):
            # First turn, or the model's last turn had no tool call to answer:
            # show it the current screen so the conversation keeps alternating.
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

    def _take_hints(self, state: AgentState) -> list[tuple[str, str]]:
        """Pop the hints queued for this run (or the shared test bucket)."""
        hints = self._advisory_hints.pop(state.run_id, None) or self._advisory_hints.pop("", None) or []
        if hints:
            _counts: dict[str, int] = {}
            for _, _src in hints:
                _label = _src or "unknown"
                _counts[_label] = _counts.get(_label, 0) + 1
            logger.debug("hints consumed (%s): [%s]", self.__class__.__name__, ", ".join(f"{k}:{v}" for k, v in _counts.items()))
        return hints

    @staticmethod
    def _hint_text(hints: list[tuple[str, str]]) -> str:
        return "Notes from Operon for this step:\n" + "\n".join(f"- {h}" for h, _ in hints)

    def _render_prompt(self, state: AgentState, hints: list[tuple[str, str]] | None = None) -> str:
        hint_block = "\n".join(f"- {h}" for h, _ in hints) if hints else "none"
        return self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            previous_summary=state.observation_history[-1].summary if state.observation_history else "none",
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            advisory_hints=hint_block,
        ) + self._skill_text

    @staticmethod
    def _subgoal_for_calls(function_calls: list[dict]) -> str:
        """Name the step by the call and its arguments, not the bare tool name.

        The progress tracker blocks a subgoal already completed on the same page,
        and this backend's page signature never changes, so a bare tool name would
        block every second ``type_text_at`` (a form's second field) for the run.
        """
        described = []
        for call in function_calls:
            args = call.get("args") or {}
            details = " ".join(f"{key}={args[key]}" for key in sorted(args) if key not in {"safety_decision", "intent"})
            described.append(f"{call.get('name', 'browser step')} {details}".strip())
        return "; ".join(described)

    @staticmethod
    def _is_completion_text(response_payload: dict, state: AgentState) -> bool:
        if "function_call" in response_payload:
            return False
        if (response_payload.get("action") or {}).get("action_type") != "wait":
            return False
        return any(
            executed.success and executed.action.action_type not in {ActionType.WAIT, ActionType.STOP, ActionType.WAIT_FOR_USER}
            for executed in state.action_history
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
            elements_reported=False,
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
        tool_outputs: list[ToolOutput] | None = None,
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
        self._add_tool_outputs(parts, tool_outputs or [], error)
        return {"role": "user", "parts": parts}

    @staticmethod
    def _add_tool_outputs(parts: list[dict], tool_outputs: list[ToolOutput], error: str | None) -> None:
        """Answer each tool call with its own output (RFC 0002), in call order. A tool
        call with no output did not run, because an earlier call in the turn failed."""
        outputs = iter(tool_outputs)
        for part in parts:
            response = part.get("function_response")
            if not response or response.get("name") not in _tools.TOOL_NAMES:
                continue
            payload = response.setdefault("response", {})
            result = next(outputs, None)
            if result is None:
                payload["error"] = f"Not run: {error}" if error else "This call did not run."
            elif result.ok:
                payload.pop("error", None)
                payload["output"] = result.output
            else:
                payload["error"] = result.output


