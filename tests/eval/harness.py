"""Offline eval harness: run the real agent loop against scripted screens.

A :class:`Scenario` is a small state machine of screens. The harness drives the
real ``AgentLoop`` with its real policy coordinator, rule engine, verifier,
recovery manager, stores and model backend; only the model *client*, the screen
and the executor are fakes:

- the screen is a generated PNG per state (distinct pixels, so screen diffs see
  every transition);
- the executor applies the scenario's transitions to the action it receives;
- the model client returns the scenario's scripted reply for whichever screen is
  showing — it plays an honest model that sees the current screen.

Each run is graded against ground truth (is the goal screen showing at the end?)
rather than against what the run claims, so false successes are measurable.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from PIL import Image, ImageDraw

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.coordinator import PolicyCoordinator
from operon.agent.policy.recovery import RuleBasedRecoveryManager
from operon.agent.policy.trust_gate import TrustGate
from operon.agent.policy.verifier import DeterministicVerifierService
from operon.core.contracts.perception import Environment
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.store.background_writer import bg_writer
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

SCREEN_SIZE = (640, 400)

Backend = Literal["computer_use", "json"]
Outcome = Literal["success", "false_success", "missed", "failure", "incomplete", "crash"]


# ── Scenario definition ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Element:
    """A visible element on a screen, reported by the scripted JSON model."""

    element_id: str
    label: str
    x: int
    y: int
    width: int = 120
    height: int = 32
    element_type: str = "button"

    def contains(self, x: int | None, y: int | None) -> bool:
        if x is None or y is None:
            return False
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def as_perception(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "is_interactable": True,
            "confidence": 0.95,
        }


@dataclass(frozen=True)
class Screen:
    screen_id: str
    summary: str
    elements: tuple[Element, ...] = ()
    page_hint: str = "unknown"
    goal: bool = False

    def element(self, element_id: str) -> Element:
        return next(e for e in self.elements if e.element_id == element_id)


ActionMatcher = Callable[[AgentAction], bool]


@dataclass(frozen=True)
class Transition:
    source: str
    when: ActionMatcher
    target: str


# A scripted model reply: a dict in the backend's wire shape. A list of replies is
# consumed one per call made while that screen is showing; the last one repeats.
Reply = dict[str, Any]


@dataclass
class Scenario:
    name: str
    intent: str
    backend: Backend
    screens: list[Screen]
    transitions: list[Transition]
    model: dict[str, Reply | list[Reply]]
    start: str = ""
    # Critic verdicts per screen (the verifier's model check). None = no critic
    # client, so the verifier uses its deterministic fallback.
    critic: dict[str, Reply] | None = None
    max_steps: int = 12
    # What the scripted human does at a WAIT_FOR_USER pause.
    on_pause: Literal["approve", "stop"] = "stop"
    max_resumes: int = 3
    trust_gate: TrustGate | None = None
    # The outcome a correct agent gets. ``known_issue`` marks a scenario the agent
    # currently gets wrong; the suite expects it to fail until the issue is fixed.
    expected: Outcome | tuple[Outcome, ...] = "success"
    known_issue: str | None = None
    description: str = ""

    def screen(self, screen_id: str) -> Screen:
        return next(s for s in self.screens if s.screen_id == screen_id)

    def accepts(self, outcome: Outcome) -> bool:
        expected = self.expected if isinstance(self.expected, tuple) else (self.expected,)
        return outcome in expected

    @property
    def start_screen(self) -> str:
        return self.start or self.screens[0].screen_id


# ── Action matchers ───────────────────────────────────────────────────────


def clicks(element: Element) -> ActionMatcher:
    return lambda a: a.action_type in {ActionType.CLICK, ActionType.DOUBLE_CLICK} and element.contains(a.x, a.y)


def hovers(element: Element) -> ActionMatcher:
    return lambda a: a.action_type is ActionType.HOVER and element.contains(a.x, a.y)


def types(text: str) -> ActionMatcher:
    return lambda a: a.action_type is ActionType.TYPE and text.lower() in (a.text or "").lower()


# ── Scripted replies ──────────────────────────────────────────────────────


def json_reply(
    action: dict[str, Any],
    *,
    confidence: float = 0.85,
    subgoal: str = "advance",
    rationale: str = "scripted",
) -> Reply:
    """A combined (JSON backend) reply; perception is filled in from the screen."""
    return {"action": action, "confidence": confidence, "active_subgoal": subgoal, "rationale": rationale}


def cu_call(name: str, **args: Any) -> Reply:
    """A Gemini Computer Use function call. Pixel coordinates in ``x``/``y`` are
    converted to the model's 0-999 grid."""
    width, height = SCREEN_SIZE
    if "x" in args:
        args["x"] = round(args["x"] * 1000 / width)
    if "y" in args:
        args["y"] = round(args["y"] * 1000 / height)
    return {"function_call": {"name": name, "args": args}}


def cu_text(text: str) -> Reply:
    """A text-only Computer Use reply (how the model says it is done)."""
    return {"text": text}


def critic_verdict(status: str, *, stop: bool = False, reason: str = "scripted critic") -> Reply:
    return {
        "status": status,
        "expected_outcome_met": status == "success",
        "stop_condition_met": stop,
        "reason": reason,
    }


# ── Fakes ─────────────────────────────────────────────────────────────────


class _World:
    """The scenario's live state: current screen, visit counts, rendered frames."""

    _PALETTE = [(240, 240, 240), (200, 225, 250), (250, 225, 200), (210, 245, 210), (245, 210, 235), (230, 230, 180)]

    def __init__(self, scenario: Scenario, frames_dir: Path) -> None:
        self.scenario = scenario
        self.current = scenario.start_screen
        self.visits: Counter[str] = Counter({self.current: 1})
        self._calls: Counter[tuple[int, str]] = Counter()
        self.model_calls = 0
        self.critic_calls = 0
        self._frames_dir = frames_dir
        self._seq = 0
        self._images = {
            screen.screen_id: self._render(screen, self._PALETTE[i % len(self._PALETTE)])
            for i, screen in enumerate(scenario.screens)
        }

    @property
    def screen(self) -> Screen:
        return self.scenario.screen(self.current)

    def apply(self, action: AgentAction) -> bool:
        """Apply the first matching transition; return True if the screen changed."""
        for transition in self.scenario.transitions:
            if transition.source == self.current and transition.when(action):
                self.current = transition.target
                self.visits[self.current] += 1
                return True
        return False

    def reply(self, table: dict[str, Reply | list[Reply]]) -> Reply:
        replies = table.get(self.current)
        if replies is None:
            raise KeyError(f"scenario {self.scenario.name!r} has no scripted reply for screen {self.current!r}")
        if isinstance(replies, list):
            key = (id(table), self.current)
            self._calls[key] += 1
            return replies[min(self._calls[key], len(replies)) - 1]
        return replies

    def write_frame(self) -> str:
        self._seq += 1
        path = self._frames_dir / f"frame_{self._seq}.png"
        self._images[self.current].save(path)
        return str(path)

    @staticmethod
    def _render(screen: Screen, background: tuple[int, int, int]) -> Image.Image:
        image = Image.new("RGB", SCREEN_SIZE, background)
        draw = ImageDraw.Draw(image)
        draw.text((16, 12), screen.summary[:80], fill=(20, 20, 20))
        for element in screen.elements:
            box = (element.x, element.y, element.x + element.width, element.y + element.height)
            draw.rectangle(box, outline=(30, 30, 30), fill=(255, 255, 255))
            draw.text((element.x + 6, element.y + 8), element.label[:18], fill=(0, 0, 0))
        return image


class FakeExecutor:
    """Executor over the scenario world. Every action 'succeeds' mechanically; whether
    it did anything useful shows up in the next screen, as on a real machine."""

    def __init__(self, world: _World) -> None:
        self._world = world
        self.actions: list[AgentAction] = []

    async def capture(self) -> CaptureFrame:
        width, height = SCREEN_SIZE
        return CaptureFrame(artifact_path=self._world.write_frame(), width=width, height=height)

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.actions.append(action)
        changed = self._world.apply(action)
        return ExecutedAction(
            action=action,
            success=True,
            detail="screen changed" if changed else "no visible change",
            artifact_path=self._world.write_frame(),
        )

    async def context_reset(self) -> None:
        return None

    async def session_reset(self, start_url: str | None = None) -> None:
        return None

    async def get_current_url(self) -> str | None:
        return None


class FakeComputerUseClient:
    """Stands in for ``GeminiComputerUseHttpClient``; returns the payload shape its
    ``generate_turn`` normalizes Gemini responses into."""

    def __init__(self, world: _World) -> None:
        self._world = world
        self.requests: list[list[dict]] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.requests.append(list(contents))
        self._world.model_calls += 1
        return self._payload(self._world.reply(self._world.scenario.model))

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        response = {"url": current_url} if current_url else {}
        if error:
            response["error"] = error
        return {
            "role": "user",
            "parts": [
                {"function_response": {"name": function_name, "response": response}},
                {"inline_data": {"mime_type": "image/png", "data": ""}},
            ],
        }

    @staticmethod
    def _payload(reply: Reply) -> dict:
        if "text" in reply:
            text = reply["text"]
            return {
                "perception": {"summary": text, "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
                "action": {"action_type": "wait", "wait_ms": 1000},
                "rationale": text,
                "confidence": 0.4,
                "active_subgoal": "gather more browser context",
                "model_content": {"parts": [{"text": text}]},
            }
        call = reply["function_call"]
        return {
            "perception": {
                "summary": f"Tool call: {call['name']}.",
                "page_hint": "unknown",
                "visible_elements": [],
                "confidence": 0.5,
            },
            "function_call": call,
            "function_calls": [call],
            "rationale": call["name"],
            "confidence": 0.7,
            "active_subgoal": call["name"],
            "model_content": {"parts": [{"function_call": call}]},
        }


class FakeGeminiClient:
    """Stands in for ``GeminiHttpClient`` for the combined JSON backend."""

    def __init__(self, world: _World) -> None:
        self._world = world
        self.prompts: list[str] = []

    async def generate_perception(self, prompt: str, screenshot_path: str) -> str:
        self.prompts.append(prompt)
        self._world.model_calls += 1
        reply = self._world.reply(self._world.scenario.model)
        screen = self._world.screen
        return json.dumps({
            "perception": {
                "summary": screen.summary,
                "page_hint": screen.page_hint,
                "visible_elements": [e.as_perception() for e in screen.elements],
                "confidence": 0.9,
            },
            **reply,
        })


class FakeCriticClient:
    def __init__(self, world: _World) -> None:
        self._world = world

    async def generate_verification(self, prompt: str, screenshot_path: str) -> str:
        self._world.critic_calls += 1
        return json.dumps(self._world.reply(self._world.scenario.critic or {}))


# ── Running and grading ───────────────────────────────────────────────────


@dataclass
class RunResult:
    scenario: str
    outcome: Outcome
    status: str
    stop_reason: str | None
    steps: int
    pauses: int
    model_calls: int
    critic_calls: int
    final_screen: str
    error: str | None = None
    actions: list[str] = field(default_factory=list)
    model_requests: list[Any] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.scenario}: {self.outcome} (status={self.status}, stop={self.stop_reason}, "
            f"steps={self.steps}, pauses={self.pauses}, screen={self.final_screen}, actions={self.actions}"
            + (f", error={self.error}" if self.error else "")
            + ")"
        )


def grade(*, claimed_success: bool, reached_goal: bool, ended: bool) -> Outcome:
    if claimed_success:
        return "success" if reached_goal else "false_success"
    if reached_goal:
        return "missed"
    return "failure" if ended else "incomplete"


def _build_loop(scenario: Scenario, world: _World, root: Path) -> tuple[AgentLoop, FakeExecutor, Any]:
    executor = FakeExecutor(world)
    if scenario.backend == "computer_use":
        model_client: Any = FakeComputerUseClient(world)
        backend: Any = BrowserComputerUseBackend(
            client=model_client,
            prompt_path=prompts_dir() / "browser_computer_use_prompt.txt",
            browser_runtime=executor,
        )
        environment = Environment.BROWSER
    else:
        model_client = FakeGeminiClient(world)
        backend = CombinedPerceptionPolicyService(
            gemini_client=model_client, prompt_path=prompts_dir() / "desktop_combined_prompt.txt"
        )
        environment = Environment.DESKTOP
    run_store = FileBackedRunStore(root / "runs")
    memory_store = FileBackedMemoryStore(root / "memory")
    critic: Any = FakeCriticClient(world) if scenario.critic is not None else object()
    loop = AgentLoop(
        capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
        perception_service=backend,
        run_store=run_store,
        policy_service=PolicyCoordinator(delegate=backend, memory_store=memory_store),
        executor=executor,
        verifier_service=DeterministicVerifierService(gemini_client=critic),
        recovery_manager=RuleBasedRecoveryManager(),
        memory_store=memory_store,
        environment=environment,
    )
    if scenario.trust_gate is not None:
        loop._trust_gate = scenario.trust_gate
    return loop, executor, model_client


async def _no_sleep(*_args: Any, **_kwargs: Any) -> None:
    return None


_TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


async def run_scenario(scenario: Scenario) -> RunResult:
    """Run one scenario to completion (or its step budget) and grade it."""
    with tempfile.TemporaryDirectory(prefix=f"operon-eval-{scenario.name}-") as tmp, ExitStack() as stack:
        root = Path(tmp)
        (root / "frames").mkdir()
        stack.enter_context(patch.dict(os.environ, {"OPERON_TEST_SAFE_MODE": "true"}))
        # The loop's settle/backoff waits are wall-clock sleeps; the fake screen
        # never animates, so skip them.
        stack.enter_context(patch("operon.agent.loop.asyncio.sleep", _no_sleep))
        # A pause must not reach the desktop, a webhook or a background timer.
        stack.enter_context(patch("operon.agent.hitl.notify_desktop", lambda **_: None))
        stack.enter_context(patch("operon.agent.hitl.post_hitl_webhook", _no_sleep))
        stack.enter_context(patch("operon.agent.hitl.start_escalation_timer", _no_sleep))
        world = _World(scenario, root / "frames")
        loop, executor, model_client = _build_loop(scenario, world, root)

        run = await loop.start_run(RunTaskRequest(intent=scenario.intent, max_steps=scenario.max_steps))
        run_id = run.run_id
        status = run.status
        pauses = resumes = 0
        steps = 0
        error: str | None = None
        while steps < scenario.max_steps:
            try:
                response = await loop.step_run(StepRequest(run_id=run_id))
            except Exception as exc:  # the API marks such runs failed; here it is its own outcome
                error = f"{type(exc).__name__}: {exc}"
                break
            steps = response.step_count
            status = response.status
            if status is RunStatus.WAITING_FOR_USER:
                pauses += 1
                if scenario.on_pause == "approve" and resumes < scenario.max_resumes:
                    resumes += 1
                    await loop.resume_run(run_id)
                    continue
                break
            if status in _TERMINAL:
                break

        state = await loop.run_store.get_run(run_id)
        await bg_writer.flush()  # artifact writes land in the temp dir; finish them before it goes
        requests = getattr(model_client, "requests", None) or getattr(model_client, "prompts", [])
        return RunResult(
            scenario=scenario.name,
            outcome="crash" if error else grade(
                claimed_success=status is RunStatus.SUCCEEDED,
                reached_goal=world.screen.goal,
                ended=status in _TERMINAL,
            ),
            status=status.value,
            stop_reason=state.stop_reason.value if state is not None and state.stop_reason else None,
            steps=steps,
            pauses=pauses,
            model_calls=world.model_calls,
            critic_calls=world.critic_calls,
            final_screen=world.current,
            error=error,
            actions=[a.action_type.value for a in executor.actions],
            model_requests=requests,
        )
