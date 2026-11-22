"""Record a real Gemini run and replay it through the real loop for free (KAN-167).

**Record.** A paid benchmark run (a batch run, KAN-165) already saves, for each
step, Gemini's answer (``computer_use_raw.json``) and the action Operon then ran
(``run.jsonl``). :func:`record_run` folds one run folder into a small recording: the
task, the settings that shape the loop, every model answer in order and every action
Operon ran. ``python -m evals.live_bench --keep-screens`` writes one per thin-harness
Gemini computer-use task to ``<report>/recordings/<task>.json``, so a batch
artifact carries them.

**Replay.** :func:`replay` runs the recorded task again through the real
``AgentLoop``, the real computer-use backend and the thin harness, with Gemini
replaced by the recorded answers and the screen by a blank frame of the recorded
size. Nothing is paid and nothing is clicked. The loop must run the same actions in
the same order as it did for real; when a code change makes it do something else
with the same model answers, the replay says at which step and what changed.

Recordings that CI replays live in ``evals/recordings/`` (``tests/evals/test_replay.py``).
Add one by copying it there from a batch artifact. Their screenshots are never
kept: a recording holds only the model's text and actions.

    python -m evals.replay record .var/runs/<run_id> --task NAME --mode desktop > rec.json
    python -m evals.replay check evals/recordings/*.json

Only thin-harness Gemini computer-use runs replay (the full harness also asked a
Gemini critic, which is not recorded). A Claude or GPT run (``OPERON_*_BACKEND=model``,
KAN-168) is refused when recorded and when replayed: its answers are not saved per turn
in Gemini's shape, so the Gemini backend would replay nonsense. A ``hybrid`` run (a
desktop app and a browser on one machine) ran through the desktop loop, so it replays
through the desktop path; the recording keeps the ``hybrid`` label (KAN-178).
A replay patches settle detection's sleep and clock with a fake clock, so any settle
wait during a replay takes no real time.
"""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any
from unittest.mock import patch

from PIL import Image

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
FORMAT = 1
# Settings a recording does not carry: secrets, storage paths, hosts and URLs of the
# machine that ran it, and test-only switches (OPERON_ALLOWED_FILE_ROOTS is kept: it
# decides which file actions need approval). Every other setting the run had (trust gate,
# permission mode, tools, zoom, platform, ...) is restored for the replay.
_SETTINGS_SKIPPED = re.compile(
    r"(KEY|KEYS|TOKEN|ROOT|DIR|URL|BUNDLE|IMAGE|HOSTS|ORIGINS|VERTEX|TRACE|"
    r"TEST_SAFE_MODE|STRICT_ERRORS|ALLOW_UNAUTHENTICATED|ENVIRONMENT|TIMEOUT|RETAIN_DAYS|TTL_S)$|VERTEX"
)
# The action fields compared between the recording and the replay.
_ACTION_FIELDS = (
    "action_type", "x", "y", "x_end", "y_end", "text", "key", "url", "press_enter",
    "clear_before_typing", "scroll_amount", "horizontal", "wait_ms", "tool", "tool_args",
)
_NO_MODEL_CALL = "No model call"
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class RecordingError(ValueError):
    """The run folder cannot be turned into a recording that replays."""


# The recording modes; a hybrid run used the desktop loop (KAN-178).
MODES = ("browser", "desktop", "hybrid")


def loop_mode(mode: str) -> str:
    """The loop a recording of ``mode`` runs through: hybrid runs used the desktop loop."""
    if mode == "hybrid":
        return "desktop"
    if mode in ("browser", "desktop"):
        return mode
    raise RecordingError(f"unknown recording mode {mode!r}; use browser, desktop or hybrid")


def _require_gemini(settings: dict, mode: str) -> None:
    """Refuse a Claude or GPT run (KAN-168 model adapter): only Gemini answers replay.
    Either backend key counts, so a run recorded under the other mode is refused too."""
    loop_mode(mode)
    for key in ("OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND"):
        if str(settings.get(key) or "").strip().lower() != "model":
            continue
        profile = str(settings.get("OPERON_MODEL_PROFILE") or "").strip() or "?"
        raise RecordingError(
            f"only Gemini computer-use runs replay: this run used {key}=model with model profile "
            f"{profile} (Claude and GPT answers are not recorded per turn)")


# ── record ──────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_logs(run_dir: Path) -> list[dict]:
    path = run_dir / "run.jsonl"
    if not path.is_file():
        raise RecordingError(f"{run_dir}: no run.jsonl")
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "step_index" in record and "executed_action" in record:
            steps.append(record)
    return sorted(steps, key=lambda r: r["step_index"])


def compact_action(action: dict | None) -> dict | None:
    """The fields of an action that the replay compares (children of a batch too)."""
    if action is None:
        return None
    out = {k: action[k] for k in _ACTION_FIELDS if action.get(k) is not None}
    if action.get("actions"):
        out["actions"] = [compact_action(a) for a in action["actions"]]
    return out


def _local(path: str | None, run_dir: Path) -> Path | None:
    """An artifact path from run.jsonl, found again inside ``run_dir`` (the folder may
    have been copied out of the runs root, e.g. into a benchmark artifact, and a
    Windows run's paths may be read on Linux)."""
    if not path:
        return None
    original = Path(path)
    if original.is_file():
        return original
    parts = PureWindowsPath(path).parts if "\\" in path else original.parts
    for i, part in enumerate(parts):
        if part.startswith("step_"):
            candidate = run_dir.joinpath(*parts[i:])
            if candidate.is_file():
                return candidate
    return None


def recordable_settings(settings: dict) -> dict[str, str]:
    """The run's settings as environment values, without the machine-specific ones."""
    out: dict[str, str] = {}
    for name, value in settings.items():
        if value is None or value == "<set>" or name == "harness_profile_resolved" or _SETTINGS_SKIPPED.search(name):
            continue
        if isinstance(value, bool):
            out[name] = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            out[name] = ",".join(str(v) for v in value)
        else:
            out[name] = str(value)
    return out


def logged_steps(run_dir: Path) -> list[dict]:
    """Each logged step's action and execution outcome, as the replay compares them."""
    steps = []
    for step in _step_logs(Path(run_dir)):
        executed = step.get("executed_action") or {}
        origin = (step.get("perception") or {}).get("monitor_origin") or [0, 0]
        steps.append({
            "step": step["step_index"],
            "action": compact_action(executed.get("action")),
            "success": bool(executed.get("success", True)),
            "detail": executed.get("detail") or "done",
            "failure_category": executed.get("failure_category"),
            "batch_failed_index": executed.get("batch_failed_index"),
            # The captured monitor's origin (a secondary monitor's is not 0, 0).
            "monitor_origin": [int(origin[0]), int(origin[1])],
        })
    return steps


def _run_host(run_dir: Path, steps: list[dict]) -> dict:
    """The operating system the run's file paths follow and its home folder: from the
    run's ``host.json``, or, for an older run, guessed from its logged paths (a Windows
    run may be recorded from a folder copied to Linux)."""
    if (run_dir / "host.json").is_file():
        return _read_json(run_dir / "host.json")
    paths = [str((s.get("perception_debug") or {}).get("raw_response_artifact_path") or "") for s in steps]
    if any(_WINDOWS_PATH.match(p) for p in paths):
        system = "nt"
    elif any(p.startswith("/") for p in paths):
        system = "posix"
    else:  # relative paths: recorded where it ran
        system = os.name
    return {"os": system, "home": str(Path.home()) if system == os.name else None}


def _gate_urls(run_dir: Path) -> list[str | None]:
    """The page URLs the trust gate judged before each action (logged only when it has
    a domain allowlist), in order."""
    path = run_dir / "trust_gate_urls.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line).get("url") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_run(run_dir: Path, *, task: str, mode: str, intent: str | None = None,
               outcome: str | None = None) -> dict:
    """One run folder as a replayable recording (raises :class:`RecordingError`)."""
    run_dir = Path(run_dir)
    run_mode = loop_mode(mode)
    settings = _read_json(run_dir / "settings.json") if (run_dir / "settings.json").is_file() else {}
    _require_gemini(settings, mode)
    # New runs record the profile they resolved to; older ones only the raw env, where an
    # unset value meant the full harness for Gemini computer use (the json backend and any
    # other value never replay either).
    backend_key = "OPERON_BROWSER_BACKEND" if run_mode == "browser" else "OPERON_DESKTOP_BACKEND"
    backend = (settings.get(backend_key) or "computer_use").lower()
    resolved = (settings.get("harness_profile_resolved") or "").lower()
    requested = (settings.get("OPERON_HARNESS_PROFILE") or "").lower()
    if resolved:
        thin = resolved == "thin" and backend != "json"
    else:
        thin = backend == "computer_use" and requested == "thin"
    if not thin:
        raise RecordingError(
            "only thin-harness runs replay (the full harness's critic is not recorded); runs saved "
            "before the thin default must have OPERON_HARNESS_PROFILE=thin to replay")
    state = _read_json(run_dir / "state.json") if (run_dir / "state.json").is_file() else {}
    steps = _step_logs(run_dir)
    if not steps:
        raise RecordingError(f"{run_dir}: no completed steps")
    turns: list[dict] = []
    size: tuple[int, int] | None = None
    for step in steps:
        debug = step.get("perception_debug") or {}
        raw = _local(debug.get("raw_response_artifact_path"), run_dir)
        prompt = _local(debug.get("prompt_artifact_path"), run_dir)
        if raw is None or prompt is None:
            raise RecordingError(f"step {step['step_index']}: the model answer was not saved")
        if not prompt.read_text(encoding="utf-8").startswith(_NO_MODEL_CALL):
            # Every answer of the step (retries and zoom included) when the client
            # listed them; otherwise the step's one answer.
            answers = raw.parent / "computer_use_answers.json"
            turns.extend(_read_json(answers) if answers.is_file() else [_read_json(raw)])
        if size is None:
            frame = _local(step.get("before_artifact_path"), run_dir)
            if frame is not None:
                with Image.open(frame) as image:
                    size = image.size
    if size is None:
        raise RecordingError(f"{run_dir}: no screenshot to take the screen size from")
    return {
        "format": FORMAT,
        "task": task,
        "mode": mode,
        "intent": intent or state.get("intent") or "",
        "outcome": outcome,
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "max_steps": state.get("max_steps"),
        "permission_mode": state.get("permission_mode"),
        "screen": {"width": size[0], "height": size[1]},
        "settings": recordable_settings(settings),
        "host": _run_host(run_dir, steps),
        "gate_urls": _gate_urls(run_dir),
        "turns": turns,
        "steps": logged_steps(run_dir),
    }


# ── replay ──────────────────────────────────────────────────────────────────


class ReplayClient:
    """Stands in for ``GeminiComputerUseHttpClient``: each turn is the next recorded answer."""

    def __init__(self, turns: list[dict], *, environment: str, function_declarations: list[dict]) -> None:
        self._turns = list(turns)
        self.environment = environment
        self.function_declarations = function_declarations
        self.used = 0
        self.extra_turns = 0
        self.answers: list[dict] = []  # as the HTTP client keeps them, so replayed steps save them too

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        if self.used < len(self._turns):
            self.used += 1
            self.answers.append(self._turns[self.used - 1])
            return self._turns[self.used - 1]
        # The loop asked for more answers than the real run did: stop it.
        self.extra_turns += 1
        return {
            "perception": {"summary": "replay ended", "page_hint": "unknown", "visible_elements": [], "confidence": 0.0},
            "action": {"action_type": "wait", "wait_ms": 1000},
            "rationale": "no recorded answer left",
            "confidence": 0.0,
            "active_subgoal": "replay ended",
            "model_content": {"parts": [{"text": "The replay has no more recorded answers."}]},
        }

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[])

    @staticmethod
    def build_function_response_content(*, function_name, screenshot_path, current_url, error=None) -> dict:
        response: dict[str, Any] = {"url": current_url} if current_url else {}
        if error:
            response["error"] = error
        return {
            "role": "user",
            "parts": [
                {"function_response": {"name": function_name, "response": response}},
                {"inline_data": {"mime_type": "image/png", "data": ""}},
            ],
        }

    def latest_usage(self):
        return None


class ReplayExecutor:
    """A blank screen of the recorded size. Each action ends the way it did in the real
    run (``outcomes``, in execution order; success once they run out), and the trust
    gate sees the pages it saw then (``gate_urls``, in order)."""

    def __init__(self, width: int, height: int, frames_dir: Path, outcomes: list[dict] | None = None,
                 gate_urls: list[str | None] | None = None) -> None:
        self.width, self.height = width, height
        self.frames_dir = frames_dir
        self.actions: list[Any] = []
        self._outcomes = list(outcomes or [])
        self._gate_urls = list(gate_urls or [])
        self._n = 0

    def _frame(self) -> str:
        self._n += 1
        path = self.frames_dir / f"frame_{self._n}.png"
        # A different shade each time, so screen-diff checks see a change.
        Image.new("RGB", (self.width, self.height), (self._n * 37 % 256, 90, 120)).save(path)
        return str(path)

    async def capture(self):
        from operon.models.capture import CaptureFrame

        # The monitor the real run captured for the step about to run.
        left, top = (self._outcomes[0].get("monitor_origin") or [0, 0]) if self._outcomes else (0, 0)
        return CaptureFrame(artifact_path=self._frame(), width=self.width, height=self.height,
                            monitor_left=int(left), monitor_top=int(top))

    async def execute(self, action):
        from operon.models.execution import ExecutedAction

        self.actions.append(action)
        outcome = self._outcomes.pop(0) if self._outcomes else {}
        if outcome.get("success", True):
            return ExecutedAction(action=action, success=True, detail="replayed", artifact_path=self._frame())
        return ExecutedAction(
            action=action, success=False, detail=outcome.get("detail") or "failed in the real run",
            artifact_path=self._frame(), failure_category=outcome.get("failure_category"),
            batch_failed_index=outcome.get("batch_failed_index"),
        )

    async def context_reset(self) -> None:
        return None

    async def session_reset(self, start_url: str | None = None) -> None:
        return None

    async def get_current_url(self) -> str | None:
        return None

    async def current_url_for_run(self, run_id: str) -> str | None:
        return self._gate_urls.pop(0) if self._gate_urls else None


@dataclass
class ReplayResult:
    task: str
    ok: bool
    message: str
    status: str = ""
    recorded: list[dict | None] = field(default_factory=list)
    replayed: list[dict | None] = field(default_factory=list)
    turns_used: int = 0  # recorded model answers the replay asked for
    turns_recorded: int = 0


def _describe(action: dict | None) -> str:
    if action is None:
        return "nothing"
    parts = [str(action.get("action_type"))]
    if "x" in action and "y" in action:
        parts.append(f"at ({action['x']}, {action['y']})")
    for key in ("text", "key", "url", "tool"):
        if key in action:
            parts.append(f"{key}={action[key]!r}")
    if action.get("actions"):
        parts.append("[" + "; ".join(_describe(a) for a in action["actions"]) + "]")
    return " ".join(parts)


def compare(task: str, recorded: list[dict | None], replayed: list[dict | None], *, extra_turns: int = 0,
            status: str = "") -> ReplayResult:
    for index, (want, got) in enumerate(zip(recorded, replayed), 1):
        if want != got:
            return ReplayResult(
                task, False,
                f"{task}: step {index}: the real run did {_describe(want)}, but this code now does "
                f"{_describe(got)} with the same Gemini answer.",
                status, recorded, replayed,
            )
    if len(replayed) < len(recorded):
        return ReplayResult(
            task, False,
            f"{task}: the run now ends after {len(replayed)} step(s) (status {status or '?'}); the real run "
            f"took {len(recorded)}, next {_describe(recorded[len(replayed)])}.",
            status, recorded, replayed,
        )
    if len(replayed) > len(recorded) or extra_turns:
        more = replayed[len(recorded)] if len(replayed) > len(recorded) else None
        return ReplayResult(
            task, False,
            f"{task}: the run no longer ends where the real run did after {len(recorded)} step(s); "
            f"it asked the model for more" + (f" and did {_describe(more)}." if more else "."),
            status, recorded, replayed,
        )
    return ReplayResult(task, True, f"{task}: {len(recorded)} step(s) replayed the same", status, recorded, replayed)


async def _no_sleep(*_args: Any, **_kwargs: Any) -> None:
    return None


class _ReplayClock:
    """Settle detection's clock during a replay: a sleep only moves the clock forward,
    so any settle wait in a replay takes no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def executed_outcomes(steps: list[dict]) -> list[dict]:
    """The execution outcomes of the recorded steps that reached the executor (a pause
    for the person is logged as a step but never executed)."""
    return [s for s in steps if (s.get("action") or {}).get("action_type") != "wait_for_user"]


def resumes_in(recording: dict) -> int:
    """How many pauses the real run was resumed from: all of them, unless it ended
    waiting at its last one (the benchmark's approval budget ran out)."""
    pauses = sum(1 for s in recording["steps"] if (s.get("action") or {}).get("action_type") == "wait_for_user")
    return pauses - 1 if pauses and recording.get("status") == "waiting_for_user" else pauses


@dataclass
class LoopRun:
    run_dir: Path
    status: str
    actions: list[dict | None]
    extra_turns: int
    turns_used: int = 0


async def run_turns(turns: list[dict], *, mode: str, intent: str, width: int, height: int, root: Path,
                    max_steps: int, outcomes: list[dict] | None = None, permission_mode: str | None = None,
                    resumes_allowed: int = 10, gate_urls: list[str | None] | None = None) -> LoopRun:
    """Drive the real thin-harness loop with ``turns`` as the model's answers on a blank
    screen, in run folders under ``root`` (the caller sets the environment). A ``hybrid``
    ``mode`` runs through the desktop loop."""
    from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
    from operon.agent.backends.desktop_computer_use import (
        DESKTOP_ENVIRONMENT,
        DesktopComputerUseBackend,
        launch_app_function,
    )
    from operon.agent.loop import AgentLoop
    from operon.agent.perception.capture import ScreenCaptureService
    from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
    from operon.agent.tools import enabled_tools, function_declarations
    from operon.api.runtime.services import desktop_prompt_name
    from operon.core.contracts.perception import Environment
    from operon.core.paths import prompts_dir
    from operon.models.common import RunStatus, RunTaskRequest, StepRequest
    from operon.store.background_writer import bg_writer
    from operon.store.memory import FileBackedMemoryStore
    from operon.store.run_store import FileBackedRunStore

    mode = loop_mode(mode)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    executor = ReplayExecutor(width, height, root / "frames", outcomes, gate_urls)
    tools = enabled_tools()
    backend: Any
    if mode == "browser":
        client = ReplayClient(turns, environment="ENVIRONMENT_BROWSER", function_declarations=function_declarations(tools))
        backend = BrowserComputerUseBackend(
            client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt",
            browser_runtime=executor, tools=tools,
        )
        environment = Environment.BROWSER
    elif mode == "desktop":
        client = ReplayClient(turns, environment=DESKTOP_ENVIRONMENT,
                              function_declarations=[launch_app_function(), *function_declarations(tools)])
        backend = DesktopComputerUseBackend(client=client, prompt_path=prompts_dir() / desktop_prompt_name(), tools=tools)
        environment = Environment.DESKTOP
    else:  # loop_mode() allows nothing else
        raise RecordingError(f"unknown loop mode {mode!r}")
    run_store = FileBackedRunStore(root / "runs")
    memory_store = FileBackedMemoryStore(root / "memory")
    loop = AgentLoop(
        capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
        perception_service=backend,
        run_store=run_store,
        policy_service=ThinPolicy(backend),
        executor=executor,
        verifier_service=OwnDoneVerifier(),
        recovery_manager=ThinRecovery(),
        memory_store=memory_store,
        environment=environment,
    )
    terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    run = await loop.start_run(RunTaskRequest(intent=intent, max_steps=max_steps, permission_mode=permission_mode))
    status, steps, resumes = run.status, 0, 0
    while steps < max_steps and status not in terminal:
        response = await loop.step_run(StepRequest(run_id=run.run_id))
        steps, status = response.step_count, response.status
        if status is RunStatus.WAITING_FOR_USER:
            if resumes >= resumes_allowed:  # the real run stopped at this pause
                break
            resumes += 1
            await loop.resume_run(run.run_id)
            status = RunStatus.RUNNING
        if client.extra_turns:
            break
    await bg_writer.flush()
    return LoopRun(
        run_dir=Path(run_store.root_dir) / run.run_id,
        status=status.value,
        # The logged steps, as a recording reads them (a pause is a step too).
        actions=[s["action"] for s in logged_steps(Path(run_store.root_dir) / run.run_id)],
        extra_turns=client.extra_turns,
        turns_used=client.used,
    )


@contextmanager
def loop_environment(settings: dict | None = None):
    """The environment a replay runs in: the recorded run's settings (trust gate,
    permission mode, tools, zoom, ...) and no real waits or alerts."""
    from operon.core.settings import _REMOVED_FULL_HARNESS_VARS, get_settings

    # Recordings made before the Full harness was removed (KAN-183) still carry its variables.
    env = {
        k: str(v) for k, v in (settings or {}).items()
        if v is not None and k not in _REMOVED_FULL_HARNESS_VARS
    }
    env["OPERON_TEST_SAFE_MODE"] = "true"
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {}))  # restores anything popped below
        # A setting the run did not have must not leak in from this machine.
        for name in recordable_settings(get_settings().effective()):
            os.environ.pop(name, None)
        stack.enter_context(patch.dict(os.environ, env))
        # This patches the asyncio module's sleep (loop.asyncio is that module), so every
        # asyncio.sleep in the process is a no-op while the replay runs.
        stack.enter_context(patch("operon.agent.loop.asyncio.sleep", _no_sleep))
        # Settle detection (KAN-110) waits and times on its own hooks: a fake clock, so
        # its max_s timeout passes at once instead of in real seconds (KAN-178).
        clock = _ReplayClock()
        stack.enter_context(patch("operon.agent.perception.settle._sleep", clock.sleep))
        stack.enter_context(patch("operon.agent.perception.settle._clock", clock.time))
        stack.enter_context(patch("operon.agent.hitl.notify_desktop", lambda **_: None))
        stack.enter_context(patch("operon.agent.hitl.post_hitl_webhook", _no_sleep))
        stack.enter_context(patch("operon.agent.hitl.start_escalation_timer", _no_sleep))
        yield


@contextmanager
def recorded_path_rules(host: dict | None):
    """Judge file paths the way the recording's machine did. A Windows run keeps its ``C:\\...`` paths and ``OPERON_ALLOWED_FILE_ROOTS``, which this
    machine's ``Path`` would resolve as unrelated relative paths, so the file guard
    compares them as Windows paths (case-insensitive), also when replaying on Windows, so
    the recorded home and working directory apply, not this machine's. As on that machine, a tool's
    relative path is under the home folder, and any other relative path (an allowed root,
    an upload) is under the working directory the run started in."""
    if not host or host.get("os") != "nt":
        yield
        return
    from operon.agent.policy import file_guard
    from operon.core.settings import get_settings

    home = PureWindowsPath(host.get("home") or r"C:\Users\replay")
    cwd = PureWindowsPath(host.get("cwd") or home)

    def windows_path(raw: str, base: PureWindowsPath = home) -> PureWindowsPath:
        text = str(raw).strip()
        if text == "~" or text.startswith(("~/", "~\\")):
            text = str(home / text[2:])
        path = PureWindowsPath(text)
        if not path.is_absolute():
            path = base / path
        return PureWindowsPath(ntpath.normpath(str(path)))

    class _WindowsPath:  # file_guard's ``Path(text).expanduser().resolve()``
        def __init__(self, raw: str) -> None:
            self.raw = raw

        def expanduser(self) -> "_WindowsPath":
            return self

        def resolve(self) -> PureWindowsPath:
            return windows_path(self.raw, cwd)

    with ExitStack() as stack:
        stack.enter_context(patch.object(file_guard, "tool_path", windows_path))
        stack.enter_context(patch.object(file_guard, "Path", _WindowsPath))
        stack.enter_context(patch.object(
            file_guard, "allowed_file_roots",
            lambda: [windows_path(p, cwd) for p in get_settings().allowed_file_roots],
        ))
        stack.enter_context(patch.object(file_guard, "_canon", lambda v: v.replace("\\", "/").casefold()))
        yield


async def replay(recording: dict) -> ReplayResult:
    """Run ``recording`` through the real loop with its recorded answers and compare."""
    if recording.get("format") != FORMAT:
        raise RecordingError(f"unknown recording format {recording.get('format')!r}")
    loop_mode(recording["mode"])
    _require_gemini(recording.get("settings") or {}, recording["mode"])
    task = recording["task"]
    recorded = [s["action"] for s in recording["steps"]]
    screen = recording["screen"]
    # The real run's own step budget, so a run that ran out of steps ends the same way.
    max_steps = int(recording.get("max_steps") or len(recorded) + 3)
    with (
        tempfile.TemporaryDirectory(prefix="operon-replay-") as tmp,
        loop_environment(recording.get("settings")),
        recorded_path_rules(recording.get("host")),
    ):
        run = await run_turns(
            recording["turns"], mode=recording["mode"], intent=recording["intent"] or task,
            width=int(screen["width"]), height=int(screen["height"]), root=Path(tmp),
            max_steps=max_steps, outcomes=executed_outcomes(recording["steps"]),
            permission_mode=recording.get("permission_mode"),
            resumes_allowed=resumes_in(recording),
            gate_urls=recording.get("gate_urls"),
        )
    result = compare(task, recorded, run.actions, extra_turns=run.extra_turns, status=run.status)
    result.turns_used, result.turns_recorded = run.turns_used, len(recording["turns"])
    return result


# ── command line ────────────────────────────────────────────────────────────


def recording_paths(directory: Path = RECORDINGS_DIR) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(prog="python -m evals.replay", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="print one run folder as a recording")
    rec.add_argument("run_dir", type=Path)
    rec.add_argument("--task", required=True)
    rec.add_argument("--mode", choices=MODES, default="desktop")
    chk = sub.add_parser("check", help="replay recordings; exit 1 if any no longer matches")
    chk.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "record":
        try:
            print(json.dumps(record_run(args.run_dir, task=args.task, mode=args.mode), indent=2))
        except RecordingError as exc:
            print(f"cannot record: {exc}", file=sys.stderr)
            return 1
        return 0
    failed = 0
    for path in args.paths or recording_paths():
        try:
            result = asyncio.run(replay(_read_json(path)))
        except RecordingError as exc:
            print(f"FAIL  {path}: {exc}")
            failed += 1
            continue
        print(("ok    " if result.ok else "FAIL  ") + result.message)
        failed += not result.ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
