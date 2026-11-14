"""Record a real Gemini run and replay it through the real loop for free (KAN-167).

**Record.** A paid benchmark run (the nightly batch, KAN-165) already saves, for each
step, Gemini's answer (``computer_use_raw.json``) and the action Operon then ran
(``run.jsonl``). :func:`record_run` folds one run folder into a small recording: the
task, the settings that shape the loop, every model answer in order and every action
Operon ran. ``python -m evals.live_bench --keep-screens`` writes one per thin-harness
Gemini computer-use task to ``<report>/recordings/<task>.json``, so the nightly
artifact carries them.

**Replay.** :func:`replay` runs the recorded task again through the real
``AgentLoop``, the real computer-use backend and the thin harness, with Gemini
replaced by the recorded answers and the screen by a blank frame of the recorded
size. Nothing is paid and nothing is clicked. The loop must run the same actions in
the same order as it did for real; when a code change makes it do something else
with the same model answers, the replay says at which step and what changed.

Recordings that CI replays live in ``evals/recordings/`` (``tests/evals/test_replay.py``).
Add one by copying it there from a nightly artifact. Their screenshots are never
kept: a recording holds only the model's text and actions.

    python -m evals.replay record .var/runs/<run_id> --task NAME --mode desktop > rec.json
    python -m evals.replay check evals/recordings/*.json

Only thin-harness runs replay (the full harness also asks a Gemini critic, which is
not recorded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
FORMAT = 1
# Settings that change what the loop does with the same model answers.
_SETTINGS_KEPT = ("OPERON_DESKTOP_PLATFORM", "OPERON_TOOLS", "OPERON_HARNESS_PROFILE")
# The action fields compared between the recording and the replay.
_ACTION_FIELDS = (
    "action_type", "x", "y", "x_end", "y_end", "text", "key", "url", "press_enter",
    "clear_before_typing", "scroll_amount", "horizontal", "wait_ms", "tool", "tool_args",
)
_NO_MODEL_CALL = "No model call"


class RecordingError(ValueError):
    """The run folder cannot be turned into a recording that replays."""


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
    have been copied out of the runs root, e.g. into a benchmark artifact)."""
    if not path:
        return None
    original = Path(path)
    if original.is_file():
        return original
    parts = original.parts
    for i, part in enumerate(parts):
        if part.startswith("step_"):
            candidate = run_dir.joinpath(*parts[i:])
            if candidate.is_file():
                return candidate
    return None


def record_run(run_dir: Path, *, task: str, mode: str, intent: str | None = None,
               outcome: str | None = None) -> dict:
    """One run folder as a replayable recording (raises :class:`RecordingError`)."""
    run_dir = Path(run_dir)
    settings = _read_json(run_dir / "settings.json") if (run_dir / "settings.json").is_file() else {}
    if (settings.get("OPERON_HARNESS_PROFILE") or "").lower() != "thin":
        raise RecordingError("only thin-harness runs replay (the full harness's critic is not recorded)")
    state = _read_json(run_dir / "state.json") if (run_dir / "state.json").is_file() else {}
    steps = _step_logs(run_dir)
    if not steps:
        raise RecordingError(f"{run_dir}: no completed steps")
    turns: list[dict] = []
    recorded_steps: list[dict] = []
    size: tuple[int, int] | None = None
    for step in steps:
        debug = step.get("perception_debug") or {}
        raw = _local(debug.get("raw_response_artifact_path"), run_dir)
        prompt = _local(debug.get("prompt_artifact_path"), run_dir)
        if raw is None or prompt is None:
            raise RecordingError(f"step {step['step_index']}: the model answer was not saved")
        if not prompt.read_text(encoding="utf-8").startswith(_NO_MODEL_CALL):
            turns.append(_read_json(raw))
        if size is None:
            frame = _local(step.get("before_artifact_path"), run_dir)
            if frame is not None:
                with Image.open(frame) as image:
                    size = image.size
        recorded_steps.append({
            "step": step["step_index"],
            "action": compact_action((step.get("executed_action") or {}).get("action")),
        })
    if size is None:
        raise RecordingError(f"{run_dir}: no screenshot to take the screen size from")
    return {
        "format": FORMAT,
        "task": task,
        "mode": mode,
        "intent": intent or state.get("intent") or "",
        "outcome": outcome,
        "status": state.get("status"),
        "screen": {"width": size[0], "height": size[1]},
        "settings": {k: settings.get(k) for k in _SETTINGS_KEPT},
        "turns": turns,
        "steps": recorded_steps,
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

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        if self.used < len(self._turns):
            self.used += 1
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
    """A blank screen of the recorded size; every action 'succeeds' and is remembered."""

    def __init__(self, width: int, height: int, frames_dir: Path) -> None:
        self.width, self.height = width, height
        self.frames_dir = frames_dir
        self.actions: list[Any] = []
        self._n = 0

    def _frame(self) -> str:
        self._n += 1
        path = self.frames_dir / f"frame_{self._n}.png"
        # A different shade each time, so screen-diff checks see a change.
        Image.new("RGB", (self.width, self.height), (self._n * 37 % 256, 90, 120)).save(path)
        return str(path)

    async def capture(self):
        from operon.models.capture import CaptureFrame

        return CaptureFrame(artifact_path=self._frame(), width=self.width, height=self.height)

    async def execute(self, action):
        from operon.models.execution import ExecutedAction

        self.actions.append(action)
        return ExecutedAction(action=action, success=True, detail="replayed", artifact_path=self._frame())

    async def context_reset(self) -> None:
        return None

    async def session_reset(self, start_url: str | None = None) -> None:
        return None

    async def get_current_url(self) -> str | None:
        return None


@dataclass
class ReplayResult:
    task: str
    ok: bool
    message: str
    status: str = ""
    recorded: list[dict | None] = field(default_factory=list)
    replayed: list[dict | None] = field(default_factory=list)


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


@dataclass
class LoopRun:
    run_dir: Path
    status: str
    actions: list[dict | None]
    extra_turns: int


async def run_turns(turns: list[dict], *, mode: str, intent: str, width: int, height: int, root: Path,
                    max_steps: int) -> LoopRun:
    """Drive the real thin-harness loop with ``turns`` as the model's answers on a blank
    screen, in run folders under ``root`` (the caller sets the environment)."""
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

    (root / "frames").mkdir(parents=True, exist_ok=True)
    executor = ReplayExecutor(width, height, root / "frames")
    tools = enabled_tools()
    backend: Any
    if mode == "browser":
        client = ReplayClient(turns, environment="ENVIRONMENT_BROWSER", function_declarations=function_declarations(tools))
        backend = BrowserComputerUseBackend(
            client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt",
            browser_runtime=executor, tools=tools,
        )
        environment = Environment.BROWSER
    else:
        client = ReplayClient(turns, environment=DESKTOP_ENVIRONMENT,
                              function_declarations=[launch_app_function(), *function_declarations(tools)])
        backend = DesktopComputerUseBackend(client=client, prompt_path=prompts_dir() / desktop_prompt_name(), tools=tools)
        environment = Environment.DESKTOP
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
    run = await loop.start_run(RunTaskRequest(intent=intent, max_steps=max_steps))
    status, steps, resumes = run.status, 0, 0
    while steps < max_steps and status not in terminal:
        response = await loop.step_run(StepRequest(run_id=run.run_id))
        steps, status = response.step_count, response.status
        if status is RunStatus.WAITING_FOR_USER:
            if resumes >= 10:  # the benchmark approves pauses too, up to its budget
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
        actions=[compact_action(a.model_dump(mode="json", exclude_none=True)) for a in executor.actions],
        extra_turns=client.extra_turns,
    )


@contextmanager
def loop_environment(settings: dict | None = None):
    """The environment a replay runs in: the recorded settings, the thin harness, no zoom
    (a recorded answer already has zoom's coordinates), and no real waits or alerts."""
    env = {k: v for k, v in (settings or {}).items() if v is not None}
    env.update({"OPERON_TEST_SAFE_MODE": "true", "OPERON_ZOOM": "off", "OPERON_HARNESS_PROFILE": "thin"})
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {}))  # restores anything popped below
        for name in _SETTINGS_KEPT:  # a setting the run did not have must not leak in from here
            os.environ.pop(name, None)
        stack.enter_context(patch.dict(os.environ, env))
        stack.enter_context(patch("operon.agent.loop.asyncio.sleep", _no_sleep))
        stack.enter_context(patch("operon.agent.hitl.notify_desktop", lambda **_: None))
        stack.enter_context(patch("operon.agent.hitl.post_hitl_webhook", _no_sleep))
        stack.enter_context(patch("operon.agent.hitl.start_escalation_timer", _no_sleep))
        yield


async def replay(recording: dict) -> ReplayResult:
    """Run ``recording`` through the real loop with its recorded answers and compare."""
    if recording.get("format") != FORMAT:
        raise RecordingError(f"unknown recording format {recording.get('format')!r}")
    task = recording["task"]
    recorded = [s["action"] for s in recording["steps"]]
    screen = recording["screen"]
    with tempfile.TemporaryDirectory(prefix="operon-replay-") as tmp, loop_environment(recording.get("settings")):
        run = await run_turns(
            recording["turns"], mode=recording["mode"], intent=recording["intent"] or task,
            width=int(screen["width"]), height=int(screen["height"]), root=Path(tmp),
            max_steps=len(recorded) + 3,
        )
    return compare(task, recorded, run.actions, extra_turns=run.extra_turns, status=run.status)


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
    rec.add_argument("--mode", choices=("browser", "desktop", "hybrid"), default="desktop")
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
        result = asyncio.run(replay(_read_json(path)))
        print(("ok    " if result.ok else "FAIL  ") + result.message)
        failed += not result.ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
