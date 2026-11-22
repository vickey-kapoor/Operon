"""Scripted recordings: desktop and hybrid runs the free replay covers (KAN-178).

The committed real recordings (``evals/recordings/dyn_*.json``) are all browser runs.
These scripts fill the gap without a paid run: each is a fixed list of Gemini-shaped
answers, run once through the real thin-harness loop on a blank screen
(:func:`evals.replay.run_turns`) and saved as an ordinary recording, so
``tests/evals/test_replay.py`` replays the desktop and the hybrid path on every PR.
They are marked ``"source": "scripted"`` and carry a neutral host, so no developer
path is committed. Generating them costs nothing (no model is called).

    python -m evals.scripted_recordings           # (re)write evals/recordings/scripted_*.json
    python -m evals.scripted_recordings --check   # exit 1 if a saved one differs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evals.replay import RECORDINGS_DIR, loop_environment, record_run, run_turns

# Gemini computer-use answers, in the shape the HTTP client normalizes them to.
_PERCEPTION = {"summary": "screen", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5}


def call(name: str, **args) -> dict:
    """One Gemini answer that calls ``name`` with ``args``."""
    fc = {"name": name, "args": args}
    return {"perception": _PERCEPTION, "function_call": fc, "function_calls": [fc], "rationale": name,
            "confidence": 0.7, "active_subgoal": name, "model_content": {"parts": [{"function_call": fc}]}}


def done() -> dict:
    """A text-only Gemini answer: the model says the task is complete."""
    return {"perception": _PERCEPTION, "action": {"action_type": "wait", "wait_ms": 1000},
            "rationale": "The task is complete.", "confidence": 0.5, "active_subgoal": "finish",
            "model_content": {"parts": [{"text": "The task is complete."}]}}


@dataclass(frozen=True)
class Script:
    mode: str
    intent: str
    turns: list[dict]


SETTINGS = {"OPERON_DESKTOP_PLATFORM": "windows"}
SCREEN = (1280, 800)
MAX_STEPS = 10
# Where a recording's file paths are judged from; neutral, not this machine's.
HOST = {"os": "posix", "home": "/home/replay", "cwd": "/home/replay"}

SCRIPTS: dict[str, Script] = {
    "scripted_desktop_notepad": Script(
        mode="desktop",
        intent="Open Notepad, type 'hello from Operon' and save the file with Ctrl+S",
        turns=[
            call("launch_app", app="notepad"),
            call("type_text_at", x=500, y=400, text="hello from Operon", press_enter=False),
            call("key_combination", keys="control+s"),
            done(),
        ],
    ),
    "scripted_hybrid_site_to_file": Script(
        mode="hybrid",
        intent=("Open http://127.0.0.1:8765/apps/mail/ in the browser, read the order number in the "
                "newest email, and write it into a new Notepad file saved with Ctrl+S"),
        turns=[
            call("launch_app", app="browser"),
            call("click_at", x=500, y=60),
            call("type_text_at", x=500, y=60, text="http://127.0.0.1:8765/apps/mail/", press_enter=True),
            call("click_at", x=400, y=300),
            call("launch_app", app="notepad"),
            call("type_text_at", x=500, y=400, text="Order 4821", press_enter=False),
            call("key_combination", keys="control+s"),
            done(),
        ],
    ),
}


def build(name: str, root: Path) -> dict:
    """Run script ``name`` through the real loop in ``root`` and return its recording."""
    script = SCRIPTS[name]
    with loop_environment(SETTINGS):
        run = asyncio.run(run_turns(script.turns, mode=script.mode, intent=script.intent,
                                    width=SCREEN[0], height=SCREEN[1], root=root, max_steps=MAX_STEPS))
    if run.status != "succeeded":
        raise RuntimeError(f"{name}: the scripted run ended {run.status}, not succeeded")
    recording = record_run(run.run_dir, task=name, mode=script.mode, outcome="success")
    recording["source"] = "scripted"
    recording["host"] = dict(HOST)
    return recording


def render(recording: dict) -> str:
    return json.dumps(recording, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.scripted_recordings", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a saved recording differs from its script")
    args = parser.parse_args(argv)
    differs = 0
    for name in SCRIPTS:
        with tempfile.TemporaryDirectory(prefix="operon-scripted-") as tmp:
            text = render(build(name, Path(tmp)))
        path = RECORDINGS_DIR / f"{name}.json"
        if args.check:
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            print(("ok    " if same else "DIFF  ") + str(path.name))
            differs += not same
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.name}")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
