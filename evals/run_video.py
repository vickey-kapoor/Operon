"""Run videos (KAN-185): turn a run's saved screenshots into a short captioned MP4.

Every run keeps a ``before.png`` per step (and a last ``after.png``) under
``.var/runs/<run_id>/step_<n>/``. This stitches them in order, scaled down, with a caption
bar (the task, then "Step N/M: <action>") and the verdict on the held last frame, into one MP4
per task that can be downloaded and watched offline. It reads files only: no model calls,
no browser, nothing spent.

    .venv/bin/python -m evals.run_video --run .var/runs/<run_id> --out run.mp4
    .venv/bin/python -m evals.run_video --all .var/runs --out-dir videos

The file is written with OpenCV (mp4v); when ``ffmpeg`` is on PATH it is re-encoded to H.264
(yuv420p, faststart) so browsers, Windows and Loom play it. Without ffmpeg the mp4v file stays.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evals.demo_gif import frames_of, task_of

FPS = 2
_CAPTION_H = 72
_LINE_MAX = 110
_STEP_DIR = re.compile(r"^step_(\d+)$")


def _clip(text: str, limit: int = _LINE_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe(action: dict[str, Any]) -> str:
    """One short phrase for a logged action dict; empty when nothing useful is there."""
    kind = str(action.get("action_type") or "").lower()
    if not kind:
        return ""
    if kind == "batch" or action.get("actions"):
        subs = [a for a in action.get("actions") or [] if isinstance(a, dict)]
        return f"batch of {len(subs)}" if subs else kind
    detail = ""
    if action.get("text"):
        detail = f' "{action["text"]}"'
    elif action.get("key"):
        detail = f" {action['key']}"
    elif action.get("url"):
        detail = f" {action['url']}"
    elif action.get("x") is not None and action.get("y") is not None:
        detail = f" at ({action['x']}, {action['y']})"
    elif action.get("scroll_amount") is not None:
        detail = f" {action['scroll_amount']}"
    elif action.get("wait_ms") is not None:
        detail = f" {action['wait_ms']} ms"
    return f"{kind}{detail}"


def actions_of(run_dir: Path) -> dict[int, str]:
    """Step number to a short action summary, read from ``run.jsonl``. Empty if unreadable."""
    out: dict[int, str] = {}
    try:
        lines = (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            entry = json.loads(line)
            index = int(entry["step_index"])
            summary = _describe(entry["executed_action"]["action"])
        except (ValueError, KeyError, TypeError):
            continue
        if summary:
            out[index] = summary
    return out


def _font() -> Any:
    try:
        return ImageFont.load_default(size=20)
    except TypeError:  # very old Pillow: bitmap default only
        return ImageFont.load_default()


def _frame_array(shot: Path, width: int, line1: str, line2: str) -> np.ndarray:
    with Image.open(shot) as img:
        frame = img.convert("RGB")
    w = min(width, frame.width) // 2 * 2
    h = max(2, round(frame.height * w / frame.width) // 2 * 2)
    if (w, h) != frame.size:
        frame = frame.resize((w, h), Image.LANCZOS)
    out = Image.new("RGB", (w, h + _CAPTION_H), (17, 17, 20))
    out.paste(frame, (0, _CAPTION_H))
    draw = ImageDraw.Draw(out)
    font = _font()
    draw.text((14, 10), line1, fill=(240, 240, 240), font=font)
    draw.text((14, 40), line2, fill=(150, 210, 255), font=font)
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def _to_h264(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = path.with_name(path.stem + ".h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
        if done.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(path)
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        tmp.unlink(missing_ok=True)


def make_video(
    run_dir: Path,
    out: Path,
    *,
    width: int = 1280,
    seconds_per_step: float = 1.0,
    last_seconds: float = 3.0,
    task: str | None = None,
    verdict: str | None = None,
) -> int:
    """Write the MP4; returns how many video frames it holds (repeats included)."""
    shots = frames_of(run_dir)
    if not shots:
        raise FileNotFoundError(f"no step screenshots in {run_dir}")
    text = task_of(run_dir) if task is None else task
    actions = actions_of(run_dir)
    steps = sum(1 for p in run_dir.iterdir() if p.is_dir() and _STEP_DIR.match(p.name))
    steps = steps or len(shots)
    line1 = _clip(f"Task: {text}") if text else ""

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0
    try:
        for i, shot in enumerate(shots):
            last = i == len(shots) - 1
            m = _STEP_DIR.match(shot.parent.name)
            number = int(m.group(1)) if m else i + 1
            if last and shot.name == "after.png":
                line2 = "Final screen"
            else:
                line2 = f"Step {number}/{steps}"
                if actions.get(number):
                    line2 += f": {actions[number]}"
            if last and verdict:
                line2 = f"{line2}  |  {verdict}"
            frame = _frame_array(shot, width, line1, _clip(line2))
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise OSError(f"cannot write video to {out}")
            repeats = max(1, round((last_seconds if last else seconds_per_step) * FPS))
            for _ in range(repeats):
                writer.write(frame)
                written += 1
    finally:
        if writer is not None:
            writer.release()
    _to_h264(out)
    return written


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:80] or "run"


def save_videos(items: list[tuple[str, Path, str, str]], target: Path) -> int:
    """Write ``target/<name>-<verdict>.mp4`` for each (name, run_dir, task_text, verdict).

    A run that cannot be rendered is skipped with a note on stderr: a video must never crash
    a benchmark. Returns how many were written."""
    written = 0
    for name, run_dir, task_text, verdict in items:
        out = target / (f"{_safe(name)}-{_safe(verdict)}.mp4" if verdict else f"{_safe(name)}.mp4")
        try:
            make_video(Path(run_dir), out, task=task_text, verdict=verdict or None)
        except Exception as exc:  # noqa: BLE001 - never let a video break the benchmark
            print(f"  {name}: no video ({exc})", file=sys.stderr)
            out.unlink(missing_ok=True)
            continue
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run_video", description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", type=Path, help="a run folder (.var/runs/<run_id>)")
    which.add_argument("--all", type=Path, help="a folder of run folders: one video per run with screenshots")
    parser.add_argument("--out", type=Path, default=Path("run.mp4"), help="with --run: the MP4 to write")
    parser.add_argument("--out-dir", type=Path, default=Path("videos"), help="with --all: where the MP4s go")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args(argv)

    if args.run:
        try:
            count = make_video(args.run, args.out, width=args.width)
        except (FileNotFoundError, OSError) as exc:
            print(f"run_video: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {args.out}: {count} frames, {args.out.stat().st_size // 1024} KB, from {args.run}")
        return 0

    runs = sorted(p for p in args.all.iterdir() if p.is_dir() and frames_of(p)) if args.all.is_dir() else []
    items = [(p.name, p, task_of(p), "") for p in runs]
    done = save_videos(items, args.out_dir)
    print(f"Wrote {done} of {len(runs)} videos to {args.out_dir}")
    return 0 if done or not runs else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
