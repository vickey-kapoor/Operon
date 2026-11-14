"""Demo GIF (KAN-141): turn one run's saved screenshots into a short README GIF.

Every run keeps a ``before.png`` per step (and an ``after.png``) under
``.var/runs/<run_id>/step_<n>/``. This stitches them in order, scaled down, with the
task as a caption bar and the last frame held longer, into one GIF. It reads files
only: no model calls, no browser, nothing spent.

    .venv/bin/python -m evals.demo_gif --latest --out docs/assets/demo.gif
    .venv/bin/python -m evals.demo_gif --run .var/runs/<run_id> --out demo.gif

The "Record demo GIF" workflow runs one short real task (a few cents of Gemini calls)
and then this, and uploads the GIF to look at before it goes in the README.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_STEP = re.compile(r"^step_(\d+)$")
_CAPTION_H = 44


def frames_of(run_dir: Path) -> list[Path]:
    """The run's screenshots in order: each step's before.png, then the last after.png."""
    steps = sorted(
        (int(m.group(1)), p) for p in run_dir.iterdir() if p.is_dir() and (m := _STEP.match(p.name))
    )
    shots = [p / "before.png" for _, p in steps if (p / "before.png").is_file()]
    for _, step in reversed(steps):
        if (step / "after.png").is_file():
            shots.append(step / "after.png")
            break
    return shots


def task_of(run_dir: Path) -> str:
    try:
        return str(json.loads((run_dir / "state.json").read_text(encoding="utf-8")).get("intent") or "")
    except (OSError, ValueError):
        return ""


def latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "state.json").is_file()] if root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs with a state.json under {root}")
    return max(runs, key=lambda p: (p / "state.json").stat().st_mtime)


def _caption(frame: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (frame.width, frame.height + _CAPTION_H), (17, 17, 20))
    out.paste(frame, (0, _CAPTION_H))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=20)
    label = text if len(text) <= 90 else text[:87] + "..."
    draw.text((14, (_CAPTION_H - 20) // 2), label, fill=(240, 240, 240), font=font)
    return out


def make_gif(
    run_dir: Path,
    out: Path,
    *,
    width: int = 960,
    frame_ms: int = 900,
    last_ms: int = 3000,
    caption: str | None = None,
) -> int:
    """Write the GIF; returns how many frames it holds."""
    shots = frames_of(run_dir)
    if not shots:
        raise FileNotFoundError(f"no step screenshots in {run_dir}")
    text = task_of(run_dir) if caption is None else caption
    frames: list[Image.Image] = []
    for shot in shots:
        with Image.open(shot) as img:
            frame = img.convert("RGB")
        if frame.width > width:
            frame = frame.resize((width, round(frame.height * width / frame.width)), Image.LANCZOS)
        if text:
            frame = _caption(frame, f"Task: {text}")
        frames.append(frame)
    durations = [frame_ms] * (len(frames) - 1) + [last_ms]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.demo_gif", description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", type=Path, help="a run folder (.var/runs/<run_id>)")
    which.add_argument("--latest", action="store_true", help="the newest run under the runs folder")
    parser.add_argument("--out", type=Path, default=Path("demo.gif"))
    parser.add_argument("--width", type=int, default=960, help="scale frames down to this width")
    parser.add_argument("--frame-ms", type=int, default=900)
    parser.add_argument("--caption", default=None, help="caption text (default: the run's task)")
    args = parser.parse_args(argv)

    from operon.core.paths import runs_dir

    try:
        run_dir = args.run if args.run else latest_run(runs_dir())
        count = make_gif(run_dir, args.out, width=args.width, frame_ms=args.frame_ms, caption=args.caption)
    except FileNotFoundError as exc:
        print(f"demo_gif: {exc}", file=sys.stderr)
        return 1
    size_kb = args.out.stat().st_size // 1024
    print(f"Wrote {args.out}: {count} frames, {size_kb} KB, from {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
