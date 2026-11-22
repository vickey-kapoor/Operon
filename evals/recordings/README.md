# Recorded runs (KAN-167, KAN-178)

Each `*.json` here is one thin-harness Gemini computer-use run: the model's answers and
the actions Operon ran. `tests/evals/test_replay.py` replays every one through the real
loop on each PR, for free (no model is called), and fails if the code now does something
different with the same answers or leaves a recorded answer unused
(`python -m evals.replay check` runs them by hand).

There are two kinds:

- **Real Gemini runs** (`dyn_*`), copied from a paid run's artifact. To add one: open a
  batch or `/bench` artifact, take `<report>/recordings/<task>.json`, read it (it holds the
  model's own words about the screen, nothing else), and copy it here. If a deliberate
  change alters what the loop does, re-record from the next paid run instead of editing
  the expected actions by hand.
- **Scripted runs** (`scripted_*`, `"source": "scripted"`): fixed Gemini-shaped answers in
  `evals/scripted_recordings.py`, run once through the real loop, so the desktop and
  hybrid paths are covered without a paid run. Regenerate them for $0 with
  `python -m evals.scripted_recordings` (`--check` says whether they are current). They
  carry a neutral host (`/home/replay`), never the machine that made them.

Modes: `browser`, `desktop` and `hybrid`. A hybrid run (a desktop app and a browser on one
machine) ran through the desktop loop, so it replays through the desktop path and keeps
its `hybrid` label. Claude and GPT runs (`OPERON_*_BACKEND=model`) are refused when
recorded and when replayed: their answers are not saved per turn in Gemini's shape.

Current recordings: `dyn_save_theme` (13 steps), `dyn_waitlist` (8) and `dyn_search_open` (5),
from the Browser Thin dynamic suite (KAN-114) in nightly run 43714855920 on 2026-11-15;
`scripted_desktop_notepad` (4, desktop) and `scripted_hybrid_site_to_file` (8, hybrid).
