# Recorded Gemini runs (KAN-167)

Each `*.json` here is one real thin-harness Gemini run: the model's answers and the
actions Operon ran. `tests/evals/test_replay.py` replays every one through the real
loop on each PR, for free, and fails if the code now does something different with
the same answers (`python -m evals.replay check` runs them by hand).

To add one: open a batch or `/bench` artifact, take `<report>/recordings/<task>.json`,
read it (it holds the model's own words about the screen, nothing else), and copy it
here. If a deliberate change alters what the loop does, re-record from the next
batch run instead of editing the expected actions by hand.

Current recordings: `dyn_save_theme` (13 steps), `dyn_waitlist` (8) and `dyn_search_open` (5),
from the Browser Thin dynamic suite (KAN-114) in nightly run 43714855920 on 2026-11-15.
