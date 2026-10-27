# Grounding and flakiness baselines

First runs of the grounding eval (`tests/eval/grounding.py`) and of the hard subset
repeated three times (`live_bench --suite hard --repeat 3`), both added in #186.

## Grounding (2026-10-27)

300 fixed samples: 100 ScreenSpot-v2 desktop, 100 ScreenSpot-v2 web (half text targets,
half icons) and 100 ScreenSpot-Pro (high-resolution professional apps, spread across its
groups). Each goes through Operon's own backend, and the first proposed click is scored
against the target's box.

| Backend | v2 desktop | v2 web | Pro | All | Cost | Per sample |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 (effort low) | 100% | 97% | 70% | **89%** | $3.20 | 2.9 s |
| Gemini 3.5 Flash Computer Use | 97% | 90% | 48% | **78%** | $1.46 | 5.5 s |

By target type, Pro icons are the hardest: Claude 60% (text 80%), Gemini 28% (text 68%).
Claude ran through `/bench grounding` (Bench command run 40452917122); Gemini ran from
a cloud session with the same command and `--backend computer_use`.

What the misses say:

- **Claude asks to zoom on dense screens.** Claude proposed no click on 9 Pro samples, and
  on 8 of them its reply said the screen was too small and it wanted to zoom in. Pro
  screens are up to 3840 px wide and are downscaled to 1280 px. Operon turns the
  toolset's `zoom` member off (`_DISABLED_MEMBERS` in
  `src/operon/clients/anthropic_computer_use.py`), so Claude stops instead. Turning
  zoom on is the obvious next experiment for dense desktop apps.
- **Gemini misclicks.** 55 of Gemini's 66 non-hits were clicks outside the box. Its 10
  no-clicks were safety confirmation pauses (`wait_for_user`). One sample was blocked
  by Gemini's safety filter.
- On the ScreenSpot-v2 sets both backends are near the ceiling, so v2 alone will not
  show grounding regressions. Pro is the set to watch.

## Hard subset, 3 runs each

Pending: the run is in progress and its results will follow in a separate PR.
