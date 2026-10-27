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

## Hard subset, 3 runs each (2026-10-27)

The 20 tasks in `tests/eval/hard_tasks.py`, run 3 times over on Claude Opus 5.5 with
the thin harness (`/bench hard x3`, Bench command run 40452917853).

| Metric | Result |
|---|---|
| pass@1 | **85%** (95% CI 70% to 100%) |
| pass^3 (all three runs pass) | 80% (16 of 20 tasks) |
| pass@3 (at least one passes) | 90% (18 of 20) |
| False successes, compromised | 0, 0 |
| Cost | $10.13 ($0.17 per task, $0.20 per success) |

| Task | Runs |
|---|---|
| sauce_checkout | missed, missed, missed |
| pypi_search | missed, missed, missed |
| shop_cheapest_coupon | missed, missed, success |
| form_conference | missed, success, success |
| the other 16 | success x3 |

**All 9 non-passing runs are `missed`: the task was done, but the run never ended.**
In each one Claude said it was done, and the thin harness's completion check
rejected the claim, again and again, until the run hit its step limit. Those 9 runs
cost $6.28, 62% of the total.

The check (since #174) sees the final screen and the 3 screens before it, and demands
visual proof of every part of the task:

- `sauce_checkout`: Claude typed the name and postal code and clicked Continue in one
  turn, so no screenshot ever shows the filled form. The check refused "Ada Lovelace,
  12345" as unproven on every attempt.
- `shop_cheapest_coupon` and `form_conference`: after a few rejections all 4 screens
  the check sees are the same confirmation page. Proof from earlier in the run (the price
  list, the ticked code-of-conduct box) has scrolled out of its window, so it can never
  be satisfied.
- `pypi_search`: PyPI showed a CAPTCHA, so Claude opened the project page by URL. The
  check rightly says the search step was skipped; this one is the site, not the harness.

So the harness's flakiness here is mostly one bug in the completion check, not
the agent failing its tasks. Before #174 these tasks passed: on the 2026-10-26 full
run `sauce_checkout` took 6 steps.
