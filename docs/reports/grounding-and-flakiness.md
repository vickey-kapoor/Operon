# Grounding and flakiness baselines

First runs of the grounding eval (`evals/grounding.py`) and of the hard subset
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

The 20 tasks in `evals/hard_tasks.py`, run 3 times over on Claude Opus 5.5 with
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

## Hard subset rerun with the completion-check fix (2026-10-27)

Same 20 tasks x3 on #189 (`/bench hard x3`, Bench command run 40567969830). The check
now also gets an action log, counts identical earlier screens once, and ends a run as
`completion_unverified` after two rejected claims.

| Metric | Before (#188) | After (#189) |
|---|---|---|
| pass@1 | 85% (CI 70% to 100%) | **95%** (CI 85% to 100%) |
| pass^3 | 80% | **95%** |
| pass@3 | 90% | 95% |
| Flaky tasks | 2 | **0** |
| False successes, compromised | 0, 0 | 0, 0 |
| Cost | $10.13 | **$5.00** ($0.083 per task) |
| Spent on misses | $6.28 | $0.55 |

`sauce_checkout`, `form_conference` and `pypi_search` now pass 3 of 3 (PyPI showed no
CAPTCHA this time). No run reached the step limit.

The 3 remaining misses are all `shop_cheapest_coupon`: the order was right every time,
but the check could not see the price list Claude chose from. It was on the run's first
screen, which had dropped out of the recent-screens window. The run's first screen is
now always included (follow-up commit on #189), which is not yet measured live.

## Apps check of the first-screen fix (2026-10-31)

`/bench apps` on main after #189 (Bench command run 41216784014): **11 of 11**, 0 false
successes, $1.06. `shop_cheapest_coupon`, which missed 3 of 3 in the rerun above,
passed in 6 steps with its first "done" accepted (`task_completed`). This is one run,
not a repeat, so it shows the fix works rather than how stable it is.

## Grounding with zoom (2026-10-31)

Claude Opus 5.5 with `OPERON_CLAUDE_ZOOM=true` (#190) on the same 300 samples
(`/bench grounding zoom`). Results pending.
