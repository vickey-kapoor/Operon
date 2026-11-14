# RFC 0003 — A repeated action is progress only if it reaches a new screen

| | |
|---|---|
| **Status** | Proposed; first calibration pass done (§6) |
| **Created** | 2026-11-13 |
| **Story** | — |
| **Related** | [`../architecture.md`](../architecture.md), PR #263 (the `xfail` test that shows the gap) |

> RFCs are design proposals, not implementation truth. The code and
> [`../architecture.md`](../architecture.md) stay authoritative.

## 1. Summary

The full harness decides whether each step made progress, and its loop guards (block a
repeated action, stop a run that loops) only fire after steps that did not. Today a
repeated action that changes the screen always counts as progress, so a model that keeps
relaunching Notepad never trips a guard and runs to `max_steps`.

Proposal: when an action repeats, compare the screen it produced with the screens the same
action produced earlier in the run. A new screen is progress (scrolling, paging with
Next); a screen the action already produced is not (the fifth Notepad launch). Ships behind
`OPERON_PROGRESS_REVISIT`, off by default.

## 2. Motivation

`meaningful_progress` (`agent/policy/progress_tracker.py`) documents the intended rule:

> Screen change + novel action = progress (typing text in Notepad).
> Screen change + repeated action = NOT progress (launch_app Notepad 5x).

The code reads "novel" as "not in `repeated_action_count`", and every progress step clears
that counter. So the step after any progress step sees every action as new again. Traced
directly (PR #263):

```
step 1..5: launch_app notepad, screen change 0.3 → progress=True, streak=0, loop=None
```

Because the streak never grows, none of the guards that need `no_progress_streak > 0` fire:
the repeat block, the same-target block, loop detection and the no-progress ceiling. The
`_prefer_launch_app_rule` guard only stops the *rule* from relaunching; the model's own
relaunch passes.

## 3. Goals / Non-goals

**Goals**

- A repeated action that keeps producing the same screen stops counting as progress, so the
  existing guards stop the run within a few steps instead of at `max_steps`.
- Legitimate repeats keep counting: scrolling a long page, Next through results, Tab
  through a form, pressing Down in a list.
- Works on every backend, including Computer Use, which reports no screen elements.
- Vision-only: decided from screenshots, nothing structural.

**Non-goals**

- The thin harness. It records progress traces but never blocks or stops on them; it is
  untouched.
- The subgoal shortcut. A new subgoal counts as progress before the screen is looked at,
  and on the Computer Use backend the subgoal is the call plus its arguments
  (`_subgoal_for_calls`), so any change in arguments is a "new subgoal". That shortcut is
  load-bearing today: replaying saved runs without it stops 2 of 8 successful runs (§6).
  It stays as it is. On that backend this RFC's rule therefore only sees exact repeats
  (same call, same arguments), which is the runaway it targets.
- Changing the existing thresholds or guard counts.

## 4. Design

### Why screenshots, not page signatures

The obvious key is "same action on the same page" using `page_signature`. It fails on the
Computer Use backend: its perception has `elements_reported=False` and no elements, so
every page's signature is `<hint>|none|none` and all pages look alike. Scrolling a long
page would read as "same action, same page" and be stopped. Screenshots exist for every
backend.

### The rule

In `ProgressTracker.update_progress_state`, after the existing checks:

1. Keep, per action signature, the after-screenshot paths of its recent successful runs:
   `ProgressState.action_outcomes: dict[str, list[str]]` — paths only, at most 3 per
   action and 20 actions, oldest dropped. **Not cleared on progress** (that is the bug).
2. When the action's signature already has outcomes, compare this step's after-screenshot
   with each with `compute_screen_change_ratio`. If any is below
   `REVISIT_SAME_SCREEN_THRESHOLD`, the action reached a screen it already produced:
   **not progress**.
3. Otherwise fall through to today's rule unchanged.

The stop condition and a subgoal change still count first, exactly as now.

With the rule, five Notepad launches go: step 1 progress (new subgoal); step 2 same screen
as step 1, so no progress, streak 1; step 3 streak 2, repeat count 2; step 4 is blocked by
the existing repeat guard. Scrolling compares each new screen with the earlier scroll
results, finds them different, and keeps counting.

### Failing open

- A missing or unreadable screenshot must never look like "same screen".
  `compute_screen_change_ratio` returns `0.0` for a missing file, which reads as identical,
  so the rule checks that both files exist first and otherwise falls back to today's rule.
- With the flag off, `action_outcomes` is neither written nor read.
- Old saved runs load: the new field defaults to empty.

### Wiring

`update_progress_state` gains an `after_artifact_path` argument; the loop already has it
(`_after_artifact_path`). Cost: up to 3 extra 480×270 grayscale compares, only on steps
whose action repeats.

## 5. Risks

- **The threshold is the whole design.** Too high and two pages of similar results read as
  the same screen, stopping a real task. Too low and a cascaded window (offset by a few
  pixels) reads as new, and the bug stays. 1% (the existing stall threshold) is the starting
  point; §6 measures it before anything defaults on.
- **Animated or live screens** (a clock, a video, a spinner) keep every screen "new", so the
  fix silently doesn't fire. That is the fail-open direction and no worse than today.
- **Stopping a run that would have succeeded.** Only through the existing guards, which
  already stop runs today after real no-progress steps; this changes when they see one.

## 6. Measurement before default-on

1. **Offline unit tests**: flip PR #263's `xfail` to a pass under the flag; add a scroll
   test (repeat with new screens keeps progressing) and a missing-screenshot test (falls
   back).
2. **Threshold calibration from saved runs** (`.var/runs/*/step_*` before/after
   screenshots, no API cost): for repeated actions, histogram the ratio between their
   after-screens, labelled by hand as "same screen" vs "new screen". Pick the threshold
   that separates them; record it in this RFC. First pass done; see below.
3. **A scripted runaway** (offline eval harness, no API cost): a scenario whose model
   relaunches the same app with the same arguments five times. Confirms the "same screen"
   side of the threshold, which saved runs don't yet cover.
4. **Benchmarks** (cost real credits; only when asked): Browser Full core and Desktop Full
   core, flag off vs on, once each. Ship default-on only if pass rate does not drop and at
   least one runaway run ends earlier.

### Calibration, first pass (2026-11-13)

Data: the 9 local runs (of 144) that kept their step screenshots and progress traces; 8
succeeded, 1 failed. Screenshots from the offline test suite (`.var/test-artifacts`) are
synthetic and were excluded.

**After-screen vs. after-screen, for actions repeated within a run:**

| Repeated action | Pairs | Ratio between after-screens | Correct verdict |
|---|---|---|---|
| `scroll` (Settings page) | 6 | 3.0 – 4.4% | new screen |
| `launch_app notepad`, after switching to Calculator and back | 1 | 12.7% | new screen (a legitimate switch back) |
| `type 2148`, into Calculator then into Notepad | 1 | 6.9% | new screen |
| `navigate` to the same search URL | 3 | no after-screen saved | — (falls back to today's rule) |

Every legitimate repeat is at least 3× above **1%**, so 1% stays the starting threshold.
Replaying all 9 runs' traces under this RFC's rule changes no step's verdict. No saved run
contains a real runaway, so the "same screen" side is unmeasured; item 3 above covers it.

**Dropping the subgoal shortcut** (counting progress only by screen change and novelty) was
replayed on the same traces and **rejected**: it stops 2 of the 8 successful runs.

| Successful run | What would have fired |
|---|---|
| Settings lookup | loop stop at step 5: three steps at 0.0% change while the page loaded |
| Notepad shopping list | stall reset and loop stop at step 4: typing changed 0.03 – 0.08% of the screen |

The cause is the screen diff, not the guards: at 480×270 grayscale, a few typed characters
fall under the 0.2% progress threshold, and the subgoal shortcut is what currently keeps
such runs alive. A finer change measure (around the action point, or at higher resolution)
would be needed before the shortcut could be narrowed; that is a separate proposal.

## 7. Alternatives considered

- **Stop clearing `repeated_action_count` on progress.** One line, but every scroll after
  the first would count as a repeat, so long scrolls and paging get blocked. Rejected.
- **Novelty against `recent_actions`** (the 6-entry window, not cleared on progress). Same
  problem as above: it can't tell a fifth Notepad launch from a fifth scroll.
- **Key on `page_signature`.** Breaks on the Computer Use backend (§4).
- **Ignore call-derived subgoals in the progress rule** (or strip arguments from the
  Computer Use subgoal). Closes the wider gap where any change in arguments counts as
  progress, but replayed on saved runs it stops 2 of 8 successful runs (§6); stripping the
  arguments also brings back the bug `_subgoal_for_calls` exists to prevent (a form's
  second `type_text_at` blocked as an already-completed subgoal).
- **A per-app "already launched" rule.** Fixes Notepad only; this is the general case and
  per-app rules belong in site adapters.
