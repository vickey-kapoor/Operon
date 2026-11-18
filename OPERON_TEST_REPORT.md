# Operon test report

Branch `dev/local-build`, 2026-11-17, Windows 11 Pro, Python 3.14, 8 GB RAM, Intel UHD graphics.
Everything here ran locally with no network model calls (`GEMINI_API_KEY=fake-test-key`).

## Results

| Layer | Command | Result |
|---|---|---|
| Unit and integration (offline) | `.venv\Scripts\python -m pytest tests -q` | **2782 passed**, 18 skipped, 367 deselected, 3 xfailed (161 s) |
| Scripted scenarios through the real loop | `.venv\Scripts\python -m evals` | **22/23 correct**, 0 false success, 0 crash |
| Real Chromium end to end | `.venv\Scripts\python -m pytest tests/browser_e2e -m browser_e2e -o addopts=""` | **15 passed**, 1 xfailed (163 s) |
| Real Windows desktop end to end | `$env:OPERON_SANDBOX_MEMORY_MB="2048"; .venv\Scripts\python -m pytest tests/desktop_e2e_windows -m windows_sandbox -o addopts=""` | **3 passed**, twice in a row (147 s, 109 s) |
| Lint | `.venv\Scripts\ruff check src tests evals --select E,F,W,I --ignore E501` | clean |
| Type check | `.venv\Scripts\mypy` | clean, 160 files |

Not run here: the Linux desktop container suite (`-m desktop_container`, Docker was not
running), `live_server` tests (need a running API and a real key), paid benchmarks.

## What the end-to-end suites prove

The real-Chromium suite runs the production loop, harness, Gemini computer-use backend and
browser executor on headless Chromium against a local deterministic site. Only the model client
is scripted, and a pixel-comparing critic stands in for the critic model. Every task is graded
from the site's own state or request log, never from the model's claims.

| # | Scenario | Result |
|---|---|---|
| 1 | Navigate a multi-page site | pass |
| 2 | Fill and submit a form | pass |
| 3 | Click the right button among five | pass |
| 4 | Scroll to an off-screen element and click | pass |
| 5 | Dropdown and modal dialog | pass |
| 6 | Content that appears after a delay | pass |
| 7 | Recover from a missed click | pass |
| 8 | Sign in, add to cart, check out | pass |
| 9 | Create a file through the tool channel and check it | pass |
| 10 | Ignore "done" claims made before any work | pass |
| 10b | Reject "done" after a wrong click without the critic | strict xfail: known limit |
| 11 | Impossible task ends FAILED within budget, no false success | pass |
| 12 | Repeated useless clicks stopped by the loop guard; waste scored | pass |

Reliability: scenarios 2, 7 and 8 run five times each in one test; 15 of 15 passed (form
4.9-7.7 s, recovery 3.5-7.9 s, workflow 6.8-21 s per run; the upper ends were under CPU load).

Windows Sandbox suite (real Windows desktop, real loop, scripted model): host capture of the
real sandbox desktop; an app outside the catalog never starts; a command typed into `cmd`
writes a file the test reads back. All 3 pass. The first version typed into Notepad and failed:
the sandbox image has no Notepad (a Store app on Windows 11), yet `launch_app` reported
success. That false success is an Operon bug, now being fixed (see `OPERON_PROGRESS.md`).

## Reliability thresholds used

- Deterministic suites (offline, evals, browser e2e): 100% pass required; a single failure
  blocks a merge.
- Evals: 0 false successes required; the one known miss (`cu_relaunch_runaway`, RFC 0003) is
  labelled in the eval table.
- Browser e2e reliability: 5/5 per repeated scenario.
- Windows Sandbox: needs a quiet machine; under heavy CPU load the sandbox closed before its
  guest started (environmental, diagnosed; the provider now reports this clearly and retries
  once).

## Metrics the evals report

`python -m evals` prints, per scenario: outcome, steps, pauses, model calls and a waste column
(wasted steps / steps, from the step-efficiency score). Example: the runaway scenario scores
10/12 wasted; scroll and form scenarios score 0. `python -m evals --json` gives the same as JSON.

## Unresolved limitations

- No real model reasoning was tested (no local model runtime or GPU; paid runs not authorised in
  this session). Mock-model results prove the pipeline, not task success rates with a real model.
- See `OPERON_PROGRESS.md` for the full list.
