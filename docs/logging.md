# Logging and diagnostics

This page describes the logging and diagnostic data Operon currently produces,
where it is stored, and the gaps found in a code review. It describes current
behavior and recommends additional events to capture.

## Logging layers

| Layer | What it records | Where it goes |
|---|---|---|
| Python application logs | Startup/configuration warnings, model retries, executor and cleanup errors, task lifecycle events | Standard output/error through the server's Uvicorn logging setup. `operon run` configures console logging at `ERROR` when no handler exists. |
| Loop trace | Capture, perception, policy, gate, execution, verification, recovery, and finish stages | Console when `OPERON_TRACE=1`; trace output is printed directly rather than sent through Python logging. |
| Structured run log | Step perception, chosen and executed actions, verification/recovery results, failure records, and timing | `runs/<run_id>/run.jsonl`; written asynchronously for normal steps and synchronously for critical pre-step failures. |
| Run state and metrics | Current state, bounded recent history, status, recovered-error counts, and end-of-run efficiency | `state.json`, `efficiency.json`, and related files in the run folder. |
| Debug artifacts | Model prompts, raw and parsed responses, usage, retries/diagnostics, execution and progress traces, screenshots, and optional task notes | Per-run `step_<n>/` folders. The exact model artifact names vary by backend. |
| Screen recordings | Browser session videos; desktop full-run recordings | Browser recordings are under `.var/browser-artifacts/<run_id>/`; desktop recordings are stored in the run folder. |

The observer API can summarize usage and export a run bundle. See the
[architecture observability overview](architecture.md#observability) and
[configuration reference](configuration.md) for the available interfaces and
settings.

## Retention and sensitive data

In a source checkout, generated data defaults to `.var/`; installed runs use
`~/.operon/`. `OPERON_RUNTIME_ROOT`, `OPERON_RUNS_ROOT`, and the browser and
desktop artifact root settings can change these locations. `.var/` and the
legacy artifact directories are ignored by Git.

The server's startup retention sweep defaults to 14 days and deletes old run
directories containing `state.json`. `OPERON_RUNS_RETAIN_DAYS=-1` disables this
sweep. The same setting covers expired browser videos in the separate browser
artifact root. The video sweep keeps recordings with a recent run state or a
live recorded owner. Run folders can contain screenshots, prompts, model replies,
task text, actions, and URLs. Review artifacts before exporting or sharing them.

## Logging improvements

- The startup sweep covers browser videos as well as run folders.
- The trust-gate URL record and explicit browser session and network messages
  record the site origin. Detailed run artifacts still need review before sharing.
- Background artifact write and append failures emit warnings.
- `operon serve --log-level` sets the Operon and Uvicorn log levels, with
  `info` as the default. Capture console output through the service manager
  or deployment environment if you need durable logs.

## What to add next

1. **Approval decision sources.** The approval audit trail records requests and
   their outcomes in `approval_audit.jsonl` and exposes them in the observer's
   run detail response. It stores request IDs, timestamps, run and step IDs,
   categories, and outcomes. Approval endpoints currently share the same path
   for interactive and automated callers, so the trail cannot identify the
   decision maker. A process crash can leave a request without an outcome.
2. **A consistent run lifecycle record.** Record start and end events for every
   run, including cancellation and interruption, with the outcome, duration,
   and reason. Today terminal rows only cover selected paths. Start here if
   the main need is reliable run counts and incident timelines.
3. **A per-run artifact health summary.** Record how many writes failed and
   which artifact types were affected. This would let the observer distinguish
   missing evidence from artifacts that were never expected.

