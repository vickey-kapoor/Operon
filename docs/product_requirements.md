# Operon Product Requirements
_Last refreshed: 2026-05-15_

## Product Goal

Operon is a vision-first computer-use agent for browser and desktop tasks. It observes screenshots, chooses coordinate-based actions, executes through native/browser executors, verifies the result, and recovers when progress stalls.

## Core Requirements

1. The agent must operate from visual perception, not DOM selectors or XPath.
2. Browser and desktop runs must share the same loop, verifier, recovery manager, and persistence model.
3. Deterministic policy rules must run before LLM policy fallback.
4. TYPE actions must remain atomic from the policy perspective.
5. Clicks must pass visual servo checks before execution.
6. Runs must produce inspectable artifacts under `.var/runs/<run_id>/` by default, with `OPERON_RUNS_ROOT` available as an override.
7. Uncertain states must either recover deterministically or pause for human input.

## Supported Modes

| Mode | Executor | Primary backend | Status |
|---|---|---|---|
| Browser | `NativeBrowserExecutor` | Gemini Computer Use | Active |
| Browser JSON | `NativeBrowserExecutor` | Gemini JSON combined backend | Optional/fallback |
| Desktop | `DesktopExecutor` | Gemini JSON combined backend | Active |
| Observable browser | `NativeBrowserExecutor` + `BrowserManager` | Browser backend | Active through `/ws/stream` |

## API Requirements

Active HTTP endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/run-task` | Create browser run |
| POST | `/step` | Advance browser run |
| POST | `/resume` | Resume paused browser run |
| POST | `/stop` | Cancel active browser run |
| POST | `/run/{run_id}/stop` | Cancel by path |
| POST | `/run/{run_id}/pause` | Pause by path |
| GET | `/run/{run_id}` | Read browser run state |
| POST | `/cleanup` | Clean up browser resources |
| GET | `/health` | Health check |
| POST | `/desktop/run-task` | Create desktop run |
| POST | `/desktop/step` | Advance desktop run |
| POST | `/desktop/resume` | Resume desktop run |
| POST | `/desktop/cleanup` | Clean up desktop resources |
| GET | `/desktop/run/{run_id}` | Read desktop run state |
| GET | `/observer/api/runs` | List recent runs |
| GET | `/observer/api/run/{run_id}` | Load run snapshot |
| GET | `/observer/api/artifact` | Serve run artifact |
| GET | `/observer/api/usage` | Summarize token usage |
| GET | `/observer/api/export/{run_id}` | Export run bundle |
| GET | `/observer/api/live-browser/{run_id}` | Return active browser frame |
| WS | `/ws/stream` | Live frames, events, and controls |

Removed or absent from the current FastAPI app:
- `/benchmark/*`
- `/benchmarks`
- `/command-center`
- `/console`
- `/dashboard`
- static UI routes

## Action Requirements

The internal policy action model supports pointer, keyboard, navigation, clipboard, screenshot, upload, read-text, batch, stop, wait, and HITL actions. The contract-layer action model in `src/operon/core/contracts/planner.py` is narrower and is used by executor adapters and route validation.

Important product constraints:
- `upload_file_native` is headed-mode only and uses the OS picker macro.
- Browser/desktop route compatibility is validated in `src/operon/core/router.py`.

## Verification Requirements

The verifier must classify outcomes as:

```text
SUCCESS
FAILURE
UNCERTAIN
PENDING
PROGRESSING_STABLE
STABLE_WAIT
```

Loop behavior:
- `SUCCESS`: advance or terminate.
- `FAILURE` / `UNCERTAIN`: recovery ladder.
- `PENDING`: wait and re-verify with 2/4/8 second backoff.
- `STABLE_WAIT`: wait 200ms and re-verify once.
- `PROGRESSING_STABLE`: advance immediately.

## Recovery Requirements

`RuleBasedRecoveryManager` must:
- stop when verifier reports a terminal condition.
- advance only on verified success or confirmed progress.
- hard-stop repeated no-progress loops.
- escalate repeated failure clusters through retry, different tactic, context reset, session reset, then stop.
- block unverified terminal success claims.

## Persistence Requirements

Every run should persist:
- current run state
- step log JSONL
- before/after screenshots
- perception artifacts
- policy artifacts
- execution traces
- verification and progress artifacts where available

Retention cleanup is handled at FastAPI lifespan startup through `operon.store.cleanup.cleanup_old_runs`.

## Optional Components

These are part of the repository but not required for the core agent:


## Known Gaps

- The repository contains historical docs that describe removed systems. Current implementation truth should be taken from `README.md`, `AGENTS.md`, this PRD, and `docs/architecture.md`.
- There is no active benchmark API route set in FastAPI.
- There is no tracked `PostRunReflector` implementation.
- There is no tracked `src/runtime` package.
- The current FastAPI router does not serve a bundled frontend.
