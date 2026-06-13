# Operon Architecture Reference
_Last refreshed: 2026-06-13_

This document describes the code currently present in this repository. It is the
single canonical engineering reference; the product "why" lives in
[`PRD.md`](./PRD.md) and contributor workflow in [`../AGENTS.md`](../AGENTS.md).

## Active Runtime Shape

```text
FastAPI app
  -> operon.api.routes
  -> AgentLoop
  -> ScreenCaptureService
  -> Perception/policy backend
  -> PolicyCoordinator
  -> executor adapter or native executor
  -> DeterministicVerifierService
  -> RuleBasedRecoveryManager
  -> FileBackedRunStore
```

Browser and desktop runs share `operon.agent.loop.AgentLoop`. The API creates one lazy singleton for browser runs and one for desktop runs.

## Package Layout

```text
src/
  agent/      loop, perception, policy, policy rules, recovery, verifier,
              browser backends, fallback backend, targeting, progress tracking
  api/        FastAPI server, routes, observer endpoints, runtime config, websocket stream
  browser/    CDP BrowserManager for observable browser mode
  clients/    Gemini, Gemini Computer Use, and Anthropic HTTP clients
  core/       shared Environment enum used to select the execution path
  executor/   browser, desktop, adapters, native upload macro
  models/     Pydantic schemas for state, perception, policy, execution, logs, memory
  store/      run persistence, memory, replay, cleanup, background writer
```

## Control Loop

`AgentLoop.step_run()` runs the current step. At a high level:

1. Prepare step artifact paths.
2. Capture the current screen through `ScreenCaptureService`.
3. Clear rolling element memory when `visual_velocity > 0.05`.
4. Perceive the screen through the configured backend.
5. Persist perception into `AgentState`.
6. Choose an action with `PolicyCoordinator`.
7. Attach target context and monitor-origin coordinate offsets.
8. Execute through the environment-specific executor.
9. Verify with `DeterministicVerifierService`.
10. Handle `PENDING` and `STABLE_WAIT` re-verification loops.
11. Recover or advance with `RuleBasedRecoveryManager`.
12. Persist step logs, progress traces, and run state.

## Runtime Modes

### Browser

Default backend: `BrowserComputerUseBackend` using `GeminiComputerUseHttpClient`.

Optional path:
- `BrowserJsonBackend` fallback when configured.

Executor: `NativeBrowserExecutor`.

### Desktop

Default backend: `CombinedPerceptionPolicyService` using Gemini JSON output.

Executor: `DesktopExecutor`.

Only `OPERON_DESKTOP_BACKEND=json` is implemented.

### Observable Browser Mode

`RunTaskRequest.mode` accepts `batch` or `observable`. Observable browser mode uses `BrowserManager` to attach to Chrome over CDP, publish JPEG frames through `/ws/stream`, and accept `inject_input` control messages.

## API Routes

The active HTTP routes are registered from `src/operon/api/routes/`.

```text
POST /run-task
POST /step
POST /resume
POST /stop
POST /run/{run_id}/stop
POST /run/{run_id}/pause
GET  /run/{run_id}
POST /cleanup
GET  /health

POST /desktop/run-task
POST /desktop/step
POST /desktop/resume
POST /desktop/cleanup
GET  /desktop/run/{run_id}

GET  /observer/api/runs
GET  /observer/api/run/{run_id}
GET  /observer/api/artifact
GET  /observer/api/usage
GET  /observer/api/export/{run_id}
GET  /observer/api/live-browser/{run_id}

WS   /ws/stream
```

There are no active `/benchmark/*`, `/command-center`, `/console`, `/dashboard`, or static HTML routes in `src/operon/api/routes/`.

## Policy Layer

`PolicyCoordinator` wraps:
- memory hint lookup from `FileBackedMemoryStore`
- `PolicyRuleEngine`
- optional episode advisory hints
- the configured policy delegate
- post-LLM guards such as `_semantic_anchor_check`

Earlier benchmark-plugin registration scaffolding has been removed from `PolicyRuleEngine`; benchmark-specific behavior must not be hardcoded into core policy rules.

Current built-in policy rules include:
- human intervention detection
- task-success stop
- desktop app launch preference
- visible form field fill
- dropdown selection
- identical type retry avoidance
- no-progress recovery
- blocking overlay dismissal
- search query handling

## Verification

`DeterministicVerifierService` returns:

```text
SUCCESS
FAILURE
UNCERTAIN
PENDING
PROGRESSING_STABLE
STABLE_WAIT
```

`AgentLoop` handles:
- `PENDING`: waits 2s, 4s, then 8s and re-verifies.
- `STABLE_WAIT`: waits 200ms and re-verifies once.
- `PROGRESSING_STABLE`: advances because the UI reacted.

The verifier is deterministic and does not use a separate video verifier module.

## Recovery

`RuleBasedRecoveryManager` (`src/operon/agent/policy/recovery.py`) maps failed or
uncertain steps to recovery decisions, keyed by a retry cluster built from subgoal,
target, and failure signal.

**Bypass cases** (decided before escalation):

| Condition | Decision |
|---|---|
| `verification.stop_condition_met` | `STOP` |
| `FailureCategory.EXECUTION_NO_PROGRESS` | terminal `STOP` |
| `VerificationStatus.SUCCESS` with expected outcome met | `ADVANCE` |
| `VerificationStatus.PROGRESSING_STABLE` | `ADVANCE` |
| TYPE failed because target was missing/not editable | `WAIT_AND_RETRY` (focus-oriented subgoal) |

**Escalation by failure cluster:**

| Attempt | Strategy | Wait |
|---|---|---|
| 1 | `RETRY_SAME_STEP` | optional |
| 2 | `RETRY_DIFFERENT_TACTIC` | optional |
| 3 | `CONTEXT_RESET` | 1000 ms |
| 4 | `SESSION_RESET` | 1500 ms |
| 5+ | terminal `STOP` (`RETRY_LIMIT_REACHED`) | none |

**Integrity guard.** `validate_benchmark_integrity()` prevents recovery from claiming
terminal success unless verification actually returned `SUCCESS`, and prevents
`ADVANCE` when the verifier set a stop boundary without success. The recovery manager
only chooses the strategy; the loop applies it, and each executor implements its own
context/session reset semantics.

## Upload Action Paths

Two browser upload actions exist, both on `NativeBrowserExecutor`:

| Action | Mechanism | Headless-safe |
|---|---|---|
| `upload_file` | Playwright file-chooser interception (sets the file directly, bypassing the OS picker) | Yes |
| `upload_file_native` | Clicks the visual target, then drives the native OS picker via `os_picker_macro.py` (type path, Enter) | No (headed only) |

`upload_file_native` is a browser action; the OS picker macro uses desktop input
primitives internally to drive the native file dialog. Failure signals include
`PICKER_NOT_DETECTED`, `FILE_NOT_REFLECTED`, and `EXECUTION_ERROR` (which covers the
headed-mode requirement).

## Persistence

Run data is file-backed:

```text
.var/runs/<run_id>/
  state.json
  run.jsonl
  step_N/
    before.png
    after.png
    perception artifacts
    policy artifacts
    execution_trace.json
    progress_trace.json
    verification_result.json
```

Memory is stored under `memory/` when enabled by the store:

```text
memory/memory.jsonl
memory/episodes.jsonl
```

Browser session videos are written under `.var/browser-artifacts/` by default. Desktop artifacts use `.var/desktop-artifacts/`. Set `OPERON_RUNTIME_ROOT`, `OPERON_RUNS_ROOT`, `OPERON_BROWSER_ARTIFACTS_ROOT`, or `OPERON_DESKTOP_ARTIFACTS_ROOT` to override these locations.

## Configuration

Core runtime configuration is in `src/operon/api/runtime_config.py`.

Important variables:
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPERON_BROWSER_BACKEND`
- `OPERON_BROWSER_FALLBACK_BACKEND`
- `OPERON_BROWSER_MODEL`
- `OPERON_DESKTOP_BACKEND`
- `OPERON_DESKTOP_MODEL`
- `OPERON_BROWSER_PLANNER_PROVIDER`
- `OPERON_DESKTOP_PLANNER_PROVIDER`
- `OPERON_BROWSER_VERIFIER_PROVIDER`
- `OPERON_DESKTOP_VERIFIER_PROVIDER`
- `OPERON_TRACE`
- `OPERON_TEST_SAFE_MODE`
- `BROWSER_HEADLESS`
- `BROWSER_WIDTH`
- `BROWSER_HEIGHT`

## Invariants

- Policy and perception must remain vision-first.
- Deterministic rules run before LLM fallback.
- TYPE remains atomic at the executor level.
- Clicks use visual servo checks before input injection.
- Rolling element memory is cleared on high visual velocity.
- Benchmark-specific behavior must not be hardcoded into core policy rules.

## Not Present / Intentionally Removed

To prevent re-introduction, the current tree intentionally does **not** include:

- `/benchmark/*`, `/command-center`, `/console`, `/dashboard`, or static-HTML/bundled-frontend routes.
- A `src/benchmarks` plugin package or policy-rule plugin registry.
- A `PostRunReflector` that auto-generates reflection records.
- A separate `src/runtime` orchestrator package or a standalone video-verifier module.

Current implementation truth is this document, `README.md`, and `AGENTS.md`.
