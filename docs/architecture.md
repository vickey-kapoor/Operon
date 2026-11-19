# Operon Architecture Reference
_Last refreshed: 2026-11-19_

This document describes the code currently present in this repository. It is the
single canonical engineering reference; contributor
workflow in [`../AGENTS.md`](../AGENTS.md).

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

Browser and desktop runs share `operon.agent.loop.AgentLoop`. The API builds one loop for browser runs and one for desktop runs on first use, and keeps both on the process runtime (`operon.core.runtime.OperonRuntime`, KAN-156). The runtime also holds a `RunContext` per run for state that belongs to that run: its pending override hint, its auto-run task, and its start and finish times. See [Runtime and State](#runtime-and-state).

The perception and policy path never reads the DOM, CSS selectors, XPath or the accessibility
tree; targets are pixel coordinates. `tests/test_invariants.py` enforces this statically by
scanning `agent/perception`, `agent/policy` and `agent/backends` for identifiers such as
`locator`, `query_selector`, `get_by_role` and `accessibility`. Structured machine state is read
only by graders after a run (`evals/desktop_state.py`, `evals/hybrid_state.py`), never by the
agent. The reason is portability: the same loop works on any app that draws pixels, and bad
targeting is fixed by improving perception rather than by reaching into app internals.

## Package Layout

```text
src/
  agent/      loop, perception, policy, policy rules, recovery, verifier,
              browser backends, fallback backend, targeting, progress tracking
  api/        FastAPI server, routes, observer endpoints, runtime config, websocket stream
  browser/    CDP BrowserManager for observable browser mode
  clients/    Gemini and Gemini Computer Use HTTP clients
  core/       shared Environment enum used to select the execution path
  executor/   browser, desktop, adapters, native upload macro
  models/     Pydantic schemas for state, perception, policy, execution, logs, memory
  store/      run persistence, memory, replay, cleanup, background writer
```

## Control Loop

`AgentLoop.step_run()` (`agent/loop.py`, about 3,000 lines) runs one step per call, so runs are
resumable and pausable between steps. It takes a per-run lock and the loop's `_executor_lock`
(one executor serves one step at a time), then calls `_step_run_locked`, which chains stage
methods over a `StepContext` (`agent/step_context.py`, KAN-155). Fields a stage sets are
`field(init=False)`, so reading one early raises.

| Order | Stage (`agent/loop.py`) | What it does |
|---|---|---|
| 0 | `_begin_step` | Refuses runs that are not PENDING or RUNNING; enforces `max_steps` (default 25, `models/common.py`); builds the `StepContext` |
| 1 | `_warmup_stage` | Step 1 only: a browser run without a start URL navigates to google.com if the trust gate allows it; 2.5 s settle (skipped in test safe mode) |
| 1b | `_consume_force_fresh_perception` | When the run record has `force_fresh_perception` set, waits for the UI to settle (1.0 s in a browser, 0.5 s on the desktop; `OPERON_SETTLE_DETECT` swaps the fixed sleep for settle detection), then clears the flag |
| 2 | `_capture_stage` | `ScreenCaptureService.capture` (`agent/perception/capture.py`) asks the executor for a frame and moves it to `step_N/before.png` |
| 3 | `_consume_override_hint`, `_add_task_note`, `_add_secrets_hint` | Queue advisory text hints for the model: a user correction, `OPERON_TASK_NOTE`, secret names |
| 4 | `_perceive_stage` | Memory hints, stagnation hint, then `perception_service.perceive` (with a liveness retry); stamps the monitor origin; saves the observation |
| 5 | `_decide_stage` | `policy_service.choose_action`; `_attach_target_context` (grounder, monitor origin); Best-of-N (`OPERON_BESTOFN_N`) |
| 6 | `_gate_stage` | Cancel check, then `_gate_pause_decision`: the model asked for a human, coordinate thrash (3 failures at one point), confidence threshold, trust gate, file guard. A blocked action becomes WAIT_FOR_USER and the run pauses |
| 7 | `_execute_and_verify_stage` | A loop of `_execute_attempt` then `_verify_attempt`. On a hard FAILURE, `_prepare_adaptation_retry` re-captures, re-perceives, re-decides and re-gates (at most `_MAX_ADAPTATION_RETRIES` = 3; never in thin) |
| 8 | `_recover_stage` | `recovery_manager.recover`, then `validate_benchmark_integrity`; progress tracking and `progress_trace.json` |
| 9 | `_record_step_stage`, `_finish_step_stage` | Writes the `StepLog` to `run.jsonl`; sets SUCCEEDED, FAILED, RUNNING, CANCELLED or WAITING_FOR_USER; applies CONTEXT_RESET and SESSION_RESET side effects; writes `step_timing.json`; on a terminal status writes `efficiency.json` |

### One action end to end

A Gemini computer-use click in batch browser mode:

1. **Capture.** `NativeBrowserExecutor.capture` (`executor/browser_native.py`) calls Playwright
   `page.screenshot` on the run's own batch Chromium. The frame size is the viewport
   (`BROWSER_WIDTH` and `BROWSER_HEIGHT`, default 1920x1080). The loop moves the frame to
   `.var/runs/<id>/step_N/before.png`.
2. **Perceive and decide in one call.** `BrowserComputerUseBackend.perceive`
   (`agent/backends/browser_computer_use.py`) renders the prompt, then
   `GeminiComputerUseHttpClient` (`clients/gemini_computer_use.py`) sends the PNG as is (no
   downscale on the Gemini path) with the `computerUse` tool plus any `OPERON_TOOLS` function
   declarations, at temperature 0. A model error is re-raised as `BackendCompatibilityError`.
   With `OPERON_ZOOM=dense` (the code default) a click is refined on a crop in one extra call,
   but only when the screen's long side is at least `OPERON_ZOOM_MIN_SIDE` (2000), so not on a
   1920x1080 viewport (`agent/actions/zoom.py`). The prompt, raw reply, parsed decision, answers
   and usage go to `step_N/computer_use_*.{txt,json}`.
3. **Scale coordinates back.** `normalize_computer_use_actions`
   (`agent/actions/translation.py`) maps Gemini's 0-999 grid to pixels with `_scale_grid_coord`
   (`value / 999 * (size - 1)`) against the frame size. Several calls in one turn become one
   `BATCH`; a `click_at` followed by an untargeted `type` is merged into one atomic TYPE at the
   click point. A `require_confirmation` safety decision holds all calls and pauses first.
4. **Policy.** In the full harness, `PolicyCoordinator.choose_action`
   (`agent/policy/coordinator.py`) runs `PolicyRuleEngine` first, then the cached model
   decision, then `_reject_premature_stop`. The loop's `_attach_target_context` applies the
   grounder (identity for the default `deterministic` grounder) and the monitor origin (0,0 for
   a browser).
5. **Gate.** `_pre_execution_gate_text` evaluates each leaf of a batch with
   `TrustGate.evaluate` (`agent/policy/trust_gate.py`), passing the current URL and the model's
   rationale, then `apply_permission_mode`. A DENY anywhere outranks a CONFIRM. Then the file
   guard (`agent/policy/file_guard.py`).
6. **Execute.** `_execute_with_hardening` runs the action through the unified adapter into
   `NativeBrowserExecutor.execute`, which does `page.mouse.click(x, y)`. The browser executor has
   no visual servo. On a retryable execution failure (full harness only) the loop re-perceives
   in perception-only mode, re-resolves the target and retries once.
7. **Verify.** `DeterministicVerifierService.verify` (`agent/policy/verifier.py`) returns a
   status. `PENDING` re-verifies after 2 s, 4 s and 8 s (`_wait_for_page_load`); `STABLE_WAIT`
   waits 200 ms, re-captures, adopts the new frame as `after.png` and re-verifies once with
   `allow_stable_wait=False` (`_wait_for_ui_stable`).
8. **Recover.** `RuleBasedRecoveryManager` (`agent/policy/recovery.py`) picks the next move.
9. **Persist.** The step log, progress trace and run state are written (see
   [Persistence](#persistence)).

On the desktop the same flow uses `DesktopExecutor.capture` (`executor/desktop.py`): an `mss`
burst of the foreground monitor (three grabs 100 ms apart, `visual_velocity` from pixel change,
one re-capture if velocity is above 0.02) at native resolution. The monitor's origin is stamped
on the perception (`_with_capture_origin`) and added to the action (`_apply_monitor_origin`) so
clicks land on secondary monitors. Every click variant goes through `_click_preamble`: a
monitor-bounds check and the visual servo `_region_has_content` (pixel variance in a 50 px
radius against a threshold calibrated at init by `_calibrate_servo_threshold`). A blank region
fails the click with `EXECUTION_TARGET_NOT_FOUND`. Input is `pyautogui`.

## Runtime Modes

### Browser

Default backend: `BrowserComputerUseBackend` using `GeminiComputerUseHttpClient`
(`computer_use`, `gemini-3.5-flash`). `OPERON_BROWSER_BACKEND=json` selects `BrowserJsonBackend`
alone. Gemini is the default model; other models (Claude, OpenAI, open-weight) are selectable
with `OPERON_BROWSER_BACKEND=model` and a profile from the model registry.

Fallback: with `OPERON_BROWSER_FALLBACK_BACKEND=json` (the default), `FallbackBackend`
hands a step to the Gemini `BrowserJsonBackend` when the primary backend's call fails.

Executor: `NativeBrowserExecutor`.

### Desktop

Default backend: the Gemini computer-use desktop backend (`desktop_computer_use.py`,
`OPERON_DESKTOP_BACKEND=computer_use`, `gemini-3.5-flash`), which adds a `launch_app` function.
`OPERON_DESKTOP_BACKEND=json` selects the older `CombinedPerceptionPolicyService` using Gemini
JSON output. The computer-use backend has no fallback.

Executor: `DesktopExecutor`.

### Harness profiles

`OPERON_HARNESS_PROFILE` (`harness_profile()` in `agent/policy/thin.py`) picks how much
scaffolding surrounds the model per backend; the layers are wired in
`api/runtime/loops.py:_loop_layers`. Gemini `computer_use` runs **full** by default.
`OPERON_HARNESS_PROFILE=thin` switches it to **thin**: `ThinPolicy` passes Gemini's action
through with no rules, guards or retries, and its own "done" ends the run (`OwnDoneVerifier`),
so the benchmark graders decide success. `json` backends always run full; the `model` backend
(Claude, OpenAI, open-weight) always runs thin. `OPERON_CRITIC=same` makes the full harness's
step critic the acting model. [configuration.md](configuration.md) has the details.

| | Full (default for Gemini `computer_use`; always for `json`) | Thin (`OPERON_HARNESS_PROFILE=thin`; always for `model`) |
|---|---|---|
| Policy | `PolicyCoordinator`: memory hints, `PolicyRuleEngine` (two rules: `_task_success_stop_rule`, and `_prefer_launch_app_rule` on desktop), the model, `_reject_premature_stop` | `ThinPolicy`: the backend's decision unchanged |
| Verifier | `DeterministicVerifierService` with a Gemini critic (`OPERON_*_VERIFIER_MODEL`, or the acting model with `OPERON_CRITIC=same`) | `OwnDoneVerifier`: failures go back to the model as tool results; the model's "done" ends the run |
| Recovery | `RuleBasedRecoveryManager` ladder plus the progress stop guard | `ThinRecovery`: always hand back to the model |
| Loop extras | Redundant-action blocking, execution retry, adaptation retry, grounder refinement | All skipped (`self.thin_harness` checks in `loop.py`) |
| Browser fallback | `FallbackBackend` to JSON when configured | Ignored |

Both profiles keep the trust gate, file guard, coordinate-thrash pause and step budget. Advisory
text (task note, secret names, change note) is allowed in thin because it changes what the model
sees, not its action.

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

The front door routes (`routes/frontdoor.py`, prefix `/app/api`, see `docs/api.md`) sit on one
service (`api/frontdoor.py`) that `operon run` shares, so both behave the same:

```text
POST /app/api/runs
GET  /app/api/runs
GET  /app/api/runs/{run_id}
GET  /app/api/runs/{run_id}/report
POST /app/api/runs/{run_id}/approve
POST /app/api/runs/{run_id}/decline
POST /app/api/runs/{run_id}/pause
POST /app/api/runs/{run_id}/takeover
POST /app/api/runs/{run_id}/handback
POST /app/api/runs/{run_id}/stop
POST /app/api/runs/{run_id}/input
POST /app/api/runs/{run_id}/resume-interrupted
GET  /app/api/interrupted
POST /app/api/stream-token
```

The browser and desktop routes also have `POST /run/{run_id}/resume-interrupted` and
`POST /desktop/run/{run_id}/resume-interrupted`, `POST /desktop/stop` and
`POST /desktop/run/{run_id}/stop`; the observer routes add `GET /observer/api/settings`.
Auth is an `X-API-Key` header, from `API_KEYS` or a key generated in `.var/api-key`.

There are no active `/benchmark/*`, `/command-center`, `/console`, `/dashboard`, or static HTML routes in `src/operon/api/routes/`.

## Policy Layer

`PolicyCoordinator` wraps:
- memory hint lookup from `FileBackedMemoryStore`
- `PolicyRuleEngine`
- the configured policy delegate
- the post-LLM guard `_reject_premature_stop` (re-plans a STOP made before any real action)

Earlier benchmark-plugin registration scaffolding has been removed from `PolicyRuleEngine`; benchmark-specific behavior must not be hardcoded into core policy rules.

Current built-in policy rules:
- task-success stop (a success phrase in the model's own text; the model confirms it first)
- desktop app launch preference (Desktop and Hybrid only)

Gemini Computer Use reports no element list and no page type, so rules that needed either
could never fire. KAN-161 removed them (form fill, dropdown selection, overlay dismissal,
search query, no-progress recovery, identical type retry, page-type human intervention) along
with the element-based guards `_semantic_anchor_check` and `_reject_hallucinated_target`.

## Verification

`DeterministicVerifierService` returns one of six states:

| Status | Produced by | Loop handling |
|---|---|---|
| SUCCESS | Terminal-state evidence, an accepted STOP, a goal-completing action, the critic, or a fallback when confidence is at least 0.5 | Advance; a SUCCESS stop with a success stop reason ends the run SUCCEEDED |
| FAILURE | Executor failure, or a rejected STOP (suspicious early stop, critic rejection, the model gave up) | In-step adaptation retry (full harness), then recovery |
| UNCERTAIN | A passive wait without progress, low confidence with no critic, a PENDING that never settles | Recovery |
| PENDING | The page looks like it is loading | Re-verify after 2 s, 4 s, then 8 s |
| STABLE_WAIT | The action visibly changed the screen | 200 ms settle, one re-verify |
| PROGRESSING_STABLE | Defined in `models/verification.py` and handled (recovery advances on it), but reserved: no verifier in `src/` currently returns it | Recovery advances |

The verifier is deterministic and does not use a separate video verifier module.

A `STOP` is the policy's claim that the task is done, and it is verified like any
other outcome: the critic judges the current screen when it can; if the critic is
unavailable or unsure, the stop is accepted only when the run has done some work
and its latest step did not fail. A rejected stop is a `FAILURE` that goes to recovery.
Only the policy's own STOP ends a run as done: a critic stop signal on any other action is kept
as a note.

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
| 1 | `RETRY_SAME_STEP` | none |
| 2 | `RETRY_DIFFERENT_TACTIC` (the subgoal is rewritten) | none |
| 3 | `CONTEXT_RESET` (`executor.context_reset()`) | 1000 ms |
| 4 | `SESSION_RESET` (`executor.session_reset(start_url)`) | 1500 ms |
| 5 (`MAX_RECOVERY_ATTEMPTS`) | terminal `STOP` (`RETRY_LIMIT_REACHED`) | none |

In the full harness this ladder runs after every step the verifier did not accept. The thin
harness uses `ThinRecovery` instead, which always hands back to the model.

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

## Tool Actions

With `OPERON_TOOLS` (`on` by default since 2026-11-14, shell opt-in with `all`; [RFC 0002](rfcs/0002-tool-action-channel.md)), the Gemini
computer-use model may call `read_file`, `list_dir`, `write_file`, `http_get` and (with `all`)
`run_shell` next to its screen actions. A call becomes an `ActionType.TOOL` action, passes the same
gates as any action (trust gate, file guard), and `ToolExecutor` (`executor/tools.py`) runs it in
front of the environment's executor; the output goes back to the model as the answer to its call.
Perception does not change: the screen stays the only thing the model sees on its own.

## Runtime and State

- **Process state.** `OperonRuntime` (`core/runtime.py`, `current_runtime()`) holds
  `browser_loop`, `desktop_loop`, `run_loops` (parallel runs), the `run_slots` semaphore,
  `desktop_provider`, operator `ControlSettings` (confidence threshold, disabled rules),
  live-stream state, and `runs: OrderedDict[str, RunContext]`. Do not add module-level globals;
  tests patch `current_runtime()` fields.
- **Per-run state.** `RunContext` holds `started_at`, `finished_at`, `override_hint` and
  `auto_run_task`.
- **Loops.** Built lazily in `api/runtime/loops.py` (`get_agent_loop`, `get_desktop_agent_loop`).
  With `OPERON_PARALLEL_RUNS` > 1, each **batch** browser run gets a `ParallelRunLoop` with its
  own executor and a slot (`start_parallel_browser_run`); a full semaphore raises `RunSlotsFull`,
  answered as HTTP 429 with `Retry-After: 5`. Observable and desktop runs share one loop and
  take turns.
- **Settings.** `get_settings()` (`core/settings.py`, KAN-157) returns one typed `Settings`
  parsed from the environment and re-parsed only when a watched variable changes. Read fields
  there, never `os.getenv`. The settings in effect are saved per run as `settings.json`.
- **Run store.** `FileBackedRunStore` (`store/run_store.py`), with its root from
  `core/paths.py`. Writes go through a background writer (`store/background_writer.py`).
- **Recovered errors.** Every broad `except` in the loop goes through `AgentLoop._recovered()`,
  which logs with a traceback and counts per site (`recovered_errors`, KAN-158); the counts are
  merged into `state.json` after each step. `OPERON_STRICT_ERRORS` (on in tests, off otherwise)
  re-raises the bug exceptions instead.
- **Resumability.** Steps are separate calls, so a paused run (WAITING_FOR_USER) resumes with
  `resume_run`. With `OPERON_CHECKPOINTS=on`, API startup calls `mark_interrupted_runs`
  (`store/checkpoints.py`), which flags in-flight runs whose owner process is dead (pid plus
  creation time, `store/owner.py`) as WAITING_FOR_USER with `stop_reason=interrupted`. Nothing
  resumes on its own; `resume_interrupted` continues one from the current screen
  (`POST /run/{id}/resume-interrupted`, `operon resume <id>`, `operon runs --interrupted`).

## Persistence

Run data is file-backed, under `.var/runs/` in a source checkout and `~/.operon/runs/` when
installed with pip (`OPERON_RUNTIME_ROOT`, `OPERON_RUNS_ROOT`):

```text
.var/runs/<run_id>/
  state.json             AgentState snapshot (histories trimmed; the full history is in run.jsonl)
  run.jsonl              one StepLog per step, plus pre-step failure rows
  settings.json          effective settings at start, secrets left out
  host.json              os, home, cwd (the replay contract)
  displays.json          Windows only: DPI awareness and per-monitor scale
  owner.json             pid, process creation time, host of the owning process
  plan.json              when OPERON_PLANNER ran
  efficiency.json        step-efficiency score at the end of the run
  trust_gate_urls.jsonl  URLs the gate judged (only with a domain allowlist)
  step_N/
    before.png  after.png
    computer_use_prompt.txt  computer_use_raw.json  computer_use_parsed.json
    computer_use_answers.json  computer_use_usage.json
    policy_decision.json  verification_result.json  perception_parsed.json
    execution_trace.json  progress_trace.json  step_timing.json
    task_note.txt  bestofn.json   (when those features are on)
```

Memory is stored under `memory/` when enabled by the store:

```text
memory/memory.jsonl
```

Browser session videos are written under `.var/browser-artifacts/` by default. Desktop artifacts use `.var/desktop-artifacts/`. Set `OPERON_RUNTIME_ROOT`, `OPERON_RUNS_ROOT`, `OPERON_BROWSER_ARTIFACTS_ROOT`, or `OPERON_DESKTOP_ARTIFACTS_ROOT` to override these locations. Screenshots capture the whole screen, so review a run folder before sharing it.

## Observability

- **Per-step artifacts** are listed under [Persistence](#persistence), including the model's
  input and output per step. Usage is aggregated by `GET /observer/api/usage`; a run is exported
  as a zip by `GET /observer/api/export/{run_id}` (`api/observer.py`).
- **Trace.** `OPERON_TRACE=1` prints `[TRACE]` stage lines from `loop.py:_trace`.
- **Recovered errors.** `recovered_errors` in `state.json` shows how many times each broad
  `except` site swallowed an error (see [Runtime and State](#runtime-and-state)).
- **Efficiency** (`agent/efficiency.py`, KAN-85, measurement only): `total_steps`,
  `model_calls`, `actions_per_model_call`, `wall_seconds`, `seconds_per_step`, `wasted_steps` and
  `wasted_ratio`, with `blocked`, `failed`, `repeat`, `noop_wait` and `revisit` counts; saved to
  `efficiency.json` and the run state.
- **Live view.** Observable browser frames go over `/ws/stream` (short-lived tokens from
  `POST /app/api/stream-token`); Linux desktop and guest executors stream through
  `LiveViewMixin` (`executor/live_view.py`).

## Configuration

Core runtime configuration is in `src/operon/api/runtime_config.py`.

Important variables:
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`
- `OPERON_BROWSER_BACKEND`
- `OPERON_BROWSER_FALLBACK_BACKEND`
- `OPERON_BROWSER_MODEL`
- `OPERON_HARNESS_PROFILE`
- `OPERON_DESKTOP_BACKEND`
- `OPERON_DESKTOP_MODEL`
- `OPERON_BROWSER_VERIFIER_MODEL` / `OPERON_DESKTOP_VERIFIER_MODEL`
- `OPERON_TRACE`
- `OPERON_TEST_SAFE_MODE`
- `BROWSER_HEADLESS`
- `BROWSER_WIDTH`
- `BROWSER_HEIGHT`

### Default-off experimental flags

Experimental work ships behind an env flag that defaults to off. Defaults are from
`core/settings.py`, and every flag is in `.env.example`.

| Flag | Default | Effect |
|---|---|---|
| `OPERON_HARNESS_PROFILE` | empty (full) | `thin` runs Gemini computer use in the thin harness |
| `OPERON_CRITIC` | empty | `same` makes the acting model the step critic |
| `OPERON_BESTOFN_N` / `OPERON_BESTOFN_CONFIDENCE` | 1 / 1.0 | Sample N candidates at uncertain steps and run the best |
| `OPERON_GROUNDER` | `deterministic` | `snap` moves near-miss clicks onto the nearest element (full only) |
| `OPERON_PARALLEL_RUNS` | 1 | Batch browser runs that may step at once, each with its own Chromium |
| `OPERON_SETTLE_DETECT` / `OPERON_SETTLE_MAX_S` | off / 3.0 | Wait for the screen to stop changing instead of a fixed sleep |
| `OPERON_CHANGE_NOTE` | off | One text line about what changed while the screen settled |
| `OPERON_FOCUS_GUARD` | off | Skip the first action after a capture if focus was stolen (Windows) |
| `OPERON_DPI_AWARE` | off | Per-monitor-v2 DPI awareness before `pyautogui` loads (Windows) |
| `OPERON_PLANNER` / `OPERON_PLANNER_MODEL` | off / `gemini-3.5-flash-lite` | One text-only plan at task start |
| `OPERON_TASK_NOTE` / `OPERON_TASK_NOTE_MAX_ITEMS` | off / 8 | A running task-state note with each step, no model call |
| `OPERON_CHECKPOINTS` | off | Mark interrupted runs at startup; resume on request |
| `OPERON_PROGRESS_REVISIT` | off | A repeated action counts as progress only if its after-screen is new (full only) |
| `OPERON_TRUST_CATEGORIES` / `OPERON_TRUST_CATEGORY_LIST` | off / all five | Named risk categories (payment, send, delete, account, install) need approval |
| `OPERON_SECRETS` | off | Secrets vault placeholders |
| `OPERON_AUTO_CONFIRM` | off | `operon run` answers the model's own confirmation (at most 10 per run) |
| `OPERON_CLICK_TEXT` | off | The trust gate reads the text under a click with OCR (needs the `ocr` extra) |
| `OPERON_CLAUDE_ZOOM` | off | Lets Claude ask for a zoomed crop of the current screenshot |
| `OPERON_SITE_ADAPTERS` | empty | Load built-in site adapters (`wikipedia`, `github`, `all`) |
| `OPERON_*_BACKEND=model` with `OPERON_MODEL_PROFILE` | off | Registry models (Claude, OpenAI, open-weight) |

Defaults that are **on**: `OPERON_TRUST_GATE`, `OPERON_TOOLS=on` (no `run_shell`),
`OPERON_ZOOM=dense`, `OPERON_BROWSER_FALLBACK_BACKEND=json`.

## Invariants

- Policy and perception must remain vision-first.
- Deterministic rules run before LLM fallback.
- TYPE remains atomic at the executor level.
- Desktop clicks (`DesktopExecutor`, `LinuxDesktopExecutor`) use visual servo checks before input injection; the browser executor has none, and OSWorld runs its own fixed-threshold check (`environments/osworld.py`).
- One executor serves one run's step at a time (`AgentLoop._executor_lock`). With `OPERON_PARALLEL_RUNS` > 1 each batch browser run gets its own loop and executor (`api/runtime/loops.py`), so the lock is per executor and runs overlap; observable-mode and desktop runs share one executor and take turns (one CDP port, one real screen).
- Benchmark-specific behavior must not be hardcoded into core policy rules.

## Testing Layers

| Layer | Where | Cost and gate |
|---|---|---|
| Offline unit and integration suite | `tests/` (`pytest tests -q`, fake `GEMINI_API_KEY`) | Free; CI on Ubuntu with `--cov-fail-under=80` and on Windows (`.github/workflows/ci.yml`), plus ruff and mypy jobs |
| Static invariants | `tests/test_invariants.py`, `tests/test_env_example_contract.py` | Free; part of the suite |
| Scripted evals | `python -m evals` (`evals/harness.py`, `evals/scenarios.py`): the real loop with a scripted model and world | Free |
| Replay recordings | `evals/replay.py`, `evals/recordings/*.json`, `tests/evals/test_replay.py` | Free; thin-harness runs only; checks the action sequence |
| Real Chromium | `tests/browser/test_real_browser*.py` | Free; opt in with `OPERON_RUN_BROWSER_TESTS=true` |
| Browser end to end | `tests/browser_e2e/`, marker `browser_e2e` | Free; needs Playwright's Chromium; CI job "Browser end to end (Chromium)" |
| Linux desktop container e2e | `tests/desktop_e2e/`, marker `desktop_container` | Free; needs Docker; CI job "Desktop (Linux container)" |
| Windows Sandbox e2e | `tests/desktop_e2e_windows/`, marker `windows_sandbox` | Free; needs Windows Sandbox; not in CI |
| Live server tests | marker `live_server` (for example `tests/integration/`) | Need uvicorn on :8080, and some drive the real mouse; deselected by `addopts` |
| Paid benchmarks | `evals/live_bench.py`, `live_desktop_bench.py`, `live_hybrid_bench.py`, `osworld.py`; workflows `benchmark.yml`, `live-bench.yml`, `live-desktop-bench.yml`, `osworld-bench.yml` | Real model credits; run only on request |

The backends take an injected client, so tests script the model: `evals/harness.py` builds a
real `AgentLoop` with `FakeComputerUseClient` (same `generate_turn` and `run_step` shape as the
Gemini client), `FakeGeminiClient`, `FakeCriticClient` and a `FakeExecutor` over a scripted
world. `addopts` in `pyproject.toml` deselects `live_server`, `desktop_container`,
`windows_sandbox` and `browser_e2e`.

## Known Limitations

- `PROGRESSING_STABLE` is handled by recovery and the efficiency score, but no verifier in
  `src/` returns it.
- Parallel runs apply only to batch browser runs; observable and desktop runs share one
  executor and take turns.
- The `model` backend (Claude, OpenAI, open-weight) always runs the thin harness; there is no
  full-harness configuration for non-Gemini models.
- Open-weight profiles need an external OpenAI-compatible server; Operon ships no local model
  runtime.
- The JSON fallback is browser-only, triggers only on `BackendCompatibilityError` from
  `perceive`, and is ignored under thin.
- The browser executor has no visual servo; OSWorld uses a fixed threshold, not calibration.
- Desktop capture is `mss` on every platform; there is no GPU capture path.
- Windows Sandbox: one sandbox at a time; if it closes before the guest is ready (a busy or
  low-memory PC) the start is retried once (`WindowsSandboxProvider.start`).
- The focus guard is Windows only (`LinuxDesktopExecutor._focus_guard_platform = False`).
- Secrets are not encrypted at rest outside Windows.
- Replay covers only thin-harness runs and compares the action sequence, not screens.
- `agent/loop.py` is still one file of about 3,000 lines.

## Not Present / Intentionally Removed

To prevent re-introduction, the current tree intentionally does **not** include:

- `/benchmark/*`, `/command-center`, `/console`, `/dashboard`, or static-HTML/bundled-frontend routes.
- A `src/benchmarks` plugin package or policy-rule plugin registry.
- A `PostRunReflector` that auto-generates reflection records.
- A separate `src/runtime` orchestrator package or a standalone video-verifier module.

Current implementation truth is this document, `README.md`, and `AGENTS.md`.
