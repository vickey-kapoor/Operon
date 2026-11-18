# Operon architecture (as implemented)

Snapshot of branch `dev/local-build` on 2026-11-17. Describes the code that exists, not plans;
every statement points at a file in `src/` (paths below are relative to `src/operon/` unless they
start with `tests/`, `evals/` or `docs/`). Planned work lives in `docs/local-build-plan.md` and
`docs/roadmap.md`.

## 1. Overview

Operon is a vision-first computer-use agent. One control loop, `AgentLoop`
(`agent/loop.py`, about 2,700 lines), drives both a Chromium browser and a desktop (Windows or
X11 Linux), deciding every action from a screenshot. The perception and policy path never reads
the DOM, CSS selectors, XPath or the accessibility tree; targets are pixel coordinates.
`tests/test_invariants.py` enforces this statically by scanning `agent/perception`,
`agent/policy` and `agent/backends` for identifiers such as `locator`, `query_selector`,
`get_by_role` and `accessibility`. Structured machine state is read only by graders after a
run (`evals/desktop_state.py`, `evals/hybrid_state.py`, task-file `expect` checks in
`taskfile.py`), never by the agent. The reason is portability: the same loop works on any app
that draws pixels, and bad targeting is fixed by improving perception rather than by reaching
into app internals.

## 2. The step pipeline

The API (or CLI/SDK through the same front door) calls `AgentLoop.step_run()` once per step,
so runs are resumable and pausable between steps. `step_run` takes a per-run lock and the
loop's `_executor_lock` (one executor serves one step at a time), sets the executor's current
run, then calls `_step_run_locked`, which chains stage methods over a `StepContext`
(`agent/step_context.py`; fields set by a stage are `field(init=False)`, so reading one early
raises):

| Order | Stage (`agent/loop.py`) | What it does |
|---|---|---|
| 0 | `_begin_step` | Refuses runs not PENDING/RUNNING; enforces `max_steps` (default 25, `models/common.py`); builds `StepContext` |
| 1 | `_warmup_stage` | Step 1 only: browser without start URL navigates to google.com if the trust gate allows; 2.5 s settle (skipped in test safe mode) |
| 2 | `_capture_stage` | `ScreenCaptureService.capture` (`agent/perception/capture.py`) asks the executor for a frame and moves it to `step_N/before.png` |
| 3 | `_consume_override_hint`, `_add_task_note`, `_add_secrets_hint` | Queue advisory text hints for the model (user correction, `OPERON_TASK_NOTE`, secret names) |
| 4 | `_perceive_stage` | Memory hints, stagnation hint, then `perception_service.perceive` (with liveness retry); stamps monitor origin; saves observation |
| 5 | `_decide_stage` | `policy_service.choose_action`; `_attach_target_context` (grounder, monitor origin); Best-of-N (`OPERON_BESTOFN_N`) |
| 6 | `_gate_stage` | Cancel check; `_gate_pause_decision`: model asked for a human, coordinate thrash (3 failures at one point), confidence threshold, trust gate, file guard. A blocked action becomes WAIT_FOR_USER and the run pauses |
| 7 | `_execute_and_verify_stage` | Loop of `_execute_attempt` then `_verify_attempt`; on a hard FAILURE `_prepare_adaptation_retry` re-captures, re-perceives, re-decides and re-gates (max 3, `_MAX_ADAPTATION_RETRIES`; never in thin) |
| 8 | `_recover_stage` | `recovery_manager.recover`, then `validate_benchmark_integrity`; progress tracking and `progress_trace.json` |
| 9 | `_record_step_stage`, `_finish_step_stage` | Writes the `StepLog` to `run.jsonl`; sets SUCCEEDED / FAILED / RUNNING / CANCELLED / WAITING_FOR_USER; applies CONTEXT_RESET / SESSION_RESET side effects; writes `step_timing.json`; on terminal status writes `efficiency.json` |

### One action end to end: a Gemini computer-use click, batch browser mode

1. **Capture.** `NativeBrowserExecutor.capture` (`executor/browser_native.py`) calls Playwright
   `page.screenshot` on the run's own batch Chromium. The frame's width/height are the viewport
   (`BROWSER_WIDTH`/`BROWSER_HEIGHT`, default 1920x1080). The loop moves it to
   `.var/runs/<id>/step_N/before.png`.
2. **Perceive and decide in one call.** `BrowserComputerUseBackend.perceive`
   (`agent/backends/browser_computer_use.py`) renders the prompt, then
   `GeminiComputerUseHttpClient` (`clients/gemini_computer_use.py`) sends the PNG as-is (no
   downscale on the Gemini path) with the `computerUse` tool plus any `OPERON_TOOLS` function
   declarations, at temperature 0. A model error is re-raised as `BackendCompatibilityError`.
   With `OPERON_ZOOM=dense` (code default) a click is refined on a crop in one extra call, but
   only when the screen's long side is at least `OPERON_ZOOM_MIN_SIDE` (2000), so not on a
   1920x1080 viewport (`agent/actions/zoom.py`). Prompt, raw reply, parsed decision, answers and
   usage go to `step_N/computer_use_*.{txt,json}`.
3. **Scale coordinates back.** `normalize_computer_use_actions` (`agent/actions/translation.py`)
   maps Gemini's 0-999 grid to pixels with `_scale_grid_coord` (`value/999 * (size-1)`) against
   the frame size. Several calls in one turn become one `BATCH`; a `click_at` followed by an
   untargeted `type` is merged into one atomic TYPE at the click point. A
   `require_confirmation` safety decision holds all calls and pauses first.
4. **Policy.** Full harness: `PolicyCoordinator.choose_action` (`agent/policy/coordinator.py`)
   runs `PolicyRuleEngine` first, then the cached model decision, then `_reject_premature_stop`.
   The loop's `_attach_target_context` applies the grounder (identity for the default
   `deterministic` grounder) and the monitor origin (0,0 for a browser).
5. **Gate.** `_pre_execution_gate_text` evaluates each leaf of a batch with
   `TrustGate.evaluate` (`agent/policy/trust_gate.py`), passing the current URL and the model's
   rationale, then `apply_permission_mode`. A DENY anywhere outranks a CONFIRM. Then the file
   guard (`agent/policy/file_guard.py`).
6. **Execute.** `_execute_with_hardening` runs the action through the unified adapter into
   `NativeBrowserExecutor.execute`, which does `page.mouse.click(x, y)`. The browser executor has
   no visual servo. On a retryable execution failure (full harness only) it re-perceives in
   perception-only mode, re-resolves the target and retries once.
7. **Verify.** `DeterministicVerifierService.verify` (`agent/policy/verifier.py`) returns a
   status; `PENDING` re-verifies after 2 s, 4 s, 8 s (`_wait_for_page_load`); `STABLE_WAIT`
   waits 200 ms, re-captures, adopts the new frame as `after.png` and re-verifies once with
   `allow_stable_wait=False` (`_wait_for_ui_stable`).
8. **Recover.** `RuleBasedRecoveryManager` (`agent/policy/recovery.py`) picks the next move.

On the desktop the same flow uses `DesktopExecutor.capture` (`executor/desktop.py`): an `mss`
burst of the foreground monitor (three grabs 100 ms apart, `visual_velocity` from pixel change,
one re-capture if velocity > 0.02), native resolution. The monitor's origin is stamped on the
perception (`_with_capture_origin`) and added to the action (`_apply_monitor_origin`) so clicks
land on secondary monitors. Every click variant goes through `_click_preamble`: a monitor-bounds
check and the visual servo `_region_has_content` (pixel variance in a 50 px radius against a
threshold calibrated at init by `_calibrate_servo_threshold`); a blank region fails the click
with `EXECUTION_TARGET_NOT_FOUND`. Input is `pyautogui`.

### Verification states

| Status | Produced by | Loop handling |
|---|---|---|
| SUCCESS | Verifier: terminal-state evidence, accepted STOP, goal-completing action, critic, or fallback when confidence >= 0.5 | Advance; a SUCCESS stop with a success stop reason ends the run SUCCEEDED |
| FAILURE | Executor failure, rejected STOP (suspicious early stop, critic rejection, model gave up) | In-step adaptation retry (full), then recovery |
| UNCERTAIN | Passive wait without progress, low confidence with no critic, PENDING that never settles | Recovery |
| PENDING | Page looks like it is loading | 2/4/8 s backoff re-verify |
| STABLE_WAIT | Action visibly changed the screen | 200 ms settle, one re-verify |
| PROGRESSING_STABLE | Defined in `models/verification.py`; no verifier in `src/` currently returns it | Recovery advances on it |

Only the policy's own STOP ends a run as done: a critic stop signal on any other action is kept
as a note (`verify`, end of method).

### Recovery ladder (full harness)

`RuleBasedRecoveryManager.recover` bypasses the ladder for `stop_condition_met` (STOP),
`EXECUTION_NO_PROGRESS` (terminal STOP), SUCCESS with outcome met and PROGRESSING_STABLE
(ADVANCE). Otherwise, per retry cluster (subgoal, target, failure signal):

| Attempt | Strategy | Default wait |
|---|---|---|
| 1 | RETRY_SAME_STEP | none |
| 2 | RETRY_DIFFERENT_TACTIC (subgoal rewritten) | none |
| 3 | CONTEXT_RESET (`executor.context_reset()`) | 1000 ms |
| 4 | SESSION_RESET (`executor.session_reset(start_url)`) | 1500 ms |
| 5 (`MAX_RECOVERY_ATTEMPTS`) | terminal STOP, `RETRY_LIMIT_REACHED` | none |

`validate_benchmark_integrity` stops recovery from claiming success that verification did not
report.

## 3. Harness profiles

`harness_profile()` (`agent/policy/thin.py`) decides per backend; the layers are wired in
`api/runtime/loops.py:_loop_layers`.

| | Full (default for Gemini `computer_use`; always for `json`) | Thin (`OPERON_HARNESS_PROFILE=thin`; always for `model`) |
|---|---|---|
| Policy | `PolicyCoordinator`: memory hints, `PolicyRuleEngine` (two rules: `_task_success_stop_rule`, `_prefer_launch_app_rule` on desktop), model, `_reject_premature_stop` | `ThinPolicy`: the backend's decision unchanged |
| Verifier | `DeterministicVerifierService` with a Gemini critic (`OPERON_*_VERIFIER_MODEL`, or the acting model with `OPERON_CRITIC=same`) | `OwnDoneVerifier`: failures go back to the model as tool results; the model's "done" ends the run |
| Recovery | `RuleBasedRecoveryManager` ladder plus progress stop guard | `ThinRecovery`: always hand back to the model |
| Loop extras | Redundant-action blocking, execution retry, adaptation retry, grounder refinement | All skipped (`self.thin_harness` checks in `loop.py`) |
| Browser fallback | `FallbackBackend` to JSON when configured | Ignored |

Both profiles keep the trust gate, file guard, coordinate-thrash pause and step budget. Advisory
text (task note, secret names, change note) is allowed in thin because it changes what the model
sees, not its action.

## 4. Model backends

Built in `api/runtime/services.py` (`build_browser_services`, `build_desktop_services`,
`build_model_backend`), selected by `OPERON_BROWSER_BACKEND` / `OPERON_DESKTOP_BACKEND`:

| Backend value | Class | Notes |
|---|---|---|
| `computer_use` (default) | `BrowserComputerUseBackend`, `DesktopComputerUseBackend` (subclass, adds `launch_app`) | Gemini computer use, default model `gemini-3.5-flash`, thinking `low`; full-resolution screenshot, 0-999 coordinates |
| `json` | `BrowserJsonBackend` / `CombinedPerceptionPolicyService` (`agent/backends/combined.py`) | Older Gemini JSON-output path; always full harness |
| `model` | Registry profile from `OPERON_MODEL_PROFILE` (`agent/backends/model_adapter/models.yaml`, `registry.py`) | Always thin |

Registry adapters actually built by `build_model_backend`:

- `claude_native`: `BrowserClaudeComputerUseBackend` / `DesktopClaudeComputerUseBackend` over
  `AnthropicComputerUseHttpClient` (`clients/anthropic_computer_use.py`). Screenshots are
  downscaled to a 1280 px long edge and coordinates scaled back
  (`browser_claude_computer_use.py:encode_screenshot`). Profiles: `claude-opus-5-5`,
  `claude-sonnet-5-5`, `claude-haiku-5-5`.
- `openai_native`: `OpenAIComputerUseBackend` (`model_adapter/openai_native.py`) over the
  Responses API; refused without pricing so the cost cap works. Profile: `openai-computer-use`.
- `openai_compat`: `ModelAdapterBackend` (`model_adapter/backend.py`) over `OpenAICompatClient`
  for any OpenAI-compatible server (base URL from `OPENWEIGHT_BASE_URL` by default), with parsers
  `neutral_tools`, `ui_tars`, `opencua` (`model_adapter/formats/`), per-profile image resize
  rules (`images.py`, incl. Qwen `smart_resize`) and coordinate spaces (`pixels`,
  `relative_1000`, `normalized`). Profiles: `qwen3-vl-32b`, `ui-tars-1.5-7b`, `opencua-32b`.
- `gemini_native` is in the schema's adapter list but the registry rejects a profile that uses
  it with a clear error (Gemini runs through the `computer_use` backends).

**Fallback.** Browser only, full harness only: with `OPERON_BROWSER_FALLBACK_BACKEND=json`
(default) `FallbackBackend` (`agent/backends/fallback.py`) switches the run to
`BrowserJsonBackend` for the rest of the run when the primary's `perceive` raises
`BackendCompatibilityError`. Desktop computer use has no fallback.

**Scripted models for tests.** The backends take an injected client. `evals/harness.py` builds a
real `AgentLoop` with `FakeComputerUseClient` (same `generate_turn`/`run_step` shape as the Gemini
client), `FakeGeminiClient`, `FakeCriticClient` and a `FakeExecutor` over a scripted world.
`tests/desktop_e2e/test_linux_desktop.py` and `tests/desktop_e2e_windows/test_windows_sandbox.py`
use a `ScriptedClient` with the real `DesktopComputerUseBackend`, thin layers, and a real guest
desktop. `evals/replay.py` replays recorded Gemini answers through the real loop.

## 5. Executors and environments

| Executor | File | Used for |
|---|---|---|
| `NativeBrowserExecutor` | `executor/browser_native.py` | All browser runs. **batch** (default): a fresh isolated Chromium per run, launched without a debug port. **observable**: one persistent Chromium launched with `--remote-debugging-port=9222` and kept across tasks; `browser/manager.py` attaches over CDP (`connect_over_cdp`) and streams JPEG frames (`Page.startScreencast`) to `/ws/stream` |
| `DesktopExecutor` | `executor/desktop.py` | Windows desktop: `mss` capture, `pyautogui` input, visual servo, focus guard, DPI handling (`executor/dpi.py`) |
| `LinuxDesktopExecutor` | `executor/linux_desktop.py` | X11 Linux (XFCE): subclass of `DesktopExecutor` plus `LiveViewMixin`; focus guard off; short ASCII typed by key, other text pasted |
| `RemoteDesktopExecutor` | `environments/remote.py` | Host side of a guest machine: HTTP to `operon guest`, screenshots returned as bytes and written locally |
| `OSWorldExecutor` | `environments/osworld.py` | OSWorld task VM server; its own fixed servo threshold (20.0) |
| `ToolExecutor` | `executor/tools.py` | Wraps an executor when `OPERON_TOOLS` enables tools (Gemini `computer_use` only); runs `read_file`, `list_dir`, `write_file`, `http_get` (and `run_shell` with `all`) |

Desktop environment providers (`environments/registry.py`, `OPERON_ENVIRONMENT`, default `live`;
`operon envs` lists them):

| Name | Provider | Executor | Runs on host |
|---|---|---|---|
| `live` | `LiveDesktopProvider` | `DesktopExecutor`, or `LinuxDesktopExecutor` when `OPERON_DESKTOP_PLATFORM=linux` | yes |
| `windows-sandbox` | `WindowsSandboxProvider` | `RemoteDesktopExecutor` to the guest's `GuestWindowsExecutor` | no |
| `linux-container` | `LinuxContainerProvider` (Docker, `docker/linux-desktop`) | `RemoteDesktopExecutor` to a guest `LinuxDesktopExecutor` | no |
| `osworld` | `OSWorldProvider` | `OSWorldExecutor` | no |

The guest (`environments/guest.py`, `operon guest`) is a FastAPI app inside the machine that
serves capture, execute, live frames and takeover input behind a one-time bearer token. The host
keeps the model key, the loop and the run files. For guests, tools run inside the guest, not on
the host (`get_desktop_agent_loop`).

## 6. Runtime and state

- **Process state.** `OperonRuntime` (`core/runtime.py`, `current_runtime()`): `browser_loop`,
  `desktop_loop`, `run_loops` (parallel runs), `run_slots` semaphore, `desktop_provider`,
  `builtin_browser`, `sdk_run_active`, operator `ControlSettings` (confidence threshold, disabled
  rules), live-stream state, and `runs: OrderedDict[str, RunContext]`.
- **Per-run state.** `RunContext`: `started_at`, `finished_at`, `override_hint`,
  `auto_run_task`.
- **Loops.** Built lazily in `api/runtime/loops.py` (`get_agent_loop`,
  `get_desktop_agent_loop`). With `OPERON_PARALLEL_RUNS` > 1, each **batch** browser run gets a
  `ParallelRunLoop` with its own executor and a slot (`start_parallel_browser_run`); a full
  semaphore raises `RunSlotsFull`, answered as HTTP 429 with `Retry-After: 5`. Observable and
  desktop runs share one loop and take turns.
- **Settings.** `get_settings()` (`core/settings.py`) returns one typed `Settings` parsed from the
  environment, re-parsed only when a watched variable changes. The API loads `.env` with
  `override=True` (`api/server.py`).
- **Run store.** `FileBackedRunStore` (`store/run_store.py`), root from `core/paths.py`
  (`.var/runs/` in a checkout, `~/.operon/runs/` installed; `OPERON_RUNS_ROOT`). Writes go
  through a background writer (`store/background_writer.py`).

```text
.var/runs/<run_id>/
  state.json             AgentState snapshot (histories trimmed; full history is in run.jsonl)
  run.jsonl              one StepLog per step, plus pre-step failure rows
  settings.json          effective settings at start, secrets left out
  host.json              os, home, cwd (replay contract)
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

- **Resumability.** Steps are separate calls, so a paused run (WAITING_FOR_USER) resumes with
  `resume_run`. With `OPERON_CHECKPOINTS=on`, API startup calls `mark_interrupted_runs`
  (`store/checkpoints.py`), which flags in-flight runs whose owner process is dead (pid plus
  creation time, `store/owner.py`) as WAITING_FOR_USER with `stop_reason=interrupted`. Nothing
  resumes on its own; `resume_interrupted` continues one from the current screen
  (`POST /run/{id}/resume-interrupted`, `operon resume <id>`).

## 7. Entry points

- **HTTP API** (`api/server.py`, routers in `api/routes/`; `X-API-Key` auth from `API_KEYS` or a
  generated key in `.var/api-key`; host allowlist and CORS from settings):
  - Browser (`routes/browser.py`): `POST /run-task`, `/step`, `/resume`, `/stop`, `/cleanup`,
    `/run/{id}/stop`, `/run/{id}/pause`, `/run/{id}/resume-interrupted`; `GET /run/{id}`, `/health`.
  - Desktop (`routes/desktop.py`): the same under `/desktop/...`.
  - Observer (`routes/observer.py`): `GET /observer/api/runs`, `/run/{id}`, `/artifact`,
    `/usage`, `/settings`, `/export/{id}`, `/live-browser/{id}`.
  - Front door (`routes/frontdoor.py`, prefix `/app/api`): `/runs` (start, list), `/runs/{id}`,
    `/report`, `/approve`, `/decline`, `/pause`, `/takeover`, `/handback`, `/stop`, `/input`,
    `/interrupted`, `/resume-interrupted`, `/stream-token`.
  - Built-in browser (`routes/builtin_browser.py`, prefix `/app/api/browser`, `OPERON_BUILTIN_BROWSER`).
  - `WS /ws/stream` (`api/ws_stream.py`); `GET /` redirects to `GET /app`, the task page
    (`api/static/app.html`; the key arrives in the `#key=` fragment printed by `operon serve`).
- **Front door service** (`api/frontdoor.py`): one implementation used by the task page, its
  routes, `operon run`, the SDK and MCP, so all behave the same.
- **CLI** (`cli.py`, entry point `operon`): `run` (`--desktop`, `--env`, `--url`,
  `--mode ask|risky|auto`, `--max-steps`, `--headless`, `--request`, `--tasks file.yaml`,
  `--junit`, `--only`, `--fail-fast`), `runs --interrupted`, `resume <run_id>`, `serve`, `mcp`,
  `setup` (`setup_command.py`), `secrets set|list|rm` (`secrets_command.py`), `cloud`
  (`cloud.py`, Azure Linux VM), `envs` (`environments/cli.py`), `guest`.
- **Python SDK** (`sdk.py`, `from operon import Agent`, lazy export in `__init__.py`):
  `Agent(desktop=, env=, mode="ask", max_steps=, on_approval=).run(task, url=)` returns a
  `RunResult` (status, final answer, steps, cost, run id). One SDK run per process
  (`sdk_run_active`); approvals are denied unless `on_approval` returns true.
- **Task files** (`taskfile.py`): YAML suites run one task per `Agent`, default mode `risky`,
  graded by `expect` checks after the run; JUnit output.
- **MCP server** (`mcp_server.py`, `operon mcp`, optional `[mcp]` extra, stdio): tools
  `start_task`, `get_task`, `wait_for_task`, `approve`, `deny`, `stop_task`,
  `list_environments`. Always mode `ask`, never approves for the client, one run at a time.

## 8. Safety

- **Trust gate** (`agent/policy/trust_gate.py`, `OPERON_TRUST_GATE`, on by default). Matches
  case-insensitive phrases against a description of the action (text, URL, target name). Order in
  `evaluate`: deny phrases -> DENY; navigation outside `OPERON_TRUST_ALLOW_DOMAINS` -> DENY; any
  action while the current page is outside the allowlist -> DENY; typing a stored secret on a
  page not on the allowlist -> CONFIRM that no mode can skip; confirm phrases (built-in default
  list when `OPERON_TRUST_CONFIRM` is empty) -> CONFIRM; named categories (payment, send, delete,
  account, install; `OPERON_TRUST_CATEGORIES`, default off) -> CONFIRM; `run_shell` -> CONFIRM;
  deny/confirm phrase in the model's stated reason when the action names no target -> CONFIRM.
- **Permission modes** (`apply_permission_mode`; per run, default `OPERON_PERMISSION_MODE` or
  `risky`): `ask` turns every non-passive ALLOW into CONFIRM; `risky` leaves the verdict;
  `auto` turns CONFIRM into ALLOW except the secret confirmation. DENY is never relaxed.
- **Approvals.** A gated action pauses the run; resuming approves only that exact action
  (fingerprint of type plus gate description, `_gate_fingerprint`). The pending gate's verdict is
  exposed (`pending_gate_verdict`) so the SDK's `ApprovalRequest.verdict` and task files can tell
  `deny` from `confirm`; task files' `approve_all` never approves a `deny`. Gemini's own
  `require_confirmation` holds the calls and pauses; `OPERON_AUTO_CONFIRM` (off) lets
  `operon run` answer it for benchmarks, never trust-gate pauses (`cli.py`).
- **File guard** (`agent/policy/file_guard.py`, always on): file-path actions must name a path
  given in the task, inside the run directory, or under `OPERON_ALLOWED_FILE_ROOTS`; this gate
  cannot be approved by resuming.
- **Secrets vault** (`core/vault.py`, `OPERON_SECRETS`, off): the model types
  `{{secret:NAME}}`; the executor resolves it at type time (`browser_native.py`, `remote.py`,
  `osworld.py`, desktop `_enter_secret`), so records keep the placeholder. Values are DPAPI
  encrypted on Windows; elsewhere 0600 files, not encrypted. Secrets go to a guest only when its
  host is a private IP literal (`windows_sandbox.py`, `private_ip_literal`).
- **Focus guard** (`OPERON_FOCUS_GUARD`, off, Windows only): `DesktopExecutor._record_focus` at
  capture and `_check_focus` before the first action fail it with `FOCUS_STOLEN` if another
  process's window took the foreground.

## 9. Observability

- Per-step artifacts listed in section 6; model I/O per step (`computer_use_prompt.txt`,
  `_raw.json`, `_parsed.json`, `_answers.json`, `_usage.json`); usage aggregated by
  `GET /observer/api/usage`; a run exported as a zip by `/observer/api/export/{id}` (`api/observer.py`).
- `OPERON_TRACE=1` prints `[TRACE]` stage lines from `loop.py:_trace`.
- `recovered_errors`: every broad `except` in the loop calls `AgentLoop._recovered`, which logs
  with traceback and counts per site; counts are merged into `state.json` after each step.
  `OPERON_STRICT_ERRORS` (on in tests) re-raises bug exceptions instead.
- Efficiency (`agent/efficiency.py`, KAN-85, measurement only): `total_steps`, `model_calls`,
  `actions_per_model_call`, `wall_seconds`, `seconds_per_step`, `wasted_steps` and `wasted_ratio`
  with `blocked`, `failed`, `repeat`, `noop_wait`, `revisit` counts; saved to `efficiency.json`
  and the run state.
- Live view: observable browser frames over `/ws/stream` (short-lived tokens from
  `/app/api/stream-token`); Linux desktop and guest executors stream through `LiveViewMixin`
  (`executor/live_view.py`).

## 10. Default-off experimental flags

From `core/settings.py` and `.env.example`.

| Flag | Default | Effect |
|---|---|---|
| `OPERON_HARNESS_PROFILE` | empty (full) | `thin` runs Gemini computer use in the thin harness |
| `OPERON_CRITIC` | empty | `same` makes the acting model the step critic |
| `OPERON_BESTOFN_N` / `_CONFIDENCE` | 1 / 1.0 | Sample N candidates at uncertain steps, run the best |
| `OPERON_GROUNDER` | `deterministic` | `snap` moves near-miss clicks onto the nearest element (full only) |
| `OPERON_PARALLEL_RUNS` | 1 | Batch browser runs that may step at once, each with its own Chromium |
| `OPERON_SETTLE_DETECT` / `_MAX_S` | off / 3.0 | Wait for the screen to stop changing instead of a fixed sleep |
| `OPERON_CHANGE_NOTE` | off | One text line about what changed while the screen settled |
| `OPERON_FOCUS_GUARD` | off | Skip the first action after a capture if focus was stolen (Windows) |
| `OPERON_DPI_AWARE` | off | Per-monitor-v2 DPI awareness before `pyautogui` loads (Windows) |
| `OPERON_PLANNER` / `_MODEL` | off / `gemini-3.5-flash-lite` | One text-only plan at task start |
| `OPERON_TASK_NOTE` / `_MAX_ITEMS` | off / 8 | Running task-state note with each step, no model call |
| `OPERON_CHECKPOINTS` | off | Mark interrupted runs at startup; resume on request |
| `OPERON_PROGRESS_REVISIT` | off | Repeated action counts as progress only if its after-screen is new (full only) |
| `OPERON_TRUST_CATEGORIES` / `_CATEGORY_LIST` | off / all five | Named risk categories need approval |
| `OPERON_SECRETS` | off | Secrets vault placeholders |
| `OPERON_AUTO_CONFIRM` | off | `operon run` answers the model's own confirmation (max 10 per run) |
| `OPERON_BUILTIN_BROWSER` | off | One saved-profile browser shown as the app's Browser tab |
| `OPERON_CLAUDE_ZOOM` | off | Lets Claude ask for a zoomed crop of the current screenshot |
| `OPERON_SITE_ADAPTERS` | empty | Load built-in site adapters (`wikipedia`, `github`, `all`) |
| `OPERON_*_BACKEND=model` + `OPERON_MODEL_PROFILE` | off | Registry models (Claude, OpenAI, open-weight) |

Defaults that are **on**: `OPERON_TRUST_GATE`, `OPERON_TOOLS=on` (no `run_shell`),
`OPERON_ZOOM=dense`, `OPERON_BROWSER_FALLBACK_BACKEND=json`.

## 11. Testing layers

| Layer | Where | Cost / gate |
|---|---|---|
| Offline unit and integration suite | `tests/` (`pytest tests -q`, fake `GEMINI_API_KEY`) | Free; CI on Ubuntu with `--cov-fail-under=80` and on Windows (`.github/workflows/ci.yml`), plus ruff and mypy jobs |
| Static invariants | `tests/test_invariants.py`, `tests/test_env_example_contract.py` | Free; part of the suite |
| Scripted evals | `python -m evals` (`evals/harness.py`, `evals/scenarios.py`): real loop, scripted model and world | Free |
| Replay recordings | `evals/replay.py`, `evals/recordings/*.json`, `tests/evals/test_replay.py` | Free; thin-harness runs only; checks the action sequence |
| Real Chromium | `tests/browser/test_real_browser*.py` | Free; opt in with `OPERON_RUN_BROWSER_TESTS=true` (there is no `tests/browser_e2e`) |
| Linux desktop container e2e | `tests/desktop_e2e/` marker `desktop_container` | Free; needs Docker; CI job "Desktop (Linux container)" |
| Windows Sandbox e2e | `tests/desktop_e2e_windows/` marker `windows_sandbox` | Free; needs Windows Sandbox; not in CI |
| Live server tests | marker `live_server` (e.g. `tests/integration/`) | Need uvicorn on :8080, some drive the real mouse; deselected by `addopts` |
| Paid benchmarks | `evals/live_bench.py`, `live_desktop_bench.py`, `live_hybrid_bench.py`, `osworld.py`; workflows `benchmark.yml`, `live-bench.yml`, `live-desktop-bench.yml`, `osworld-bench.yml` | Real model credits; run only on request |

`addopts` in `pyproject.toml` deselects `live_server`, `desktop_container` and `windows_sandbox`.

## 12. Known limitations (verifiable in the code)

- `PROGRESSING_STABLE` is handled by recovery and the efficiency score but no verifier in `src/`
  returns it.
- Parallel runs apply only to batch browser runs (`runs_in_parallel`); observable and desktop
  runs share one executor and take turns.
- The `model` backend (Claude, OpenAI, open-weight) always runs the thin harness
  (`harness_profile`); there is no full-harness configuration for non-Gemini models.
- Open-weight profiles need an external OpenAI-compatible server; Operon ships no local model
  runtime, and `models.yaml` notes that open-weight model ids are confirmed only when first
  served. `gemini_native` is rejected at registry validation.
- The JSON fallback is browser-only, triggers only on `BackendCompatibilityError` from
  `perceive`, and is ignored under thin.
- The browser executor has no visual servo; OSWorld uses a fixed threshold, not calibration.
- Desktop capture is `mss` (GDI) on every platform; there is no GPU capture path.
- Windows Sandbox: one sandbox at a time; if it closes before the guest is ready (busy or
  low-memory PC) the start is retried once (`WindowsSandboxProvider.start`).
- Focus guard is Windows only (`LinuxDesktopExecutor._focus_guard_platform = False`).
- Secrets are not encrypted at rest outside Windows.
- Replay covers only thin-harness runs and compares the action sequence, not screens.
- The SDK allows one run per process; the MCP server one run at a time.
- `.env.example` describes `OPERON_ZOOM` as "off (default)" but the code default is `dense`;
  `BROWSER_HEADLESS` defaults to false in `Settings` while `.env.example` sets `true`.
- `agent/loop.py` is still one file of about 2,700 lines.
