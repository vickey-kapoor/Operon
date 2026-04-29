# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Operon

**Zero-Abstraction, Vision-First Enterprise Operator.**

Operon is a clinical, execution-only computer-use engine. It perceives the world as raw pixels and operates exclusively in coordinate space. There is no DOM traversal, no CSS selectors, no XPath, no accessibility tree — the agent sees only what a camera would see and acts only where it can see. Every run is a direct march toward a visually confirmed terminal state.

The core loop: **capture → perceive → update state → choose action → execute → verify → recover**.

General-purpose: accepts any `intent` via `RunTaskRequest`. Desktop automation uses pyautogui + mss for full-screen control. `PageHint` accepts arbitrary snake_case strings from the LLM.

## Environment Setup

Python **3.11** required. Use a local venv:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Copy `.env.example` to `.env` and set your Gemini key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`).

Runtime backend and model selection (env vars):

- `OPERON_DESKTOP_BACKEND=json` (default desktop path)
- `OPERON_BROWSER_BACKEND=computer_use` (Gemini Computer Use)
- `OPERON_BROWSER_FALLBACK_BACKEND=json` (JSON fallback for browser)
- `BROWSER_HEADLESS=true` (headless Playwright)

Planner provider override (per-mode, defaults to `gemini`):

- `OPERON_DESKTOP_PLANNER_PROVIDER=anthropic` — use Claude as the desktop planner
- `OPERON_BROWSER_PLANNER_PROVIDER=anthropic` — use Claude as the browser planner
- `ANTHROPIC_API_KEY` — required when either planner provider is `anthropic`

Model overrides (all optional, see `src/api/runtime_config.py` for defaults):

- `OPERON_DESKTOP_MODEL`, `OPERON_DESKTOP_PLANNER_MODEL`, `OPERON_DESKTOP_VERIFIER_MODEL`
- `OPERON_BROWSER_MODEL`, `OPERON_BROWSER_PLANNER_MODEL`, `OPERON_BROWSER_VERIFIER_MODEL`, `OPERON_BROWSER_FALLBACK_MODEL`

If PowerShell fails to launch external processes (COM+ errors), run:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```

## Common Commands

**Run tests (safe default — excludes live-server tests):**

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
.venv\Scripts\python -m pytest tests\ -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py
```

**Single test file:**

```powershell
.venv\Scripts\python -m pytest tests\test_agent_loop.py -q
```

**Lint:**

```powershell
ruff check src tests --select E,F,W,I --ignore E501
```

**Run the form benchmark:**

```powershell
$env:FORM_BENCHMARK_URL = "https://practice-automation.com/form-fields/"
python -m src.agent.benchmark
```

**Start the API server:**

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

**Replay / summarize a stored run:**

```powershell
python -m src.store.replay <run_id>
python -m src.store.summary <run_id>
python -m src.store.summary runs
```

## Architecture

### Core Loop (`src/agent/loop.py`)

`AgentLoop` orchestrates every run. A single step does:

1. **Capture** — `ScreenCaptureService` fires `DesktopExecutor._capture_burst_sync()`, which grabs three frames at t₀, t+100ms, t+200ms. Pixel diff between the last two frames produces a `visual_velocity` float on the returned `CaptureFrame`. If velocity > 2%, the capture service waits 300ms and re-bursts before returning — ensuring perception never sees a layout shift or animation mid-frame.
2. **Perceive** — `GeminiPerceptionService` sends the stable screenshot to Gemini and returns a typed `ScreenPerception` (visible elements, page hint, focused element). High velocity is logged as a `CoordDriftWarning` so runs can be audited for unstable captures.
3. **Choose action** — `PolicyCoordinator` runs `PolicyRuleEngine` (deterministic rules in named priority order), then falls back to `GeminiPolicyService` (LLM prompt). When a rule fires, its name is stamped on `PolicyDecision.rule_name`. On the next step, if the LLM is called, the previous rule's outcome is injected as a **Rule-Augmented Generation** trace in the advisory hints.
4. **Block redundancy** — `AgentLoop._block_redundant_action` prevents repeated actions without progress using `ProgressState` counters.
5. **Execute** — `DesktopExecutor` performs the action via pyautogui/mss. Before any click, `_region_has_content()` captures a 100×100px crop around (x, y) and checks pixel variance. A uniform region (variance < 20) means the target has shifted — logged as `CoordDriftWarning`, returned as `EXECUTION_TARGET_NOT_FOUND`. `AgentLoop._execute_with_hardening()` owns one bounded retry on stale/shifted targets.
6. **Verify** — `DeterministicVerifierService.verify()` runs `check_terminal_state()` first: a pure visual predicate that checks for goal-completion evidence (benchmark success tokens, intent-derived terminal signals) in the current perception. The loop ends as soon as the terminal state is visually confirmed — regardless of which action triggered it. Only after this check does the verifier evaluate action-level success/failure.
7. **Video verify** (conditional) — If `screen_diff` detects no visual change, `VideoVerifier` records a 3-second clip, re-executes the action, and sends the clip to Gemini for temporal analysis. Triggers only for idempotent actions (click, press_key, hotkey, launch_app, scroll, hover); adds ~5-8s to uncertain steps.
8. **Recover** — `RuleBasedRecoveryManager` decides whether to continue, retry, or stop via a staged ladder: soft-retry → subgoal reset → hard-stop.
9. **Update rule trace** — After verification, if `decision.rule_name` is set, `state.last_rule_trace` is formatted with rule name + action + outcome. This trace is injected into the next LLM prompt via `add_advisory_hints(source="rule_trace")`. If the LLM decided this step, `last_rule_trace` is cleared.
10. **Record step** — Memory store writes per-step records. On verification failure, `_decay_active_hints()` appends a decay record with `weight = current_effective_weight × 0.5` for each hint active in the current benchmark+page_hint+subgoal context.
11. **Reflect** (on terminal) — `PostRunReflector` analyzes the completed run, detects failure patterns, and writes `MemoryRecord` entries. On success, `_filter_one_shot_steps()` prunes any step that used a retry before passing the trajectory to `_extract_episode()` — only the shortest-path steps are stored.
12. **Log** — `StepLog` is appended to `runs/<run_id>/run.jsonl`; every artifact goes under `runs/<run_id>/step_N/`.

Terminal conditions: `FORM_SUBMITTED_SUCCESS`, `STOP_BEFORE_SEND`, `TASK_COMPLETED` (success); retry limit, max step limit, repeated loop detection (failure). `WAITING_FOR_USER` is a non-terminal pause — the run resumes when `POST /resume` is called.

### Strict Baseline Requirements

These are enforced at `DesktopExecutor.__init__()` via `validate_display_baseline()` (skipped when `OPERON_TEST_SAFE_MODE=true`):

| Requirement | Value | Enforcement |
|---|---|---|
| Primary display resolution | ≥ 1280×720 (1920×1080 recommended) | `HardwareBaselineError` raised below minimum |
| DPI scaling | 100% (96 DPI) recommended | `CoordDriftWarning` logged if != 100%; not a hard stop because per-monitor DPI awareness mitigates drift |
| Loading/shimmer state | Wait exactly 500ms | Prompt directive + verifier `_is_page_loading()` returns `PENDING` |

`HardwareBaselineError` is defined in `src/executor/desktop.py`. It is a `RuntimeError` subclass raised when the display geometry cannot support reliable coordinate targeting.

**Temporal Physics rule:** If the perception shows a loading skeleton, shimmer animation, spinner, or fewer than 3 elements on a content page — issue `wait 500ms` and re-perceive. Never act on a transient UI state. This rule is enforced in both prompt files and in `DeterministicVerifierService._is_page_loading()`.

### Execution Philosophy

**Direct path always wins.** The agent is an industrial tool, not an explorer.

- **Search / Direct URL before menus.** If a search bar or direct URL navigation is available, it is the primary path. Menu traversal is a last resort.
- **No exploration.** The agent does not browse, read, or learn about a site. If the target is not visible, it searches or navigates — never wanders.
- **Atomic TYPE.** For any visible input, search box, or text area, issue TYPE directly. The executor acquires focus atomically (click → 50ms → text dispatch in a single call). A policy-level CLICK-before-TYPE on the same element is always wrong and will cause double-click or duplicate-input behavior.
- **Terminal-state focus.** Every action is evaluated against the question: "Does the current screen confirm the goal?" The verifier checks this first, before checking action success.

### Dual Execution Paths

Operon has two execution modes sharing the same loop, verifier, recovery, and persistence:

- **Desktop mode** — full-screen automation via `pyautogui` + `mss`. Uses combined JSON perception+policy through `GeminiHttpClient`.
- **Browser mode** — Playwright-based. Primary backend is `BrowserComputerUseBackend` (Gemini Computer Use with coordinate normalization and multi-call turns); fallback is `BrowserJsonBackend`. `NativeBrowserExecutor` translates actions to Playwright calls. Browser sessions are video-recorded under `.browser-artifacts/` and linked into the run snapshot for the observer UI.

Backend selection is handled by `src/agent/backend.py` based on env vars. `src/agent/action_translation.py` bridges Computer Use action formats to the internal `AgentAction` schema.

### Runtime Package (`runtime/`)

A top-level `runtime/` package (not under `src/`) provides the unified contract layer:

- `UnifiedOrchestrator` — receives `LegacyContractBundle` from `LegacyOperonContractAdapter` each step, runs adaptation strategy lookup, detects OS file pickers via perception, and advances `AgentRuntimeState`
- `AgentRuntimeState` — structured per-run mutable state tracking subgoal progress, last perception summary, retry context, and advisory hints; separate from `AgentState` in `src/models/`
- `LegacyOperonContractAdapter` — translates the existing `AgentState`/`ScreenPerception`/`PolicyDecision`/`ExecutedAction`/`VerificationResult` types into the unified `LegacyContractBundle` format consumed by `UnifiedOrchestrator`

`AgentLoop` creates a `UnifiedOrchestrator` singleton and maintains a `unified_states` dict of `AgentRuntimeState` per run. After each step's verify phase, the loop translates to contracts and calls `UnifiedOrchestrator.process_step()`.

### Policy Layer (`src/agent/policy_coordinator.py`, `policy_rules.py`, `policy.py`)

`PolicyCoordinator` wraps `GeminiPolicyService` with a rule layer:

- `PolicyRuleEngine` runs engine primitives in named priority order via an explicit loop (not a Python `or`-chain). Each rule that fires sets `self._last_fired_rule` — accessible via `last_fired_rule_name()`.
- `PolicyDecision` carries `rule_name: str | None`. When a rule fires, the coordinator stamps the rule name on the decision via `model_copy`.
- `AgentState` carries `last_rule_trace: str | None`. After verification, the loop formats `"[RULE TRACE] Rule 'X' fired at step N | action=Y | outcome=Z"` and stores it. The coordinator injects it into the next LLM call via `add_advisory_hints(source="rule_trace")`. This is **Rule-Augmented Generation** — the LLM planner always knows what the deterministic layer last attempted and whether it worked.
- `PolicyCoordinator.reset_run_context(run_id)` clears `_active_episode`, `_replay_state`, and the hint cache at the start of each new run, preventing task context from bleeding across runs. Browser auth state (cookies/session tokens) is intentionally preserved at the executor layer for authenticated benchmark flows.
- Memory hints from `FileBackedMemoryStore` are injected into both the rule engine and the LLM prompt. Hints with effective weight below 0.1 are pruned before injection.

**Engine primitives (always active, in priority order):**

| Rule | Trigger |
|---|---|
| `_human_intervention_rule` | Page hint contains any `HITL_PAGE_HINT_KEYWORDS` keyword (debounced: 2 consecutive matches required) |
| `_task_success_stop_rule` | `page_hint == FORM_SUCCESS` or success tokens visible |
| `_dropdown_menu_select_rule` | Dropdown open + unselected option visible |
| `_avoid_identical_type_retry` | Memory hint `avoid_identical_type_retry` + repeated TYPE failure on same element |
| `_no_progress_recovery_rule` | No-progress streak exceeds threshold |
| `_dismiss_blocking_overlay_rule` | Blocking dialog/banner detected |
| `_search_query_rule` | Intent contains search query + search input visible |

Benchmark-specific plugin rules are registered via `BENCHMARK_REGISTRY` and run before engine primitives.

### Atomic Execution and Visual Servo (System Invariants)

These two behaviors are **non-negotiable system invariants**. Do not add code that bypasses either.

**Atomic Focus+Type:**
The executor merges focus acquisition and text dispatch into a single atomic call. When `_exec_type()` is invoked with coordinates, it performs: move cursor → click → 50ms wait → clipboard-paste text, all within one executor call. This eliminates the race condition where a separate CLICK step changes screen state (autocomplete, focus stealing) before the TYPE arrives.

Consequences:
- **Never emit a policy-level CLICK immediately followed by TYPE on the same element.** The executor handles this internally. A policy pre-click wastes a step and may double-click.
- `_avoid_identical_type_retry` fires when this pattern is violated — it re-establishes focus rather than retrying the raw type.
- `_focus_before_type_rule` is memory-gated (only fires when `click_before_type` hint is active). Do not add explicit focus steps outside that rule.
- Both `policy_prompt.txt` and `browser_combined_prompt.txt` contain the **ATOMIC TYPE DIRECTIVE** instructing the LLM to TYPE directly.

**Visual Servo / Crosshair Verification:**
Before any `pyautogui.click`, `_region_has_content(x, y, radius=50)` captures a 100×100px crop around the target and computes pixel variance. Variance < 20 (uniform solid region) means the element has shifted or disappeared — the click is aborted with `EXECUTION_TARGET_NOT_FOUND` and logged as `CoordDriftWarning`. The try/except around the mss call ensures a pixel-grab failure never blocks a click.

**Visual Velocity:**
`_capture_burst_sync()` samples three frames with 100ms sleeps between them (inside `asyncio.to_thread`) and computes the fraction of pixels that changed between the last two. This float is stored in `CaptureFrame.visual_velocity`. The capture service re-bursts after 300ms when velocity > 2%. The loop stage that follows never perceives an animating frame.

### Intent-Based Re-resolution (`src/agent/loop.py`, `src/agent/selector.py`)

- Targetable `AgentAction`s carry serializable `target_context` with normalized `TargetIntent`, original target signature, top candidate evidence, and original matched signals.
- Retry-time re-resolution is deterministic only and reuses `DeterministicTargetSelector.reresolve()`.
- Re-resolution keeps current semantic and spatial evidence primary; prior element id, prior name, prior screen region, and signal continuity are only weak tie-break signals.
- Failures are explicit: `target_reresolution_failed` or `target_reresolution_ambiguous`.
- Traceability is recorded inside `execution_trace.json` via `reresolution_trace`.

### State & Models (`src/models/`)

All Pydantic v2. Key types:

| File | Purpose |
|---|---|
| `state.py` — `AgentState` | Full mutable run state. Includes `last_rule_trace: str | None` for Rule-Augmented Generation. |
| `capture.py` — `CaptureFrame` | Screenshot output. Includes `visual_velocity: float` from the burst capture. |
| `perception.py` — `ScreenPerception` | Typed output of one perception call (elements, page hint). |
| `policy.py` — `PolicyDecision`, `AgentAction` | What the policy chose and why. Includes `rule_name: str | None` stamped when a deterministic rule fired. |
| `execution.py` — `ExecutedAction` | Outcome of the executor, including `ExecutionTrace` with per-attempt detail. |
| `progress.py` — `ProgressTrace` | Snapshot of loop progress state at one step. |
| `common.py` | Shared enums: `RunStatus`, `StopReason`, `FailureCategory`, `LoopStage`. |
| `memory.py` — `MemoryRecord`, `MemoryHint` | Advisory hint schema. `MemoryRecord` carries `weight: float` (default 1.0); halved on each verification failure; pruned below 0.1. |

### Memory Protocol — Decaying Episodic Memory

Operon's memory system is self-pruning. It optimizes for the shortest path, not historical completeness.

**Per-step memory records** (`FileBackedMemoryStore.record_step()`):
- Written as JSONL entries tagged with benchmark, page_hint, subgoal, action type, and outcome.
- On verification failure, `_decay_active_hints()` appends a decay record for each hint active in the current context: `weight = current_effective_weight × 0.5`. This is geometric decay — a hint that fails repeatedly converges toward zero and is pruned from `get_hints()` results.
- `get_hints()` computes the mean weight per (key, hint) bucket across all matching records. Buckets with mean weight < 0.1 are excluded.

**Episode extraction** (`PostRunReflector._extract_episode()`):
- Only called on successful runs.
- Before extraction, `_filter_one_shot_steps()` removes any step where `failure_category is not None` or where `retry_counts` increased that step. Only first-attempt successes remain.
- The extracted episode encodes the **optimal trajectory** — the exact sequence of actions that achieved the goal without hesitation. Multi-attempt steps are discarded because they represent recovered failures, not learned patterns worth repeating.
- Episodes are keyed by (normalized_intent, benchmark) and increment `success_count` on each re-success.

**Session isolation** (`PolicyCoordinator.reset_run_context()`):
- Called by `AgentLoop.start_run()` at the beginning of every new run.
- Clears `_active_episode`, `_replay_state`, and the hint cache so no episodic context from a prior task bleeds into the new one.
- Browser-level auth state (cookies, session tokens) is **intentionally preserved** — it is managed at the Playwright executor layer and must survive across runs of authenticated benchmark flows (e.g., WebArena/GitLab).

### Persistence (`src/store/`)

- `FileBackedRunStore` — in-memory dict + `runs/<run_id>/state.json` on disk; no database.
- `FileBackedMemoryStore` — append-only JSONL; weight-aware `get_hints()`; decays on failure; 1-step-filtered episode extraction.
- `run_logger.py` — appends `StepLog` / `PreStepFailureLog` as JSONL.
- `replay.py`, `summary.py` — read-only analysis tools.

### API (`src/api/`)

FastAPI app at `src/api/server.py`. Routes in `src/api/routes.py`:

- `POST /run-task` — create a run record
- `POST /step` — advance a run one step
- `POST /resume` — resume a `WAITING_FOR_USER` run
- `GET /run/{id}` — read run state
- `GET /health`
- `GET /` or `GET /desktop-pilot` — Operon Pilot UI (unified desktop + browser)
- `GET /observer/api/runs`, `GET /observer/api/run/{id}`, `GET /observer/api/artifact` — run data endpoints

The `AgentLoop` singleton is built lazily on first request via `get_agent_loop()`.

### Human-in-the-Loop (`src/agent/hitl.py`)

When the agent encounters a page it cannot handle autonomously (CAPTCHA, login wall, cookie consent, 2FA, age gate, payment, T&C, bot-detection block), `_human_intervention_rule` in `PolicyRuleEngine` fires (after 2 consecutive matching steps — debounced to prevent false positives) and issues `ActionType.WAIT_FOR_USER`. The loop then calls `_pause_for_user()` which:

1. Calls `generate_hitl_message()` — Gemini produces a 2-sentence explanation grounded in the current intent and visible elements.
2. Sets `AgentState.hitl_message` and transitions to `RunStatus.WAITING_FOR_USER` (non-terminal).
3. Calls `notify_desktop()` — Windows balloon tip via PowerShell, macOS `osascript`, Linux `notify-send`.
4. Starts `start_escalation_timer()` as an asyncio task — re-notifies at 2 min → 10 min → 30 min if the run is still paused.
5. The Pilot UI shows a full-screen overlay with a live screenshot refreshing every 3s and a "Resume Agent" button that calls `POST /resume`.

`HITL_PAGE_HINT_KEYWORDS` covers: `captcha`, `recaptcha`, `robot`, `login`, `sign_in`, `cookie_consent`, `gdpr`, `age_verification`, `two_factor`, `2fa`, `mfa`, `otp`, `terms_and_conditions`, `payment`, `checkout`, `blocked`, `access_denied`, `bot_detection`.

### OS File Picker Macro (`src/executor/os_picker_macro.py`)

`run_os_picker_macro()` is a deterministic, LLM-free primitive invoked by `NativeBrowserExecutor` after clicking an upload control that opens a native OS file dialog. It polls for a picker window via `pygetwindow` keyword matching, types the absolute file path with `pyautogui.write`, presses Enter, then polls for the window to close. Returns `PickerMacroResult` with `PickerOutcome` enum (`SUCCESS`, `PICKER_NOT_DETECTED`, `FILE_NOT_REFLECTED`, `UNAVAILABLE`).

### Clients (`src/clients/gemini.py`, `src/clients/anthropic.py`)

`GeminiHttpClient` wraps raw HTTP calls for both perception and policy Gemini requests. Prompt templates live in `prompts/perception_prompt.txt` and `prompts/policy_prompt.txt`.

`AnthropicHttpClient` wraps the Anthropic Messages API for text-only planner and vision-based verifier calls. Requires `ANTHROPIC_API_KEY`. Supports configurable model, retry backoff, and HTTP/2 via `httpx`.

## Run Data Layout

```
runs/
  <run_id>/
    state.json          # AgentState snapshot
    run.jsonl           # StepLog entries (one JSON object per line)
    reflection.json     # PostRunReflector output (patterns + episode extracted)
    step_N/
      before.png
      after.png
      perception_prompt.txt
      perception_raw.txt
      perception_parsed.json
      policy_prompt.txt
      policy_raw.txt
      policy_decision.json
      execution_trace.json
      progress_trace.json
      verification_result.json
      selector_trace.json

.browser-artifacts/     # Browser session video recordings
memory/
  memory.jsonl          # Append-only MemoryRecord log (with weight field)
  episodes.jsonl        # Compressed episode trajectories (1-step-filtered)
```

## Coding Style

- **Functional over stateful.** Prefer pure functions and immutable data transformations. Stateful objects (`AgentState`, `ScreenPerception`) are Pydantic models mutated only at explicit loop boundaries — not inside helper functions.
- **Async/await for all I/O.** Every network call, file read, and subprocess interaction must use `async`/`await`. Blocking I/O inside an `async` function must be wrapped with `asyncio.to_thread`. Never call `time.sleep` in async code; use `asyncio.sleep`.
- **Pydantic v2 for all state boundaries.** Every object that crosses a service boundary must be a `StrictModel` subclass. Validate at ingestion (`model_validate`), not at use. Do not use raw dicts for inter-service data.
- **No DOM, no selectors, no XPaths.** Operon is vision-only. Element targeting uses `UIElement` coordinates from perception output. Any code that references HTML structure, CSS selectors, accessibility trees, or Playwright `locator()` in the policy/perception path is a contract violation. This is a zero-abstraction architecture.
- **Rules before LLM.** Deterministic logic belongs in `PolicyRuleEngine`. If a condition can be expressed as a Python predicate over `ScreenPerception` + `AgentState`, it must be a rule, not a prompt instruction.
- **One responsibility per service.** `PerceptionService` returns typed screen state. `PolicyService` returns an action decision. `VerifierService` returns a verification result. Do not let concerns leak across these boundaries.
- **Invariants are inviolable.** Atomic Focus+Type and Visual Servo are system invariants. No new code may emit a CLICK before a TYPE on the same element, and no click may bypass the `_region_has_content()` check.

## Debug Skill

When a task fails or produces unexpected behavior, always start with the run's JSONL log before reading source code:

```powershell
# Tail the last N steps of a run
.venv\Scripts\python -m src.store.summary <run_id>

# Read raw step logs (one JSON object per line)
cat runs/<run_id>/run.jsonl
```

Each line in `run.jsonl` is a `StepLog` with: `step`, `stage`, `action_type`, `status`, `failure_category`, `stop_reason`, and `rationale`. Look for:

- `failure_category` — maps directly to `FailureCategory` enum values; pinpoints whether the failure is in perception, policy, execution, or verification
- `stop_reason` — explains why the run terminated; unexpected values like `MAX_RETRIES_EXCEEDED` or `REPEATED_LOOP_DETECTED` indicate a recovery failure
- `rationale` — the policy's stated reason for its chosen action; a hallucinated rationale usually points to a prompt or perception quality issue
- `CoordDriftWarning` in logs — indicates a visual servo failure or DPI mismatch; check `execution_trace.json` for the affected coordinates

Per-step artifacts under `runs/<run_id>/step_N/` contain the raw model input/output:

| File | What to check |
|---|---|
| `perception_parsed.json` | Element coordinates, page_hint, confidence — are elements where the policy thinks they are? |
| `perception_diagnostics.json` | Quality gate outcome, salvage attempts — was perception rejected or salvaged? |
| `policy_decision.json` | Full `PolicyDecision` including `active_subgoal`, `confidence`, and `rule_name` |
| `execution_trace.json` | Per-attempt detail, re-resolution trace — did the target shift? Was a CoordDriftWarning triggered? |
| `verification_result.json` | Critic verdict, terminal state check result, `recovery_hint` |
| `selector_trace.json` | DeterministicTargetSelector candidates and scoring — why was a particular element chosen? |

## Commit Style

Use short imperative subjects with prefixes: `Fix:`, `Docs:`, `CI:`, `Chore:`, `Refactor:`. CI enforces Ruff rules `E,F,W,I`; `E501` (line length) is ignored.
