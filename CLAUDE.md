# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Operon

Vision-driven computer-use engine. It operates by running a closed loop: **capture → perceive → update state → choose action → execute → verify → recover**.

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

1. **Capture** — `ScreenCaptureService` screenshots the current screen
2. **Perceive** — `GeminiPerceptionService` sends the screenshot to Gemini and returns a typed `ScreenPerception` (visible elements, page hint, focused element)
3. **Choose action** — `PolicyCoordinator` first tries `PolicyRuleEngine` (deterministic rules + memory hints), then falls back to `GeminiPolicyService` (LLM prompt)
4. **Block redundancy** — `AgentLoop._block_redundant_action` prevents repeated actions without progress using `ProgressState` counters
5. **Execute** — `DesktopExecutor` performs the action via pyautogui/mss. `AgentLoop._execute_with_hardening()` owns one bounded retry: on `stale_target_before_action`, `target_shifted_before_action`, or `target_lost_before_action`, it captures fresh perception and re-runs the deterministic selector against the original `TargetIntent` plus lightweight target context instead of relying only on the old `target_element_id`
6. **Verify** — `DeterministicVerifierService` checks whether the outcome matches what was expected
7. **Video verify** (conditional) — If `screen_diff` detects no visual change, `VideoVerifier` records a 3-second video via `ScreenRecorder`, re-executes the action, and sends the clip to Gemini for temporal analysis. This only triggers for idempotent actions (click, press_key, hotkey, launch_app, scroll, hover) and adds ~5-8s to uncertain steps.
8. **Recover** — `RuleBasedRecoveryManager` decides whether to continue, retry, or stop the run via a staged recovery ladder: soft-retry → subgoal reset → hard-stop, escalating only when the same failure repeats on the same target. The critic's `recovery_hint` (from the critic prompt at `prompts/critic_prompt.txt`) is injected into the ladder to influence escalation decisions.
9. **Reflect** (on terminal) — `PostRunReflector` analyzes the completed run, extracts failure patterns, and writes `MemoryRecord` entries for future runs
10. **Log** — `StepLog` is appended to `runs/<run_id>/run.jsonl`; every artifact (screenshots, prompt/raw/parsed files, traces, recordings) goes under `runs/<run_id>/step_N/`

Terminal conditions: `FORM_SUBMITTED_SUCCESS`, `STOP_BEFORE_SEND` (success); retry limit, max step limit, repeated loop detection (failure). `WAITING_FOR_USER` is a non-terminal pause — the run resumes when `POST /resume` is called.

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

`AgentLoop` creates a `UnifiedOrchestrator` singleton and maintains a `unified_states` dict of `AgentRuntimeState` per run. After each step's verify phase (step 6), the loop translates to contracts and calls `UnifiedOrchestrator.process_step()`.

### Policy Layer (`src/agent/policy_coordinator.py`, `policy_rules.py`, `policy.py`)

`PolicyCoordinator` wraps `GeminiPolicyService` with a rule layer:

- `PolicyRuleEngine` runs 6 deterministic rules in priority order (selector-based matching via `src/agent/selector.py` and `geometry.py`)
- Memory hints from `FileBackedMemoryStore` are injected into both the rule engine and the LLM prompt
- If no rule fires, the LLM prompt in `prompts/policy_prompt.txt` is rendered and sent to Gemini (or Claude via `AnthropicPolicyService` when `planner_provider=anthropic`)
- `AnthropicPolicyService` (`src/agent/anthropic_policy.py`) subclasses `GeminiPolicyService`, reusing the same prompt renderer and strict output parser but routing generation through `AnthropicHttpClient` (`src/clients/anthropic.py`)

### Intent-Based Re-resolution (`src/agent/loop.py`, `src/agent/selector.py`)

- Targetable `AgentAction`s now carry serializable `target_context` with normalized `TargetIntent`, original target signature, top candidate evidence, and original matched signals
- Retry-time re-resolution is deterministic only and reuses `DeterministicTargetSelector.reresolve()` rather than ad hoc DOM/id rematching
- Re-resolution keeps current semantic and spatial evidence primary; prior element id, prior name, prior screen region, and signal continuity are only weak tie-break signals
- Failures are explicit: `target_reresolution_failed` or `target_reresolution_ambiguous`
- Traceability is recorded inside `execution_trace.json` via `reresolution_trace` with trigger reason, original intent, candidates considered, whether the old id was reused, and the final outcome

**Rule classification — not yet separated in code, but conceptually distinct:**

| Rule | Type | Trigger |
|---|---|---|
| `_human_intervention_rule` | Engine primitive (HITL) | page hint contains any `HITL_PAGE_HINT_KEYWORDS` keyword |
| `_login_page_guardrail` | Benchmark-specific (Gmail) | `page_hint == GOOGLE_SIGN_IN` + memory hint `authenticated_start_required` |
| `_form_success_stop_rule` | Engine primitive (generic stop) | `page_hint == FORM_SUCCESS` or "success"/"thank you"/"submitted" in elements |
| `_avoid_identical_type_retry` | Engine primitive | Memory hint `avoid_identical_type_retry` + repeated TYPE failure on same element |
| `_compose_already_visible_rule` | Benchmark-specific (Gmail) | Subgoal contains "compose" + `GMAIL_COMPOSE` page hint or compose input visible |
| `_submit_form_when_ready_rule` | Benchmark-specific (form) | `page_hint == FORM_PAGE` + name/email/message all successfully typed |
| `_focus_before_type_rule` | Engine primitive | Memory hint `click_before_type` + target not focused |

Rule 0 (`_human_intervention_rule`) is a pure engine primitive — it fires on any page hint containing a HITL keyword and requires no memory seed. Rules 3 and 6 are memory-gated engine primitives seeded for both benchmarks. Rules 1, 4, and 5 embed domain knowledge that will not generalise to a third benchmark. The domain bleed also runs through: `PageHint` enum (four of six values are Gmail-specific), `_seed_default_guardrails()` seeding Gmail guardrails on every run, `benchmark_name_for_intent()` defaulting unknown intents to `FORM_BENCHMARK`, and the label tokens in `_required_form_fields_completed()`. A third benchmark is the natural forcing function to split this into a base engine rule set and per-benchmark rule registrations.

### State & Models (`src/models/`)

All Pydantic v2. Key types:

| File | Purpose |
|---|---|
| `state.py` — `AgentState` | Full mutable run state (history, progress counters, subgoal, status, `hitl_message`) |
| `perception.py` — `ScreenPerception` | Typed output of one perception call (elements, page hint) |
| `policy.py` — `PolicyDecision`, `AgentAction` | What the policy chose and why |
| `execution.py` — `ExecutedAction` | Outcome of the executor, including `ExecutionTrace` with per-attempt detail |
| `progress.py` — `ProgressTrace` | Snapshot of loop progress state at one step |
| `common.py` | Shared enums: `RunStatus`, `StopReason`, `FailureCategory`, `LoopStage`; `RunResponse` includes `hitl_message: str \| None` |
| `memory.py` — `MemoryRecord`, `MemoryHint` | Advisory hint schema stored by `FileBackedMemoryStore` |

### Persistence (`src/store/`)

- `FileBackedRunStore` — in-memory dict + `runs/<run_id>/state.json` on disk; no database
- `FileBackedMemoryStore` — accumulates `MemoryRecord` entries per benchmark; surfaced as `MemoryHint` list on each step
- `run_logger.py` — appends `StepLog` / `PreStepFailureLog` as JSONL
- `replay.py`, `summary.py` — read-only analysis tools

### API (`src/api/`)

FastAPI app at `src/api/server.py`. Routes in `src/api/routes.py`:

- `POST /run-task` — create a run record
- `POST /step` — advance a run one step
- `POST /resume` — resume a `WAITING_FOR_USER` run (called by Pilot UI's "Resume Agent" button after human intervention)
- `GET /run/{id}` — read run state
- `GET /health`
- `GET /` or `GET /desktop-pilot` — Operon Pilot UI (unified desktop + browser)
- `GET /observer/api/runs`, `GET /observer/api/run/{id}`, `GET /observer/api/artifact` — run data endpoints

The `AgentLoop` singleton is built lazily on first request via `get_agent_loop()`.

### Human-in-the-Loop (`src/agent/hitl.py`)

When the agent encounters a page it cannot handle autonomously (CAPTCHA, login wall, cookie consent, 2FA, age gate, payment, T&C, bot-detection block), `_human_intervention_rule` in `PolicyRuleEngine` fires and issues `ActionType.WAIT_FOR_USER`. The loop then calls `_pause_for_user()` which:

1. Calls `generate_hitl_message()` — Gemini produces a 2-sentence explanation of what happened and what the human needs to do, grounded in the current intent and visible elements.
2. Sets `AgentState.hitl_message` and transitions to `RunStatus.WAITING_FOR_USER` (non-terminal).
3. Calls `notify_desktop()` — Windows balloon tip via PowerShell, macOS `osascript`, Linux `notify-send`.
4. Starts `start_escalation_timer()` as an asyncio task — re-notifies at 2 min → 10 min → 30 min if the run is still paused.
5. The Pilot UI shows a full-screen overlay with the LLM message, a live screenshot refreshing every 3 s, and a "Resume Agent" button that calls `POST /resume`.

`HITL_PAGE_HINT_KEYWORDS` — the frozenset that drives keyword matching — covers: `captcha`, `recaptcha`, `robot`, `login`, `sign_in`, `cookie_consent`, `gdpr`, `age_verification`, `two_factor`, `2fa`, `mfa`, `otp`, `terms_and_conditions`, `payment`, `checkout`, `blocked`, `access_denied`, `bot_detection`.

### OS File Picker Macro (`src/executor/os_picker_macro.py`)

`run_os_picker_macro()` is a deterministic, LLM-free primitive invoked by `NativeBrowserExecutor` after clicking an upload control that opens a native OS file dialog (headed mode only). It polls for a picker window via `pygetwindow` keyword matching, types the absolute file path with `pyautogui.write`, presses Enter, then polls for the window to close. Returns `PickerMacroResult` with a `PickerOutcome` enum (`SUCCESS`, `PICKER_NOT_DETECTED`, `FILE_NOT_REFLECTED`, `UNAVAILABLE`). Failures map to standard executor failure categories.

### Atomic Execution Pattern

The executor merges focus and type into a single atomic operation rather than issuing two sequential actions. When a TYPE action targets an element that is not currently focused, `DesktopExecutor` performs a click on the target's center coordinates immediately before writing the text — both within the same executor call. This eliminates the race condition where a separate CLICK step can change screen state before the TYPE arrives (e.g., an autocomplete dropdown opening and stealing focus between steps).

Consequences for policy and rules:
- **Never emit a CLICK immediately followed by TYPE on the same element.** The executor handles focus internally; a pre-click from the policy wastes a step and can cause double-click behavior on some inputs.
- `_avoid_identical_type_retry` (`policy_rules.py`) fires when this pattern is violated — it re-establishes focus rather than retrying the raw type, which is the correct recovery path.
- `_focus_before_type_rule` is memory-gated and only fires when the `click_before_type` hint is active (seeded from failure patterns). Do not add explicit focus steps outside that rule.

When adding new action types that write to an element, follow the same pattern: embed the focus acquisition inside the executor call, not as a separate policy step.

### Clients (`src/clients/gemini.py`, `src/clients/anthropic.py`)

`GeminiHttpClient` wraps raw HTTP calls for both perception and policy Gemini requests. Prompt templates live in `prompts/perception_prompt.txt` and `prompts/policy_prompt.txt`.

`AnthropicHttpClient` wraps the Anthropic Messages API for text-only planner and vision-based verifier calls. Requires `ANTHROPIC_API_KEY`. Supports configurable model, retry backoff, and HTTP/2 via `httpx`.

## Run Data Layout

```
runs/
  <run_id>/
    state.json          # AgentState snapshot
    run.jsonl           # StepLog entries (one JSON object per line)
    step_1/
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

.browser-artifacts/     # Browser session video recordings
```

## Coding Style

- **Functional over stateful.** Prefer pure functions and immutable data transformations. Stateful objects (`AgentState`, `ScreenPerception`) are Pydantic models mutated only at explicit loop boundaries — not inside helper functions.
- **Async/await for all I/O.** Every network call, file read, and subprocess interaction must use `async`/`await`. Blocking I/O inside an `async` function must be wrapped with `asyncio.to_thread`. Never call `time.sleep` in async code; use `asyncio.sleep`.
- **Pydantic v2 for all state boundaries.** Every object that crosses a service boundary (perception output, policy decision, verification result, step log) must be a `StrictModel` subclass. Validate at ingestion (`model_validate`), not at use. Do not use raw dicts for inter-service data.
- **No DOM, no selectors, no XPaths.** Operon is vision-only. Element targeting uses `UIElement` coordinates from perception output. Any code that references HTML structure, CSS selectors, accessibility trees, or Playwright `locator()` in the policy/perception path is a contract violation.
- **Rules before LLM.** Deterministic logic belongs in `PolicyRuleEngine`, not prompt engineering. If a condition can be expressed as a Python predicate over `ScreenPerception` + `AgentState`, it must be a rule, not an instruction appended to the planner prompt.
- **One responsibility per service.** `PerceptionService` returns typed screen state. `PolicyService` returns an action decision. `VerifierService` returns a verification result. Do not let concerns leak across these boundaries.

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

Per-step artifacts under `runs/<run_id>/step_N/` contain the raw model input/output:

| File | What to check |
|---|---|
| `perception_parsed.json` | Element coordinates, page_hint, confidence — are elements where the policy thinks they are? |
| `perception_diagnostics.json` | Quality gate outcome, salvage attempts — was perception rejected or salvaged? |
| `policy_decision.json` | Full `PolicyDecision` including `active_subgoal` and `confidence` |
| `execution_trace.json` | Per-attempt detail, re-resolution trace — did the target shift between steps? |
| `verification_result.json` | Critic verdict and `recovery_hint` — what did the verifier conclude? |

## Commit Style

Use short imperative subjects with prefixes: `Fix:`, `Docs:`, `CI:`, `Chore:`, `Refactor:`. CI enforces Ruff rules `E,F,W,I`; `E501` (line length) is ignored.
