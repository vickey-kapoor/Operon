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

Runtime backend selection (env vars):

- `OPERON_DESKTOP_BACKEND=json` (default desktop path)
- `OPERON_BROWSER_BACKEND=computer_use` (Gemini Computer Use)
- `OPERON_BROWSER_FALLBACK_BACKEND=json` (JSON fallback for browser)
- `BROWSER_HEADLESS=true` (headless Playwright)

If PowerShell fails to launch external processes (COM+ errors), run:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```

## Common Commands

**Run tests:**

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests\ -q
```

**Single test file:**

```powershell
python -m pytest tests\test_agent_loop.py -q
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
8. **Recover** — `RuleBasedRecoveryManager` decides whether to continue, retry, or stop the run
9. **Reflect** (on terminal) — `PostRunReflector` analyzes the completed run, extracts failure patterns, and writes `MemoryRecord` entries for future runs
10. **Log** — `StepLog` is appended to `runs/<run_id>/run.jsonl`; every artifact (screenshots, prompt/raw/parsed files, traces, recordings) goes under `runs/<run_id>/step_N/`

Terminal conditions: `FORM_SUBMITTED_SUCCESS`, `STOP_BEFORE_SEND` (success); retry limit, max step limit, repeated loop detection (failure).

### Dual Execution Paths

Operon has two execution modes sharing the same loop, verifier, recovery, and persistence:

- **Desktop mode** — full-screen automation via `pyautogui` + `mss`. Uses combined JSON perception+policy through `GeminiHttpClient`.
- **Browser mode** — Playwright-based. Primary backend is `BrowserComputerUseBackend` (Gemini Computer Use with coordinate normalization and multi-call turns); fallback is `BrowserJsonBackend`. `NativeBrowserExecutor` translates actions to Playwright calls. Browser sessions are video-recorded under `.browser-artifacts/` and linked into the run snapshot for the observer UI.

Backend selection is handled by `src/agent/backend.py` based on env vars. `src/agent/action_translation.py` bridges Computer Use action formats to the internal `AgentAction` schema.

### Policy Layer (`src/agent/policy_coordinator.py`, `policy_rules.py`, `policy.py`)

`PolicyCoordinator` wraps `GeminiPolicyService` with a rule layer:

- `PolicyRuleEngine` runs 6 deterministic rules in priority order (selector-based matching via `src/agent/selector.py` and `geometry.py`)
- Memory hints from `FileBackedMemoryStore` are injected into both the rule engine and the LLM prompt
- If no rule fires, the LLM prompt in `prompts/policy_prompt.txt` is rendered and sent to Gemini

### Intent-Based Re-resolution (`src/agent/loop.py`, `src/agent/selector.py`)

- Targetable `AgentAction`s now carry serializable `target_context` with normalized `TargetIntent`, original target signature, top candidate evidence, and original matched signals
- Retry-time re-resolution is deterministic only and reuses `DeterministicTargetSelector.reresolve()` rather than ad hoc DOM/id rematching
- Re-resolution keeps current semantic and spatial evidence primary; prior element id, prior name, prior screen region, and signal continuity are only weak tie-break signals
- Failures are explicit: `target_reresolution_failed` or `target_reresolution_ambiguous`
- Traceability is recorded inside `execution_trace.json` via `reresolution_trace` with trigger reason, original intent, candidates considered, whether the old id was reused, and the final outcome

**Rule classification — not yet separated in code, but conceptually distinct:**

| Rule | Type | Trigger |
|---|---|---|
| `_login_page_guardrail` | Benchmark-specific (Gmail) | `page_hint == GOOGLE_SIGN_IN` + memory hint `authenticated_start_required` |
| `_form_success_stop_rule` | Engine primitive (generic stop) | `page_hint == FORM_SUCCESS` or "success"/"thank you"/"submitted" in elements |
| `_avoid_identical_type_retry` | Engine primitive | Memory hint `avoid_identical_type_retry` + repeated TYPE failure on same element |
| `_compose_already_visible_rule` | Benchmark-specific (Gmail) | Subgoal contains "compose" + `GMAIL_COMPOSE` page hint or compose input visible |
| `_submit_form_when_ready_rule` | Benchmark-specific (form) | `page_hint == FORM_PAGE` + name/email/message all successfully typed |
| `_focus_before_type_rule` | Engine primitive | Memory hint `click_before_type` + target not focused |

Rules 3 and 6 are memory-gated engine primitives seeded for both benchmarks. Rules 1, 4, and 5 embed domain knowledge that will not generalise to a third benchmark. The domain bleed also runs through: `PageHint` enum (four of six values are Gmail-specific), `_seed_default_guardrails()` seeding Gmail guardrails on every run, `benchmark_name_for_intent()` defaulting unknown intents to `FORM_BENCHMARK`, and the label tokens in `_required_form_fields_completed()`. A third benchmark is the natural forcing function to split this into a base engine rule set and per-benchmark rule registrations.

### State & Models (`src/models/`)

All Pydantic v2. Key types:

| File | Purpose |
|---|---|
| `state.py` — `AgentState` | Full mutable run state (history, progress counters, subgoal, status) |
| `perception.py` — `ScreenPerception` | Typed output of one perception call (elements, page hint) |
| `policy.py` — `PolicyDecision`, `AgentAction` | What the policy chose and why |
| `execution.py` — `ExecutedAction` | Outcome of the executor, including `ExecutionTrace` with per-attempt detail |
| `progress.py` — `ProgressTrace` | Snapshot of loop progress state at one step |
| `common.py` | Shared enums: `RunStatus`, `StopReason`, `FailureCategory`, `LoopStage` |
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
- `POST /resume` — resume a paused/stopped run
- `GET /run/{id}` — read run state
- `GET /health`
- `GET /` or `GET /desktop-pilot` — Operon Pilot UI (unified desktop + browser)
- `GET /observer/api/runs`, `GET /observer/api/run/{id}`, `GET /observer/api/artifact` — run data endpoints

The `AgentLoop` singleton is built lazily on first request via `get_agent_loop()`.

### Clients (`src/clients/gemini.py`)

`GeminiHttpClient` wraps raw HTTP calls for both perception and policy Gemini requests. Prompt templates live in `prompts/perception_prompt.txt` and `prompts/policy_prompt.txt`.

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

## Commit Style

Use short imperative subjects with prefixes: `Fix:`, `Docs:`, `CI:`, `Chore:`, `Refactor:`. CI enforces Ruff rules `E,F,W,I`; `E501` (line length) is ignored.
