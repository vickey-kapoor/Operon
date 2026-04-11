# Operon

![Operon loop mark](assets/operon-loop-mark.svg)

Vision-driven computer-use engine. Operon operates desktop apps and browser
sessions through a closed loop — **capture → perceive → decide → execute →
verify → recover → reflect** — using only screenshots and (when needed) short
video clips. No DOM scraping, no accessibility tree.

License: Apache-2.0

## Highlights

- **Vision-only** — the agent sees what a human sees. Works on any app that
  renders pixels.
- **Two execution paths, one loop** — desktop (pyautogui + mss) and browser
  (Playwright + Gemini Computer Use) share the same agent loop, verifier,
  recovery, persistence, and memory.
- **Deterministic rules first, LLM fallback** — `PolicyRuleEngine` runs
  selector-matched rules before `GeminiPolicyService` is invoked.
- **Hardened execution** — if a target moves between perceive and execute,
  `DeterministicTargetSelector.reresolve()` re-binds to the original
  `TargetIntent` instead of a stale element id.
- **Video verification** — when a screenshot diff shows no change, Operon
  records a 3-second clip and asks Gemini for temporal evidence.
- **Self-improvement** — `PostRunReflector` writes `MemoryRecord` hints and
  compresses successful runs into reusable `Episode` trajectories; both are
  surfaced on future runs.
- **Unified contract layer** — every step is also validated through typed
  `PerceptionOutput` / `PlannerOutput` / `ActorOutput` / `CriticOutput`
  contracts for cross-environment consistency.

## Architecture

```text
FastAPI routes  (src/api/)
    |
    +-- AgentLoop  (src/agent/loop.py)
          |
          +-- Capture       (mss screenshot)
          +-- Perceive      (Gemini → ScreenPerception)
          +-- Decide        PolicyCoordinator
          |                   ├── PolicyRuleEngine  (deterministic rules)
          |                   └── GeminiPolicyService (LLM fallback)
          +-- Re-resolve    DeterministicTargetSelector (on stale target)
          +-- Execute       DesktopExecutor | NativeBrowserExecutor
          +-- Verify        DeterministicVerifier + VideoVerifier
          +-- Recover       RuleBasedRecoveryManager
          +-- Reflect       PostRunReflector  (on terminal)
          +-- Persist       FileBackedRunStore + FileBackedMemoryStore
          |
          +-- Unified Contract Layer (observer + validator)
                LegacyOperonContractAdapter
                  → PerceptionOutput / PlannerOutput / ActorOutput / CriticOutput
                UnifiedOrchestrator  (core/router.py routing rules)
                AgentRuntimeState    (per-run unified state)
```

The unified layer runs **on top of** the legacy loop: `AgentLoop` still owns
execution ordering and termination, while the contract bundle is validated per
step and used to drive `orchestrator.adaptation_strategy_for()` when needed.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/agent/` | Main loop, backends, policy coordinator, selectors, verifier, recovery, reflector |
| `src/api/` | FastAPI routes, Pilot UI, Task Console, observer endpoints |
| `src/executor/` | `DesktopExecutor` (pyautogui) and `NativeBrowserExecutor` (Playwright) |
| `src/clients/` | Gemini HTTP clients (text + Computer Use) |
| `src/models/` | Pydantic v2 schemas: state, perception, policy, execution, episode, memory |
| `src/store/` | File-backed run store, memory store, episode cache, replay, summary |
| `core/` | Unified contract definitions and routing rules |
| `core/contracts/` | `perception.py`, `planner.py`, `actor.py`, `critic.py` |
| `runtime/` | `UnifiedOrchestrator`, `LegacyOperonContractAdapter`, `AgentRuntimeState`, `benchmark_runner.py` |
| `executors/` | Thin unified adapters wrapping the legacy executors |
| `prompts/` | Prompt templates (perception, policy, browser combined, Computer Use) |
| `docs/` | Contract + migration notes (`agent_contract.md`, `migration_cleanup.md`, `phase4_migration_note.md`) |
| `examples/contracts/` | Example JSON contract bundles for browser / desktop / cross-environment |
| `tests/` | Unit, route, phase-based integration, benchmark, and browser regressions |

See `CODEBASE_OVERVIEW.md` for a higher-level walkthrough.

## Browser Runtime

- `BrowserComputerUseBackend` — primary; Gemini Computer Use turns with
  normalized coordinates and multi-call support
- `BrowserJsonBackend` — JSON-mode fallback
- `NativeBrowserExecutor` — Playwright execution surface
- Browser sessions are video-recorded to `.browser-artifacts/` and linked back
  into the run snapshot so the observer can replay them
- `POST /cleanup` closes per-run browser sessions after completion

## Self-Improvement Mechanism

After each terminal run, `PostRunReflector`:

1. Analyzes the full step trajectory and extracts failure patterns as
   `MemoryRecord` entries (`FileBackedMemoryStore`).
2. On successful runs, compresses the trajectory into an `Episode`
   (`src/models/episode.py`) keyed by a normalized intent.

On future runs, both are surfaced: hints are injected into the rule engine and
the LLM prompt, and matching episodes act as an advisory trajectory cache.

## Setup

Python 3.11 is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Copy `.env.example` to `.env` and set your Gemini key:

```env
GOOGLE_API_KEY=your-gemini-api-key
```

Runtime environment variables:

- `OPERON_DESKTOP_BACKEND=json`
- `OPERON_BROWSER_BACKEND=computer_use`
- `OPERON_BROWSER_FALLBACK_BACKEND=json`
- `BROWSER_HEADLESS=true`
- `OPERON_TRACE=1` — print cyan `[TRACE]` lines from the loop

If PowerShell fails to launch external processes on Windows:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```

## Run

Start the API:

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Core endpoints:

- `POST /run-task` — create a run
- `POST /step` — advance a run one step
- `POST /resume` — resume a paused/stopped run
- `POST /stop` — stop an in-flight run
- `POST /cleanup` — close resources for a browser-native run
- `GET  /run/{run_id}` — read run state
- `GET  /health`

UI + observer:

- `GET /` or `/desktop-pilot` — Operon Pilot UI (desktop + browser)
- `GET /console` — Task Console UI
- `GET /observer/api/runs` — list stored runs
- `GET /observer/api/run/{run_id}` — full run bundle
- `GET /observer/api/export/{run_id}` — export a run as a zip bundle
- `GET /observer/api/live-browser/{run_id}` — live browser view for an active run
- `GET /observer/api/artifact` — fetch a single artifact file

## Run Data Layout

```
runs/
  <run_id>/
    state.json                 # AgentState snapshot
    run.jsonl                  # StepLog entries (one JSON object per line)
    step_1/
      before.png
      after.png
      perception_prompt.txt
      perception_raw.txt
      perception_parsed.json
      policy_prompt.txt
      policy_raw.txt
      policy_decision.json
      execution_trace.json     # includes reresolution_trace when fired
      progress_trace.json

.browser-artifacts/            # Playwright session video recordings
```

## Tests

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests -q
ruff check src tests --select E,F,W,I --ignore E501
```

Targeted suites:

```powershell
python -m pytest tests\test_agent_loop.py -q
python -m pytest tests\test_contracts.py tests\test_phase2_routing.py tests\test_phase3_adaptation.py -q
python -m pytest tests\test_phase4_integration.py tests\test_phase5_benchmark.py tests\test_phase5_executors.py -q
```

## Replay & Summarize

```powershell
python -m src.store.replay <run_id>
python -m src.store.summary <run_id>
python -m src.store.summary runs
```

## Commit Style

Short imperative subjects with prefixes: `Fix:`, `Docs:`, `CI:`, `Chore:`,
`Refactor:`. CI enforces Ruff `E,F,W,I` with `E501` (line length) ignored.
