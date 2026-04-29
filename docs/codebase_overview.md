# Operon Codebase Overview

## Core Concept

Operon is a **vision-only** desktop/browser automation engine. No DOM, no accessibility tree — it looks at screenshots (and sometimes video) to decide what to do, like a human would.

## The Loop (everything flows through `src/agent/loop.py`)

Every step follows this cycle:

```
Screenshot → Gemini "what's on screen?" → Pick action → Execute → Verify → Recover if needed
```

1. **Capture** — takes a screenshot (`mss`)
2. **Perceive** — sends it to Gemini, gets back structured `ScreenPerception` (elements, page hint, focused element)
3. **Decide** — `PolicyCoordinator` tries deterministic rules first (`PolicyRuleEngine`), falls back to Gemini LLM
4. **Execute** — `DesktopExecutor` (pyautogui) or `NativeBrowserExecutor` (Playwright) performs the action
5. **Verify** — checks if it worked; if no visual change detected, records a 3-second video and asks Gemini to analyze it
6. **Recover** — retry, continue, or stop

## Two Execution Paths

- **Desktop** — pyautogui + mss, full-screen control, combined JSON perception+policy via Gemini
- **Browser** — Playwright, primary backend is Gemini Computer Use (coordinate-based), JSON fallback available

Both share the same loop, verifier, recovery, and persistence.

## Key Architectural Layers

| Layer | Files | Role |
|---|---|---|
| **Agent loop** | `src/agent/loop.py` | Orchestrates the step cycle |
| **Backends** | `backend.py`, `combined.py`, `browser_computer_use.py`, `browser_json.py` | Perception+policy strategy per mode |
| **Policy** | `policy_coordinator.py`, `policy_rules.py`, `policy.py` | Rule engine (6 deterministic rules) + LLM fallback |
| **Selectors** | `selector.py`, `geometry.py` | Target element matching and re-resolution |
| **Executors** | `executor/desktop.py`, `executor/browser_native.py` | Translate actions to pyautogui/Playwright calls |
| **Verification** | `verifier.py`, `video_verifier.py`, `screen_diff.py` | Deterministic + video-based outcome checking |
| **Recovery** | `recovery.py`, `reflector.py` | Retry/stop decisions; post-run learning |
| **Models** | `src/models/` | All Pydantic v2 — state, perception, policy, execution, progress |
| **Persistence** | `src/store/` | File-backed run store + memory store, JSONL logging |
| **API** | `src/api/` | FastAPI server, Pilot UI, observer endpoints |
| **Gemini clients** | `src/clients/gemini.py`, `gemini_computer_use.py` | HTTP calls to Gemini (image + video payloads) |

## Self-Improvement Mechanism

After a run ends, `PostRunReflector` analyzes what happened, extracts failure patterns, and writes `MemoryRecord` entries. On future runs, `FileBackedMemoryStore` surfaces these as `MemoryHint`s — injected into both the rule engine and the LLM prompt. This means the system learns from past failures.

## Data Flow

A run is triggered via `POST /run-task`, advanced via `POST /step`. Everything is persisted under `runs/<run_id>/` — screenshots, prompts sent to Gemini, raw responses, parsed decisions, execution traces, and progress snapshots. Browser runs also get video under `.browser-artifacts/`.

## Unified Contract Layer (runs on top of the loop)

After each execution attempt, `AgentLoop._record_unified_step()` translates the step's legacy objects into four typed Pydantic contracts and validates them through the unified orchestrator:

```
LegacyOperonContractAdapter (src/runtime/legacy_adapter.py)
  → PerceptionOutput  (src/core/contracts/perception.py)  — semantic, no coordinates
  → PlannerOutput     (src/core/contracts/planner.py)     — action + rationale
  → ActorOutput       (src/core/contracts/actor.py)       — execution result
  → CriticOutput      (src/core/contracts/critic.py)      — success / retry / failure

UnifiedOrchestrator (src/runtime/orchestrator.py)
  → validates all four contracts for cross-field consistency
  → checks routing rules via src/core/router.py (allowed action types per environment)
  → updates AgentRuntimeState (src/runtime/state.py) per run

src/executor/browser_adapter.py / src/executor/desktop_adapter.py
  → thin adapters that translate PlannerAction → legacy AgentAction
```

This layer is an **observer and validator**, not a replacement for `AgentLoop`. It tracks unified state and can drive adaptation retries (via `orchestrator.adaptation_strategy_for()`), but the loop still owns execution ordering and termination. See `docs/migration_cleanup.md` for the full classification of active vs. deleted files.

## What Makes It Interesting

- **No DOM dependence** — purely visual, works on any app
- **Hardened execution** — if a target moves between perceive and execute, it re-resolves deterministically using `TargetIntent` + `DeterministicTargetSelector`
- **Redundancy blocking** — `ProgressState` counters prevent the agent from repeating the same failed action
- **Video verification** — when a screenshot diff shows no change, it records video of the action and asks Gemini to analyze temporal evidence
