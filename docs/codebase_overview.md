# Operon Codebase Overview

## Core Concept

Operon is a **vision-only** desktop/browser automation engine. No DOM, no accessibility tree — it looks at screenshots (and optionally video) to decide what to do, exactly like a human would.

## The Loop (everything flows through `src/agent/loop.py`)

Every step follows this cycle:

```
Screenshot → stabilise → perceive → smooth + ghost-detect → decide → anchor-check → execute → verify → recover
```

1. **Capture** — takes a 3-frame burst screenshot. Measures `visual_velocity` (pixel diff between frames). If velocity > 5%, clears the `RollingElementBuffer` to prevent stale ghost coordinates from a previous app context.
2. **Perceive** — sends it to Gemini, gets back structured `ScreenPerception`. Then:
   - **Coordinate smoothing**: per-element jitter < 3 px snapped back to previous-frame values (eliminates Gemini sub-pixel variance that would cause targets to vibrate).
   - **Ghost detection**: elements visible in T-1 but absent in T-0 on a stable screen become `GhostElement`s. TTL = 2 frames — they're purged automatically if the element doesn't reappear.
3. **Decide** — `PolicyCoordinator` runs in order: benchmark plugins → rule engine (`PolicyRuleEngine`) → LLM. Post-LLM, `_semantic_anchor_check` inspects the decision's coordinates: if the point is > 15 px from every visible element's bounding box, the action is intercepted and the agent re-perceives with an anchor hint injected into the next LLM prompt.
4. **Execute** — `DesktopExecutor` (pyautogui, adaptive servo) or `NativeBrowserExecutor` (Playwright). Before every click, `_region_has_content()` samples a 100×100 px crop. A calibrated variance threshold (tuned at startup from 5 desktop crops) gates whether the click proceeds or aborts with `CoordDriftWarning`.
5. **Verify** — checks if it worked. In order:
   - Terminal state check (success page hint / benchmark tokens) — exits immediately on success
   - Reaction check: sends before+after frames to Gemini, looks for micro-reactions (ripple, focus ring, spinner) → `PROGRESSING_STABLE` (advance)
   - Motion check: if screen is actively changing → `STABLE_WAIT` (wait 200 ms, re-capture, re-verify once)
   - Page loading check → `PENDING` (2–8 s exponential backoff)
   - Model critic: Gemini image+text assessment
6. **Recover** — retry, wait, context reset, session reset, or stop based on the escalation ladder

## Two Execution Paths

- **Desktop** — pyautogui + mss, full-screen control. Combined JSON perception+policy via Gemini, or separate perception + Anthropic/Gemini planner.
- **Browser** — Playwright. Primary backend: Gemini Computer Use (coordinate-based, multi-turn). JSON fallback available. Sessions video-recorded under `.browser-artifacts/`.

Both share the same loop, verifier, recovery, persistence, and memory layers.

## Key Architectural Layers

| Layer | Files | Role |
|---|---|---|
| **Agent loop** | `src/agent/loop.py` | Orchestrates the step cycle; owns `STABLE_WAIT` and `PENDING` re-verify logic |
| **Backends** | `backend.py`, `combined.py`, `browser_computer_use.py`, `browser_json.py` | Perception+policy strategy per mode |
| **Policy** | `policy_coordinator.py`, `policy_rules.py`, `policy.py` | Rule engine (8 deterministic rules) + LLM + post-LLM anchor check |
| **Spatial persistence** | `models/memory.py` (`RollingElementBuffer`), `agent/perception.py` | 3-frame coord smoothing, ghost TTL, buffer clear on high velocity |
| **Selectors** | `selector.py` | Target element matching, re-resolution on drift |
| **Executors** | `executor/desktop.py`, `executor/browser_native.py` | pyautogui/Playwright + adaptive visual servo |
| **Verification** | `verifier.py`, `video_verifier.py`, `screen_diff.py` | 6-status verification: SUCCESS / FAILURE / UNCERTAIN / PENDING / PROGRESSING_STABLE / STABLE_WAIT |
| **Recovery** | `recovery.py`, `reflector.py` | Retry/stop escalation; post-run learning |
| **Models** | `src/models/` | Pydantic v2 — state, perception (GhostElement), policy, execution (visual_variance), verification (6 statuses), logs (decision_source) |
| **Persistence** | `src/store/` | File-backed run store + memory store, JSONL logging |
| **API** | `src/api/` | FastAPI server, Pilot UI, observer (live log with `[RULE]`/`[LLM]` colouring) |
| **Gemini clients** | `src/clients/gemini.py`, `gemini_computer_use.py` | HTTP calls: single image, video, multi-image (reaction check) |

## Spatial Persistence in Detail

`RollingElementBuffer` (3-frame deque in `src/models/memory.py`) is the single source of truth for cross-step element state:

- **Coordinate smoothing** — tiny per-element jitter corrected before ghost detection so false ghosts aren't created by sub-pixel drift.
- **Ghost elements** — absent elements with stable screen marked as `GhostElement` in `ScreenPerception.ghost_elements`. TTL=2 means they survive 2 more frames max. The policy prompt explains to the LLM that ghosts are occluded, not gone.
- **High-velocity clear** — any `visual_velocity > 5%` clears the buffer entirely; a major UI transition (app switch, full navigation) must not bleed prior-window coordinates into the new context.
- **Run reset** — `PolicyCoordinator.reset_run_context()` also clears the buffer via `element_buffer.clear()`.

## Verification Status Model

| Status | What it means | What the loop does |
|---|---|---|
| `SUCCESS` | Outcome confirmed | Advance or stop |
| `FAILURE` | Action definitively failed | Recovery ladder |
| `UNCERTAIN` | Can't confirm | Recovery ladder |
| `PENDING` | Page mid-load (blank/sparse) | Backoff 2–4–8 s, re-verify |
| `PROGRESSING_STABLE` | Gemini confirmed UI micro-reaction | Advance immediately — no retry |
| `STABLE_WAIT` | Screen actively animating post-action | 200 ms wait, re-capture, re-verify once |

## Step-Level Logging

Every `StepLog` in `run.jsonl` now carries:
- `decision_source`: `"[RULE] rule_name"` or `"[LLM] gemini"` — who made the policy decision
- `visual_variance`: float from the servo check — the raw pixel² variance at the click target

The observer emits a `decision_source` event per step. Console.html colours rule decisions green and LLM decisions blue in the live log feed.

## Self-Improvement Mechanism

After a run ends, `PostRunReflector` analyzes what happened, extracts failure patterns, and writes `MemoryRecord` entries. On future runs, `FileBackedMemoryStore` surfaces these as `MemoryHint`s — injected into both the rule engine and the LLM prompt. Hints decay geometrically on failure (weight halved each time) and are pruned when weight < 0.1. Successful runs yield `Episode`s — compressed optimal trajectories replayed on similar future tasks.

## Data Flow

A run is triggered via `POST /run-task`, advanced via `POST /step`. Everything is persisted under `runs/<run_id>/` — screenshots, all Gemini prompts and raw responses, parsed decisions, execution traces, and progress snapshots. Browser runs also produce video under `.browser-artifacts/`.

## Unified Contract Layer (observer on top of the loop)

After each step, `AgentLoop._record_unified_step()` translates to four typed Pydantic contracts:

```
LegacyOperonContractAdapter  (src/runtime/legacy_adapter.py)
  → PerceptionOutput  — semantic, no coordinates
  → PlannerOutput     — action + rationale
  → ActorOutput       — execution result
  → CriticOutput      — success / retry / failure

UnifiedOrchestrator  (src/runtime/orchestrator.py)
  → validates cross-field consistency
  → checks routing rules (src/core/router.py)
  → updates AgentRuntimeState per run
```

This is an **observer and validator**, not a replacement for `AgentLoop`. The loop still owns execution ordering and termination.
