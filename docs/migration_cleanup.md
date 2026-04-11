# Migration Cleanup Audit

**Date:** 2026-04-11
**Scope:** Classify all files relative to the unified architecture introduced in
`runtime/`, `core/`, and `executors/`. Determine what is active, what is
transitional, and what can be removed in a future pass.

---

## Active Control Path

The single active control path is:

```
AgentLoop (src/agent/loop.py)
  → CaptureService        (src/agent/capture.py)
  → PerceptionService     (src/agent/perception.py  /  src/agent/combined.py)
  → PolicyCoordinator     (src/agent/policy_coordinator.py)
      → PolicyRuleEngine  (src/agent/policy_rules.py)
      → GeminiPolicyService (src/agent/policy.py)
  → DeterministicTargetSelector (src/agent/selector.py + geometry.py)
  → DesktopExecutor / NativeBrowserExecutor (src/executor/)
  → DeterministicVerifierService (src/agent/verifier.py)
  → VideoVerifier         (src/agent/video_verifier.py)
  → RuleBasedRecoveryManager (src/agent/recovery.py)
  → PostRunReflector      (src/agent/reflector.py)
  → FileBackedRunStore    (src/store/run_store.py)
  → FileBackedMemoryStore (src/store/memory.py)
  → run_logger            (src/store/run_logger.py)
```

The unified layer runs **on top of** the legacy loop as a post-step
recorder/validator. It does NOT replace the loop — it observes and validates
each step for cross-environment contract compliance:

```
LegacyOperonContractAdapter (runtime/legacy_adapter.py)
  → translates each step's legacy objects into unified contracts
UnifiedOrchestrator (runtime/orchestrator.py)
  → validates the contract bundle and updates AgentRuntimeState
core/router.py
  → environment-level routing rules (browser vs desktop action sets)
core/contracts/
  → strict Pydantic contracts (perception, planner, actor, critic)
executors/browser_executor.py / desktop_executor.py
  → thin adapters used by test_phase4_integration and test_phase5_executors
```

---

## Files: Kept (and why)

### src/agent/

| File | Status | Reason |
|---|---|---|
| `capture.py` | **Active** | Only capture implementation; imported by loop.py and routes.py |
| `perception.py` | **Active** | Perception parsing; imported by loop.py, combined.py, tests |
| `policy.py` | **Active** | PolicyService ABC + GeminiPolicyService + parse_policy_output; imported by policy_coordinator.py, routes.py, many tests |
| `policy_coordinator.py` | **Active** | Wraps rule engine + LLM policy; wired as `policy_service` in routes.py |
| `policy_rules.py` | **Active** | Deterministic rule engine called by PolicyCoordinator every step |
| `recovery.py` | **Active** | RecoveryManager ABC + RuleBasedRecoveryManager; wired in routes.py |
| `verifier.py` | **Active** | VerifierService ABC + DeterministicVerifierService; wired in routes.py |
| `reflector.py` | **Active** | PostRunReflector; called by loop.py on run completion |
| `selector.py` | **Active** | DeterministicTargetSelector; used by loop.py and policy_rules.py |
| `geometry.py` | **Active** | Spatial helpers imported by selector.py |
| `video_verifier.py` | **Active** | VideoVerifier called by loop.py when screen diff detects no change |
| `backend.py` | **Active** | AgentBackend ABC; extended by combined.py, browser_computer_use.py, fallback_backend.py |
| `combined.py` | **Active** | CombinedPerceptionPolicyService; default desktop + JSON browser backend |
| `fallback_backend.py` | **Active** | FallbackBackend; wired in routes.py for computer_use with JSON fallback |
| `browser_json.py` | **Active** | BrowserJsonBackend (thin combined.py subclass for browser prompt path) |
| `browser_computer_use.py` | **Active** | BrowserComputerUseBackend; primary browser backend for Gemini Computer Use |
| `action_translation.py` | **Active** | Normalizes Computer Use function calls to internal AgentAction schema |
| `benchmark.py` | **Active** | Entry point for local form + Gmail benchmark runs (`python -m src.agent.benchmark`) |
| `capture.py` | **Active** | ScreenCaptureService |
| `screen_diff.py` | **Active** | Imported by loop.py at runtime for video-verify gating |
| `screen_recorder.py` | **Active** | Imported by video_verifier.py |

### src/store/

| File | Status | Reason |
|---|---|---|
| `memory.py` | **Active** | FileBackedMemoryStore + benchmark helpers; used by loop.py, policy_coordinator.py |
| `run_store.py` | **Active** | FileBackedRunStore; used by loop.py, routes.py |
| `run_logger.py` | **Active** | append_step_log; used by loop.py |
| `replay.py` | **Active** | load_run_replay; used by observer.py |
| `summary.py` | **Active** | _load_state_from_path, generate_run_metrics; used by observer.py and benchmark.py |
| `background_writer.py` | **Active** | bg_writer; used by policy.py, policy_coordinator.py, perception.py, combined.py |

### runtime/

| File | Status | Reason |
|---|---|---|
| `orchestrator.py` | **Active** | UnifiedOrchestrator; used by loop.py and many tests (test_contracts, test_phase2_routing, test_phase3_adaptation) |
| `state.py` | **Active** | AgentRuntimeState + StepState; used by loop.py, orchestrator.py, tests |
| `legacy_adapter.py` | **Active** | LegacyOperonContractAdapter; used by loop.py to translate legacy objects per step |
| `benchmark_runner.py` | **Active** | Phase5BenchmarkRunner; used by test_phase5_benchmark.py |

### core/

| File | Status | Reason |
|---|---|---|
| `router.py` | **Active** | validate_plan_route, route_plan; used by orchestrator.py, test_contracts, test_phase2_routing |
| `contracts/perception.py` | **Active** | PerceptionOutput; used throughout |
| `contracts/planner.py` | **Active** | PlannerOutput, PlannerAction, ActionType; used throughout |
| `contracts/actor.py` | **Active** | ActorOutput, ActorAction, ExecutorChoice; used throughout |
| `contracts/critic.py` | **Active** | CriticOutput, CriticOutcome, FailureType; used throughout |

### executors/

| File | Status | Reason |
|---|---|---|
| `browser_executor.py` | **Active** | BrowserExecutor thin adapter; used by loop.py and test_phase4_integration, test_phase5_executors |
| `desktop_executor.py` | **Active** | DesktopExecutor thin adapter; used by loop.py and test_phase5_executors |

---

## Files: Deprecated

None remaining. All previously deprecated files have been deleted (see below).

---

## Files: Deleted

| File | Reason |
|---|---|
| `runtime/run_phase5_live.py` | Standalone phase-specific live harness. Not imported or tested anywhere. Hardcoded task list duplicating `benchmark_runner.py`. |
| `runtime/run_upload_live.py` | One-off cross-environment demo with hardcoded machine paths. Not imported or tested anywhere. |

---

## Risky Leftovers to Revisit

### 1. Domain bleed in policy_rules.py and PageHint

`PolicyRuleEngine` contains benchmark-specific rules (`_login_page_guardrail`,
`_compose_already_visible_rule`, `_submit_form_when_ready_rule`) that are
hardcoded for Gmail and the form benchmark. `PageHint` enum has four of six
values that are Gmail-specific. These are still the active rules for every run.

**Risk:** A third benchmark will require splitting these into a base engine rule
set and per-benchmark rule registrations.

**Canonical location when split:** `core/` (engine primitives) vs
`src/agent/policy_rules.py` (benchmark-specific overlays).

### 2. PolicyCoordinator vs core/router.py — no overlap, but confusingly named

`core/router.py` validates environment-level routing (which action types are
allowed in browser vs desktop). `PolicyCoordinator` + `PolicyRuleEngine`
implement step-level decision rules (when to stop, when to focus, etc.). These
are genuinely distinct concerns, not duplicates, but their names suggest
overlap.

### 3. UnifiedOrchestrator is observer-only, not the primary control path

`UnifiedOrchestrator.process_step()` runs as a post-step recorder inside
`loop.py._record_unified_step()`. It validates each step's contract bundle and
updates `AgentRuntimeState`, but it does not control execution ordering,
retries, or termination — `AgentLoop` still owns that. This is intentional
during the migration period but should be documented as a gap: the orchestrator
cannot yet drive the loop end-to-end without the legacy `AgentLoop` wrapper.

### 4. benchmark_runner.py Phase 5 tasks use stub callbacks

`run_phase5_benchmark_suite()` in `runtime/benchmark_runner.py` accepts
`run_task_fn` and `get_status_fn` as external callbacks. The Phase 5 tests use
async stubs. The harness is not yet wired to the real `AgentLoop.start_run()`
and `AgentLoop.step_run()` — connecting these would complete the Phase 5 live
run path.

---

## What the Single Active Control Path Is

1. HTTP request hits `POST /run-task` or `POST /step` in `src/api/routes.py`
2. `get_agent_loop()` returns the singleton `AgentLoop`
3. `AgentLoop.step_run()` runs: capture → perceive → rule check → LLM policy → execute → verify → video-verify → recover → reflect → log
4. After each execution attempt, `_record_unified_step()` translates the step to unified contracts via `LegacyOperonContractAdapter` and validates them through `UnifiedOrchestrator`
5. `AgentRuntimeState` is updated per-run in `loop.unified_states`
6. Terminal runs trigger `PostRunReflector.reflect()` to write `MemoryRecord` entries for future runs
