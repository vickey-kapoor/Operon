# Operon Architecture Reference
_Last refreshed: 2026-04-29_

## Directory Layout

```
src/
  agent/        loop.py, perception.py, policy.py, policy_coordinator.py,
                policy_rules.py, verifier.py, recovery.py, reflector.py,
                benchmark.py, capture.py, selector.py, hitl.py,
                backend.py, action_translation.py, fallback_backend.py
  api/          server.py, routes.py, runtime_config.py, benchmark_suite.py
  clients/      gemini.py, anthropic.py
  executor/     browser.py (ABC), browser_native.py, desktop.py,
                browser_adapter.py, desktop_adapter.py, os_picker_macro.py
  models/       state.py, capture.py, perception.py, policy.py,
                execution.py, verification.py, recovery.py, memory.py,
                progress.py, common.py, logs.py, selector.py
  store/        run_store.py, memory.py, run_logger.py,
                background_writer.py, replay.py, summary.py
  benchmarks/   (benchmark-specific plugin files)
  runtime/      orchestrator.py, state.py, legacy_adapter.py, benchmark_runner.py
  core/         contracts/, router.py
prompts/        browser_combined_prompt.txt, browser_computer_use_prompt.txt,
                critic_prompt.txt, desktop_combined_prompt.txt,
                desktop_perception_prompt.txt, desktop_policy_prompt.txt,
                perception_prompt.txt, policy_prompt.txt,
                video_verification_prompt.txt
tests/          55+ test files
docs/           architecture.md (this file), agents.md, codebase_overview.md,
                product_requirements.md, claude_task.md, agent_contract.md,
                migration_cleanup.md, phase4_migration_note.md, upload_paths.md
```

---

## Control Loop (`src/agent/loop.py` — class `AgentLoop`)

### `__init__` key params
`capture_service`, `perception_service`, `run_store`, `policy_service`, `executor`, `verifier_service`, `recovery_manager`, `memory_store`, `gemini_client` (dedicated for video verify), `environment`, `unified_orchestrator`

### Step order in `step_run()`
1. ORCHESTRATE — unified orchestrator check
2. CAPTURE — burst 3 frames, pick stable frame, measure `visual_velocity`
3. PERCEIVE — typed `ScreenPerception` from Gemini
4. UPDATE_STATE — append to `observation_history`
5. CHOOSE_ACTION — rule engine → episode replay → LLM (`PolicyDecision`)
6. EXECUTE — `_execute_with_hardening()`, one bounded retry on drift
7. VERIFY — `check_terminal_state()` first, then action-level check
8. `_maybe_video_verify()` — fires only for `expected_change ∈ {content,navigation,dialog}` AND ratio < threshold AND idempotent action AND `video_path` not None
9. RECOVER — escalation ladder → `_apply_recovery_actions()`
10. Record rule trace, decay hints on failure, write JSONL step log

### Terminal conditions (success)
`FORM_SUBMITTED_SUCCESS`, `STOP_BEFORE_SEND`, `TASK_COMPLETED`, page_hint==FORM_SUCCESS

### Key constants
`MAX_NO_PROGRESS_STEPS=3`, `MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS=2`, `RECENT_WINDOW_SIZE=6`

---

## Models (`src/models/`)

### Enums
| Name | Values |
|---|---|
| `RunStatus` | PENDING, RUNNING, WAITING_FOR_USER, SUCCEEDED, FAILED, CANCELLED |
| `LoopStage` | ORCHESTRATE, CAPTURE, PERCEIVE, UPDATE_STATE, CHOOSE_ACTION, EXECUTE, VERIFY, RECOVER |
| `ActionType` | CLICK, DOUBLE_CLICK, TYPE, SELECT, NAVIGATE, HOTKEY, PRESS_KEY, SCROLL, DRAG, HOVER, LAUNCH_APP, WAIT, WAIT_FOR_USER, STOP, BATCH, READ_CLIPBOARD, WRITE_CLIPBOARD, SCREENSHOT_REGION, FILE_PORTER, RIGHT_CLICK, FOCUS |
| `ExpectedChange` | NONE, FOCUS, CONTENT, NAVIGATION, DIALOG — gates video verifier |
| `RecoveryStrategy` | RETRY_SAME_STEP, RETRY_DIFFERENT_TACTIC, WAIT_AND_RETRY, BACKOFF, CONTEXT_RESET, SESSION_RESET, STOP, ADVANCE |
| `PageHint` | FORM_PAGE, FORM_SUCCESS, UNKNOWN |
| `UIElementType` | BUTTON, INPUT, LINK, TEXT, DIALOG, ICON, WINDOW, UNKNOWN |

### Key Pydantic models
| Model | File | Key fields |
|---|---|---|
| `AgentState` | state.py | run_id, intent, status, step_count, start_url, last_rule_trace, observation_history |
| `CaptureFrame` | capture.py | image_path, timestamp, visual_velocity: float |
| `ScreenPerception` | perception.py | page_hint, url, visible_elements: list[UIElement], focused_element_id |
| `UIElement` | perception.py | element_id, element_type, x, y, width, height, is_interactable, canonical_name |
| `PolicyDecision` | policy.py | action: AgentAction, rationale, confidence, active_subgoal, expected_change, rule_name |
| `AgentAction` | policy.py | action_type, x, y, text, key, url, clear_before_typing, press_enter, actions (batch) |
| `ExecutedAction` | execution.py | action, execution: ExecutionTrace, failure_category, duration_ms |
| `VerificationResult` | verification.py | status, expected_outcome_met, stop_condition_met, failure_type, failure_category |
| `RecoveryDecision` | recovery.py | strategy, message, retry_after_ms, failure_category, terminal, stop_reason |
| `MemoryRecord` | memory.py | key, benchmark, hint, outcome, page_hint, subgoal, weight (decays on failure) |
| `MemoryHint` | memory.py | key, hint, source, count |
| `StepLog` | logs.py | run_id, step_index, artifact paths, perception/policy/execution/verification/recovery |

All models inherit `StrictModel` (extra="forbid").

---

## Policy Layer

### Rule engine primitives (priority order, `src/agent/policy_rules.py`)
| Rule | Trigger |
|---|---|
| `_human_intervention_rule` | page_hint contains HITL keyword × 2 consecutive steps (debounced) |
| `_task_success_stop_rule` | page_hint==FORM_SUCCESS or success tokens visible |
| `_dropdown_menu_select_rule` | dropdown open + unselected option visible |
| `_avoid_identical_type_retry` | memory hint `avoid_identical_type_retry` + repeated TYPE failure |
| `_no_progress_recovery_rule` | no-progress streak > threshold |
| `_dismiss_blocking_overlay_rule` | blocking dialog/banner detected |
| `_search_query_rule` | intent has search query + search input visible |

Benchmark-specific rules registered via `BENCHMARK_REGISTRY` dict, run before primitives.

### Rule-Augmented Generation
After each step, if `decision.rule_name` is set → `state.last_rule_trace = "[RULE TRACE] Rule 'X' fired at step N | action=Y | outcome=Z"`. Injected into next LLM prompt via `add_advisory_hints(source="rule_trace")`. Cleared when LLM decides.

### `parse_policy_output()` — loud failure
Explicitly checks `"expected_change" in parsed` before `model_validate()`. Raises `PolicyError` if missing — Gemini MUST emit the field.

---

## Executor Layer

### DesktopExecutor (`src/executor/desktop.py`)
- Pyautogui + mss, full-screen control
- `validate_display_baseline()` → raises `HardwareBaselineError` if resolution < 1280×720
- `_region_has_content(x, y, radius=50)` — 100×100px crop, variance < 20 → abort click with `CoordDriftWarning`
- `context_reset()` — Escape×2, click body center, Ctrl+Home
- `session_reset(start_url)` — context_reset + Alt+Tab cycle

### NativeBrowserExecutor (`src/executor/browser_native.py`)
- Playwright-based, video-recorded under `.browser-artifacts/`
- `context_reset()` — clear cookies, reset zoom, focus window
- `session_reset(start_url)` — close tab, open fresh session, navigate to start_url
- `current_url_for_run(run_id)` — live URL for observer

### Executor ABC (`src/executor/browser.py`)
- `context_reset()` / `session_reset()` — no-op defaults; overridden above
- `capture()` → `CaptureFrame`
- `execute(action)` → `ExecutedAction`

---

## Perception

`GeminiPerceptionService.perceive(screenshot: CaptureFrame, state: AgentState) → ScreenPerception`

Quality gate: min 1 interactive element; on FORM_PAGE: min 2 interactive + 1 text. Retry once on quality failure. Raises `PerceptionLowQualityError` on second failure.

Desktop uses `desktop_perception_prompt.txt`. Browser uses `perception_prompt.txt`. Combined uses `desktop_combined_prompt.txt` / `browser_combined_prompt.txt`.

---

## Verifier

`DeterministicVerifierService.verify(state, decision, executed_action) → VerificationResult`

1. `check_terminal_state()` — checks for success page hint or benchmark success tokens first. Returns terminal SUCCESS immediately if confirmed. Returns None if no evidence.
2. Only if not terminal: evaluate action-level success/failure.
3. `_is_page_loading()` — shimmer/spinner/< 3 elements → PENDING (don't verify yet).

---

## Recovery

`RuleBasedRecoveryManager.recover(state, decision, executed_action, verification) → RecoveryDecision`

Escalation ladder (keyed on `state.retry_count`):
- 0 → RETRY_SAME_STEP
- 1 → RETRY_DIFFERENT_TACTIC (wait 500ms)
- 2 → WAIT_AND_RETRY (wait 1000ms)
- 3 → CONTEXT_RESET
- 4 → SESSION_RESET
- ≥5 → STOP (MAX_RECOVERY_ATTEMPTS=5)

`_apply_recovery_actions()` in loop.py calls `executor.context_reset()` / `executor.session_reset()`.

---

## Memory / Store

### FileBackedMemoryStore (`src/store/memory.py`)
- `memory/memory.jsonl` — append-only MemoryRecord log
- `memory/episodes.jsonl` — compressed episode trajectories
- `get_hints(benchmark, page_hint, subgoal, recent_failure_category, limit=4)` — weight-ranked
- `record_step()` — writes compact record; on failure, `_decay_active_hints()` halves weight of active hints
- `_WEIGHT_PRUNE_THRESHOLD = 0.1` — pruned from results below this
- Episodes only from successful runs, only 1st-attempt steps (no recovered failures)

### Run data layout
```
runs/<run_id>/
  state.json, run.jsonl
  step_N/before.png, after.png, perception_parsed.json,
         policy_decision.json, execution_trace.json,
         verification_result.json, selector_trace.json
memory/memory.jsonl, episodes.jsonl
```

---

## API Routes (`src/api/routes.py`)

### Browser
`POST /run-task`, `POST /step`, `POST /resume`, `POST /stop`, `POST /cleanup`, `GET /run/{id}`, `GET /health`

### Desktop
`POST /desktop/run-task`, `POST /desktop/step`, `POST /desktop/resume`, `POST /desktop/cleanup`, `GET /desktop/run/{id}`

### Observer / Telemetry
`GET /observer/api/runs`, `/observer/api/run/{id}`, `/observer/api/usage`, `/observer/api/artifact`, `/observer/api/export/{id}`, `/observer/api/live-browser/{id}`

### Benchmarks
`POST /benchmark/run-suite`, `POST /benchmark/stop-suite/{id}`, `POST /benchmark/run-task`, `GET /benchmark/tasks`, `GET /benchmark/suite/{id}`

### Static UI
`/` → landing.html, `/console` → console.html, `/dashboard` → dashboard.html, `/benchmarks` → benchmarks.html

### Loop builders
Dedicated `video_gemini_client = GeminiHttpClient(...)` always passed as `gemini_client` to `AgentLoop` in both `get_agent_loop()` and `get_desktop_agent_loop()` — independent of verifier provider.

---

## Clients

### GeminiHttpClient (`src/clients/gemini.py`)
Default model: `gemini-2.5-flash` (env: `GEMINI_MODEL`). Browser model: `gemini-2.5-computer-use-preview-10-2025`. Methods: `generate_perception()`, `generate_policy()`, `generate_video_verification()`.

### AnthropicHttpClient (`src/clients/anthropic.py`)
Default model: `claude-sonnet-4-20250514`. Methods: `generate_policy()` (text-only), `generate_verification()` (image+text).

---

## Runtime / Phase 2 (`runtime/`)

- `UnifiedOrchestrator.process_step(perception, planner, actor, critic, current_state) → StepState`
- `AgentRuntimeState` — shared mutable state per run (subgoal, url, retry context, advisory hints)
- `LegacyOperonContractAdapter` — bridges Phase 1 `AgentLoop` API to Phase 2 contracts

---

## Environment Variables

### Desktop
`OPERON_DESKTOP_BACKEND` (json), `OPERON_DESKTOP_MODEL`, `OPERON_DESKTOP_PLANNER_PROVIDER` (gemini|anthropic), `OPERON_DESKTOP_PLANNER_MODEL`, `OPERON_DESKTOP_VERIFIER_PROVIDER`, `OPERON_DESKTOP_VERIFIER_MODEL`, `OPERON_DESKTOP_FALLBACK_MODEL`

### Browser
`OPERON_BROWSER_BACKEND` (computer_use), `OPERON_BROWSER_MODEL`, `OPERON_BROWSER_PLANNER_PROVIDER`, `OPERON_BROWSER_PLANNER_MODEL`, `OPERON_BROWSER_VERIFIER_PROVIDER`, `OPERON_BROWSER_VERIFIER_MODEL`, `OPERON_BROWSER_FALLBACK_BACKEND` (json), `OPERON_BROWSER_FALLBACK_MODEL`

### Global
`GEMINI_API_KEY` / `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPERON_USE_VERTEX`, `VERTEX_PROJECT`, `VERTEX_LOCATION`, `OPERON_TRACE` (1=enable trace events), `OPERON_TEST_SAFE_MODE`, `BROWSER_WIDTH` (1920), `BROWSER_HEIGHT` (1080), `BROWSER_HEADLESS`, `CORS_ORIGINS`, `FORM_BENCHMARK_URL`

---

## Tests (55+ files)

Key files: `test_agent_loop.py`, `test_policy.py`, `test_policy_coordinator.py`, `test_verifier.py`, `test_recovery.py`, `test_perception.py`, `test_memory.py`, `test_desktop_executor.py`, `test_browser_native.py`, `test_video_verification.py`, `test_desktop_routes.py`, `test_schema_validation.py`, `test_episode.py`, `test_reflector.py`, `test_gemini_client.py`, `test_action_translation.py`, `test_selector.py`, `test_run_store_and_logging.py`, `test_failure_taxonomy.py`, `test_hitl.py`, `test_webarena.py`

Excluded from default run: `test_e2e_quick_tasks.py`, `test_bug_fixes_verification.py` (live server required).

Safe default command:
```powershell
$env:GEMINI_API_KEY = "fake-test-key"
.venv\Scripts\python -m pytest tests\ -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py
```

---

## Invariants (never bypass)

1. **Atomic TYPE**: executor merges click+type into one call. Never emit CLICK then TYPE on same element from policy.
2. **Visual Servo**: `_region_has_content()` runs before every click. Variance < 20 → abort with `CoordDriftWarning`.
3. **expected_change required**: Gemini policy output must include `expected_change` field. Missing → `PolicyError` (loud fail).
4. **No DOM/selectors**: vision-only. No XPath, CSS, accessibility tree, Playwright `locator()` in policy/perception path.
5. **Rules before LLM**: if a condition is expressible as a Python predicate, it belongs in `PolicyRuleEngine`.
