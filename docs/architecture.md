# Operon Architecture Reference
_Last refreshed: 2026-04-29 · Updated: 2026-05-01_

## Directory Layout

```
src/
  agent/        loop.py, perception.py, policy.py, policy_coordinator.py,
                policy_rules.py, verifier.py, video_verifier.py, recovery.py,
                reflector.py, benchmark.py, capture.py, selector.py, hitl.py,
                screen_diff.py, backend.py, action_translation.py,
                fallback_backend.py, combined.py
  api/          server.py, routes.py, observer.py, runtime_config.py,
                benchmark_suite.py, static/
  clients/      gemini.py, anthropic.py, gemini_computer_use.py
  executor/     browser.py (ABC), browser_native.py, desktop.py,
                browser_adapter.py, desktop_adapter.py, os_picker_macro.py
  models/       state.py, capture.py, perception.py, policy.py,
                execution.py, verification.py, recovery.py, memory.py,
                progress.py, common.py, logs.py, selector.py, usage.py
  store/        run_store.py, memory.py, run_logger.py,
                background_writer.py, replay.py, summary.py
  benchmarks/   (benchmark-specific plugin files + registry)
  runtime/      orchestrator.py, state.py, legacy_adapter.py, benchmark_runner.py
  core/         contracts/, router.py
prompts/        browser_combined_prompt.txt, browser_computer_use_prompt.txt,
                critic_prompt.txt, desktop_combined_prompt.txt,
                desktop_perception_prompt.txt, desktop_policy_prompt.txt,
                perception_prompt.txt, policy_prompt.txt,
                video_verification_prompt.txt, reaction_check_prompt.txt
tests/          60+ test files
docs/           architecture.md, agents.md, codebase_overview.md,
                product_requirements.md, agent_contract.md,
                migration_cleanup.md, upload_paths.md
```

---

## Control Loop (`src/agent/loop.py` — class `AgentLoop`)

### `__init__` key params
`capture_service`, `perception_service`, `run_store`, `policy_service`, `executor`,
`verifier_service`, `recovery_manager`, `memory_store`, `gemini_client` (dedicated for
video verify), `environment`, `unified_orchestrator`

### Step order in `step_run()`
1. **ORCHESTRATE** — unified orchestrator check
2. **CAPTURE** — burst 3 frames, pick stable frame, measure `visual_velocity`
   - If `visual_velocity > 5%`: `element_buffer.clear()` on the perception service — prevents cross-app ghost contamination after major UI transitions
3. **PERCEIVE** — typed `ScreenPerception` from Gemini
   - Coordinate smoothing: per-element jitter < 3 px snapped to previous-frame coords when velocity < 2%
   - Ghost detection: elements in T-1 absent in T-0 on stable screen → `GhostElement`s with 2-frame TTL via `RollingElementBuffer`
4. **UPDATE_STATE** — append to `observation_history`
5. **CHOOSE_ACTION** — rule engine → episode replay → LLM → post-LLM guards (`PolicyDecision`)
   - Post-LLM: `_semantic_anchor_check` in `PolicyCoordinator` — if coord > 15 px from every element, intercept with WAIT + anchor hint
6. **EXECUTE** — `_execute_with_hardening()`, one bounded retry on target drift
7. **VERIFY** — `check_terminal_state()` first, then:
   - Reaction check (CLICK/TYPE): `VideoVerifier.verify_reaction()` → `PROGRESSING_STABLE`
   - Motion check: `_screen_is_in_motion()` → `STABLE_WAIT` (200 ms retry)
   - Page loading: `_is_page_loading()` → `PENDING` (2–8 s backoff)
   - Model critic: Gemini image+text critique
8. **`_maybe_video_verify()`** — fires only for `expected_change ∈ {content,navigation,dialog}` AND ratio < threshold AND idempotent action AND `video_path` not None AND `verification.video_verified` is False
9. **RECOVER** — escalation ladder → `_apply_recovery_actions()`
10. Record rule trace, decay hints on failure, write JSONL step log with `decision_source` + `visual_variance`

### STABLE_WAIT handling (`_wait_for_ui_stable`)
When `verify()` returns `STABLE_WAIT` (screen actively changing post-action):
1. `asyncio.sleep(0.2)` — 200 ms
2. Re-capture + re-perceive
3. Re-call `verify()` once
4. If still `STABLE_WAIT` → downgrade to `UNCERTAIN`

### PENDING handling (`_wait_for_page_load`)
Exponential backoff: 2 s → 4 s → 8 s (14 s total). Downgrades to `UNCERTAIN` if page never settles.

### Terminal conditions (success)
`FORM_SUBMITTED_SUCCESS`, `STOP_BEFORE_SEND`, `TASK_COMPLETED`, page_hint==FORM_SUCCESS

### Key constants
`MAX_NO_PROGRESS_STEPS=3`, `MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS=2`,
`RECENT_WINDOW_SIZE=6`, `_PENDING_BACKOFF_DELAYS=(2.0, 4.0, 8.0)`

---

## Models (`src/models/`)

### Enums
| Name | Values |
|---|---|
| `RunStatus` | PENDING, RUNNING, WAITING_FOR_USER, SUCCEEDED, FAILED, CANCELLED |
| `LoopStage` | ORCHESTRATE, CAPTURE, PERCEIVE, UPDATE_STATE, CHOOSE_ACTION, EXECUTE, VERIFY, RECOVER |
| `ActionType` | CLICK, DOUBLE_CLICK, RIGHT_CLICK, TYPE, SELECT, NAVIGATE, HOTKEY, PRESS_KEY, SCROLL, DRAG, HOVER, LAUNCH_APP, WAIT, WAIT_FOR_USER, STOP, BATCH, READ_CLIPBOARD, WRITE_CLIPBOARD, SCREENSHOT_REGION, UPLOAD_FILE, UPLOAD_FILE_NATIVE, READ_TEXT, FILE_PORTER |
| `ExpectedChange` | NONE, FOCUS, CONTENT, NAVIGATION, DIALOG — gates video verifier |
| `VerificationStatus` | SUCCESS, FAILURE, UNCERTAIN, PENDING, PROGRESSING_STABLE, STABLE_WAIT |
| `RecoveryStrategy` | RETRY_SAME_STEP, RETRY_DIFFERENT_TACTIC, WAIT_AND_RETRY, BACKOFF, CONTEXT_RESET, SESSION_RESET, STOP, ADVANCE |
| `PageHint` | FORM_PAGE, FORM_SUCCESS, UNKNOWN — accepts arbitrary snake_case from LLM |
| `UIElementType` | BUTTON, INPUT, LINK, TEXT, DIALOG, ICON, WINDOW, UNKNOWN |

### Key Pydantic models
| Model | File | Key fields |
|---|---|---|
| `AgentState` | state.py | run_id, intent, status, step_count, start_url, last_rule_trace, force_fresh_perception, observation_history |
| `CaptureFrame` | capture.py | artifact_path, width, height, visual_velocity: float, monitor_left/top |
| `ScreenPerception` | perception.py | page_hint, visible_elements: list[UIElement], focused_element_id, ghost_elements: list[GhostElement], is_empty_frame, liveness_retries |
| `UIElement` | perception.py | element_id, element_type, x, y, width, height, is_interactable, primary_name, name_source, is_unlabeled, usable_for_targeting |
| `GhostElement` | perception.py | element_id, element_type, primary_name, x, y, width, height, is_interactable — elements occluded, not gone |
| `PolicyDecision` | policy.py | action: AgentAction, rationale, confidence, active_subgoal, expected_change, rule_name |
| `AgentAction` | policy.py | action_type, x, y, text, key, url, clear_before_typing, press_enter, target_element_id, target_context |
| `ExecutedAction` | execution.py | action, success, detail, artifact_path, execution_trace, failure_category, visual_variance: float |
| `VerificationResult` | verification.py | status, expected_outcome_met, stop_condition_met, failure_type, failure_category, video_verified, video_detail, patience_retries |
| `RecoveryDecision` | recovery.py | strategy, message, retry_after_ms, failure_category, terminal, stop_reason |
| `MemoryRecord` | memory.py | key, benchmark, hint, outcome, page_hint, subgoal, weight (decays on failure) |
| `MemoryHint` | memory.py | key, hint, source, count |
| `StepLog` | logs.py | run_id, step_index, artifact paths, perception/policy/execution/verification/recovery, **decision_source**, **visual_variance** |

All models inherit `StrictModel` (extra="forbid").

---

## Policy Layer

### `PolicyCoordinator` (`src/agent/policy_coordinator.py`)

Wraps `PolicyRuleEngine` + LLM delegate. Call order in `choose_action()`:

1. Benchmark-specific plugins (highest priority)
2. `PolicyRuleEngine` engine primitives
3. Episode replay (if matching trajectory exists)
4. LLM delegate (`GeminiPolicyService` or `AnthropicPolicyService`)
5. **`_semantic_anchor_check`** (post-LLM, highest-priority guard):
   - Fires when LLM coord is > 15 px from every visible element's bounding box
   - Returns WAIT(300ms) + `force_fresh_perception=True` + anchor hint injected into advisory hints for next step
   - Uses `_nearest_element_by_box()` from `policy_rules.py`
6. `_reject_hallucinated_target`
7. `_reject_premature_stop`

`reset_run_context(run_id)` clears: episode/replay state, hint cache, rule trace, `element_buffer`.

### Rule engine primitives (priority order, `src/agent/policy_rules.py`)
| Rule | Trigger |
|---|---|
| `_human_intervention_rule` | page_hint contains HITL keyword × 2 consecutive steps (debounced) |
| `_task_success_stop_rule` | page_hint==FORM_SUCCESS or success tokens visible |
| `_form_visible_field_fill_rule` | form page + visible unfilled field matching intent |
| `_dropdown_menu_select_rule` | dropdown open + unselected option visible |
| `_avoid_identical_type_retry` | memory hint `avoid_identical_type_retry` + repeated TYPE failure |
| `_no_progress_recovery_rule` | no-progress streak > threshold (issues Escape + `force_fresh_perception`) |
| `_dismiss_blocking_overlay_rule` | blocking dialog/banner + stuck signal |
| `_search_query_rule` | intent has search query + search input visible |

Benchmark-specific rules registered via `BENCHMARK_REGISTRY`, run before engine primitives.

### Rule-Augmented Generation
After each step, if `decision.rule_name` is set → `state.last_rule_trace = "[RULE TRACE] Rule 'X' fired at step N | action=Y | outcome=Z"`. Injected into next LLM prompt via `add_advisory_hints(source="rule_trace")`. Cleared when LLM decides.

### Decision source logging
`StepLog.decision_source` is set to `"[RULE] rule_name"` or `"[LLM] gemini"` at step completion. Observer emits this as an event in the live log feed; console.html renders rule events in green and LLM events in blue.

---

## Spatial Persistence Layer

### `RollingElementBuffer` (`src/models/memory.py`)
Rolling 3-frame deque of `list[UIElement]` per frame with ghost-element TTL tracking.

- `push(elements)` — record current frame
- `prev_frame()` — T-1 elements
- `update_ghosts(new_ghosts, current_elements) → list[GhostElement]`
  - Reappeared elements removed immediately
  - Surviving ghosts decremented by 1 each frame
  - TTL=0 → purged (prevents stale clicks on closed windows)
  - New ghosts start at `_GHOST_TTL_FRAMES = 2`
- `clear()` — resets frame deque + active ghost dict

`SpatialCache = RollingElementBuffer` — backwards-compatible alias.

### Coordinate smoothing (`src/agent/perception.py`)
`_smooth_element_coords(current, prev) → (smoothed, snap_count)`

When `visual_velocity < 2%`: for each element in current frame whose matched T-1 element drifted by `0 < Δ < 3 px` on both axes, snap back to T-1 coordinates. Prevents "vibrating" click targets from Gemini sub-pixel variance.

### `_with_spatial_persistence(perception, screenshot) → ScreenPerception`
Called on every successful perception. Sequence:
1. Coordinate smoothing (if stable)
2. Ghost detection (compare smoothed elements vs T-1, build `new_ghosts` list)
3. `element_buffer.update_ghosts(new_ghosts, current_elements)` — applies TTL
4. `element_buffer.push(current_elements)`
5. Return perception with updated `visible_elements` + `ghost_elements`

Buffer cleared via `reset_element_buffer()` when `visual_velocity > 5%` in `loop.py`.

---

## Executor Layer

### DesktopExecutor (`src/executor/desktop.py`)
- pyautogui + mss, full-screen control
- `validate_display_baseline()` → raises `HardwareBaselineError` if resolution < 1280×720
- **Adaptive visual servo**:
  - `_calibrate_servo_threshold()` at init: samples 5 random 100×100px desktop crops to establish Idle Noise Floor
  - Flat desktop (noise < 5 px²) → threshold = 20 px²; normal (5–40 px²) → `noise_floor × 0.75`; noisy (> 40 px²) → capped at 80 px²
  - `_region_has_content(x, y, radius=50, baseline_variance=None)` — uses calibrated threshold unless overridden; logs `region_variance`, `noise_floor`, `threshold` on every check
  - Variance < threshold → `CoordDriftWarning` + abort click with `EXECUTION_TARGET_NOT_FOUND`
  - Returns `(has_content: bool, variance: float)` — variance stored in `ExecutedAction.visual_variance`
- `context_reset()` — Escape×2, click body center, Ctrl+Home
- `session_reset(start_url)` — context_reset + Alt+Tab cycle

### NativeBrowserExecutor (`src/executor/browser_native.py`)
- Playwright-based, video-recorded under `.browser-artifacts/`
- `context_reset()` — clear cookies, reset zoom, focus window
- `session_reset(start_url)` — close tab, open fresh session, navigate to start_url
- `current_url_for_run(run_id)` — live URL for observer

### Executor ABC (`src/executor/browser.py`)
- `capture()` → `CaptureFrame` (3-frame burst in desktop, single in browser)
- `execute(action)` → `ExecutedAction`
- `context_reset()` / `session_reset()` — no-op defaults; overridden above

---

## Perception

`GeminiPerceptionService.perceive(screenshot, state) → ScreenPerception`

**Quality gate**: min 1 interactive element; on FORM_PAGE: min 2 interactive + 1 text. Retry once on quality failure. Salvage pass on final retry. Raises `PerceptionLowQualityError` on unsalvageable output. Returns `is_empty_frame=True` on zero elements (caller retries rather than hard-failing).

After successful perception, `_with_spatial_persistence()` applies coordinate smoothing + ghost detection before returning.

Desktop uses `desktop_perception_prompt.txt`. Browser uses `perception_prompt.txt`. Combined uses `desktop_combined_prompt.txt` / `browser_combined_prompt.txt`.

---

## Verifier

`DeterministicVerifierService.verify(state, decision, executed_action) → VerificationResult`

Accepts optional `video_verifier: VideoVerifier` for reaction checking.

Decision tree (in order):
1. `check_terminal_state()` — success page hint or benchmark tokens → terminal SUCCESS
2. STOP action handling
3. Goal-completing action check (READ_TEXT, HOVER, etc.)
4. Execution failure → FAILURE
5. `_is_page_loading()` — sparse elements + unknown hint + nav action → PENDING
6. `_passive_wait_needs_more_signal()` — WAIT action on inspect-style task → UNCERTAIN
7. **Reaction check** (CLICK/TYPE, `video_verifier` present): `_reaction_verify()` → `PROGRESSING_STABLE` if Gemini confirms visible micro-reaction (confidence ≥ 0.5)
8. **`_screen_is_in_motion()`** — before/after pixel diff > `CURSOR_ONLY_THRESHOLD` → `STABLE_WAIT`
9. `_model_verify()` — Gemini image+text critic
10. Low confidence (< 0.5) → UNCERTAIN
11. Default → SUCCESS

### `VerificationStatus` semantics
| Status | Meaning | Loop action |
|---|---|---|
| `SUCCESS` | Outcome confirmed | Advance / stop |
| `FAILURE` | Action definitively failed | Recovery ladder |
| `UNCERTAIN` | Can't confirm either way | Recovery ladder |
| `PENDING` | Page mid-load (blank/sparse) | Backoff 2–4–8 s, re-verify |
| `PROGRESSING_STABLE` | UI reacted (Gemini-confirmed), no full change | Advance immediately |
| `STABLE_WAIT` | Screen actively animating post-action | Wait 200 ms, re-capture, re-verify once |

### Reaction check (`VideoVerifier.verify_reaction`)
Sends `[before.png, after.png]` as a multi-image request to Gemini via `generate_reaction_check()`. Looks for: click ripple, focus ring, loading indicator, button state change, cursor appearing. Returns `ReactionCheckResult(ui_reacted, reaction_description, confidence)`.

---

## Recovery

`RuleBasedRecoveryManager.recover(state, decision, executed_action, verification) → RecoveryDecision`

Special handling before the ladder:
- `stop_condition_met=True` → STOP (terminal)
- `EXECUTION_NO_PROGRESS` → STOP immediately
- `SUCCESS` → ADVANCE
- **`PROGRESSING_STABLE`** → ADVANCE (UI reacted, keep going)

Escalation ladder (keyed on `state.retry_counts`):
- attempt 1 → RETRY_SAME_STEP
- attempt 2 → RETRY_DIFFERENT_TACTIC
- attempt 3 → CONTEXT_RESET (wait 1000 ms)
- attempt 4 → SESSION_RESET (wait 1500 ms)
- attempt ≥ 5 → STOP (MAX_RECOVERY_ATTEMPTS=5)

`validate_benchmark_integrity()` blocks: unverified success claims and ADVANCE past a stop boundary.

---

## Memory / Store

### `RollingElementBuffer` — see Spatial Persistence Layer above

### `FileBackedMemoryStore` (`src/store/memory.py`)
- `memory/memory.jsonl` — append-only MemoryRecord log
- `memory/episodes.jsonl` — compressed episode trajectories
- `get_hints(benchmark, page_hint, subgoal, recent_failure_category, limit=4)` — weight-ranked; prunes buckets below `_WEIGHT_PRUNE_THRESHOLD = 0.1`
- `record_step()` — writes compact record; on failure, `_decay_active_hints()` halves weight of active hints (geometric decay → convergence to zero)
- Episodes: successful runs only, 1st-attempt steps only (recovered failures excluded)

### Run data layout
```
runs/<run_id>/
  state.json          # AgentState snapshot
  run.jsonl           # StepLog entries (decision_source + visual_variance included)
  reflection.json     # PostRunReflector output
  step_N/
    before.png, after.png
    perception_prompt.txt, perception_raw.txt, perception_parsed.json
    policy_prompt.txt, policy_raw.txt, policy_decision.json
    execution_trace.json, progress_trace.json
    verification_result.json, verification_prompt.txt
    selector_trace.json, perception_diagnostics.json

memory/memory.jsonl, episodes.jsonl
.browser-artifacts/   # Browser session video recordings
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
`/` → landing.html, `/console` → console.html (live log with `[RULE]`/`[LLM]` event colouring), `/dashboard` → dashboard.html, `/benchmarks` → benchmarks.html

### Loop builders
Both `get_agent_loop()` and `get_desktop_agent_loop()` create a `VideoVerifier` instance and pass it to both `AgentLoop` (for `_maybe_video_verify`) and `DeterministicVerifierService` (for reaction checking). Dedicated `video_gemini_client` independent of verifier provider.

---

## Clients

### `GeminiHttpClient` (`src/clients/gemini.py`)
Default model: `gemini-2.5-flash` (env: `GEMINI_MODEL`). Methods:
- `generate_perception(prompt, screenshot_path)` — single image
- `generate_policy(prompt)` — text only
- `generate_video_verification(prompt, video_path)` — MP4 video
- `generate_reaction_check(prompt, frame_paths: list[str])` — multi-image sequence for UI reaction detection
- `generate_verification(prompt, screenshot_path)` — image+text critic (via AnthropicHttpClient when using Anthropic verifier)

### `AnthropicHttpClient` (`src/clients/anthropic.py`)
Default model: `claude-sonnet-4-20250514`. Methods: `generate_policy()` (text), `generate_verification()` (image+text).

---

## Runtime / Unified Contracts (`src/runtime/`)

- `UnifiedOrchestrator.process_step(perception, planner, actor, critic, current_state) → StepState`
- `AgentRuntimeState` — shared mutable state per run (subgoal, url, retry context, advisory hints)
- `LegacyOperonContractAdapter` — bridges `AgentLoop` Phase 1 API to Phase 2 contracts

This layer is an **observer and validator**, not a replacement for `AgentLoop`. The loop still owns execution ordering and termination.

---

## Environment Variables

### Desktop
`OPERON_DESKTOP_BACKEND` (json), `OPERON_DESKTOP_MODEL`, `OPERON_DESKTOP_PLANNER_PROVIDER` (gemini|anthropic), `OPERON_DESKTOP_PLANNER_MODEL`, `OPERON_DESKTOP_VERIFIER_PROVIDER`, `OPERON_DESKTOP_VERIFIER_MODEL`, `OPERON_DESKTOP_FALLBACK_MODEL`

### Browser
`OPERON_BROWSER_BACKEND` (computer_use), `OPERON_BROWSER_MODEL`, `OPERON_BROWSER_PLANNER_PROVIDER`, `OPERON_BROWSER_PLANNER_MODEL`, `OPERON_BROWSER_VERIFIER_PROVIDER`, `OPERON_BROWSER_VERIFIER_MODEL`, `OPERON_BROWSER_FALLBACK_BACKEND` (json), `OPERON_BROWSER_FALLBACK_MODEL`

### Global
`GEMINI_API_KEY` / `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPERON_USE_VERTEX`, `VERTEX_PROJECT`, `VERTEX_LOCATION`, `OPERON_TRACE` (1=enable trace events), `OPERON_TEST_SAFE_MODE` (skip display baseline + servo calibration), `BROWSER_WIDTH` (1920), `BROWSER_HEIGHT` (1080), `BROWSER_HEADLESS`, `CORS_ORIGINS`, `FORM_BENCHMARK_URL`

---

## Tests

Key files: `test_agent_loop.py`, `test_policy.py`, `test_policy_coordinator.py`, `test_verifier.py`, `test_recovery.py`, `test_perception.py`, `test_memory.py`, `test_desktop_executor.py`, `test_browser_native.py`, `test_video_verification.py`, `test_desktop_routes.py`, `test_schema_validation.py`, `test_episode.py`, `test_reflector.py`, `test_gemini_client.py`, `test_action_translation.py`, `test_selector.py`, `test_run_store_and_logging.py`, `test_failure_taxonomy.py`, `test_hitl.py`, `test_webarena.py`, `test_browser_computer_use_smoke.py`

Excluded from default run: `test_e2e_quick_tasks.py`, `test_bug_fixes_verification.py`, `test_live_execution.py`, `test_browserbase_integration.py`, `test_file_porter_integration.py`, `test_upload_file_native_integration.py` (require live server / hardware / env).

Safe default command:
```powershell
$env:GEMINI_API_KEY = "fake-test-key"
.venv\Scripts\python -m pytest tests\ -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py
```

---

## Invariants (never bypass)

1. **Atomic TYPE**: executor merges click+type into one call. Never emit CLICK then TYPE on the same element from policy.
2. **Visual Servo**: `_region_has_content()` runs before every click. Variance < calibrated threshold → abort with `CoordDriftWarning`. Returns `(bool, float)` — variance stored in `ExecutedAction.visual_variance`.
3. **expected_change required**: Gemini policy output must include `expected_change` field. Missing → `PolicyError` (loud fail).
4. **No DOM/selectors**: vision-only. No XPath, CSS, accessibility tree, Playwright `locator()` in policy/perception path.
5. **Rules before LLM**: if a condition is expressible as a Python predicate, it belongs in `PolicyRuleEngine`.
6. **Post-LLM anchor guard**: `PolicyCoordinator._semantic_anchor_check` runs after every LLM decision. Never remove or bypass.
7. **Buffer cleared on high velocity**: `element_buffer.clear()` when `visual_velocity > 5%`. Prevents ghost coordinates from prior app context.
