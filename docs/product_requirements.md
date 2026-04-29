# Operon — Product Requirements Document

**Vision:** Vision-only computer use agent. No DOM, no selectors. Sees the screen like a human and acts on it.

**Loop:** `capture → perceive → decide → execute → verify → recover`

Desktop (pyautogui + mss) and browser (Playwright + Gemini Computer Use) share one loop, one verifier, one recovery manager, one persistence layer.

---

## 1. Core Control Loop

**What:** Closed-loop autonomous agent. Each step is discrete and API-driven — the caller advances the loop or it runs autonomously.

**Stages:**

| Stage | Responsibility |
|---|---|
| Capture | Screenshot the current screen |
| Perceive | Send screenshot to Gemini → typed `ScreenPerception` |
| Update state | Merge perception into `AgentState` |
| Choose action | Rule engine first, LLM fallback |
| Execute | Perform action; one bounded retry on stale/shifted target |
| Verify | Deterministic check + optional video verify |
| Recover | Decide continue / retry / stop based on failure category |
| Reflect (terminal) | Extract failure patterns → write `MemoryRecord` |

**Tech:** `src/agent/loop.py` — `AgentLoop` singleton per mode. Steps via `POST /step` or `POST /desktop/step`.

---

## 2. Dual Execution Modes

| | Desktop | Browser |
|---|---|---|
| Executor | `DesktopExecutor` (pyautogui) | `NativeBrowserExecutor` (Playwright Chromium) |
| Capture | `mss` full-screen | `page.screenshot()` |
| Primary backend | `CombinedPerceptionPolicyService` (Gemini JSON) | `BrowserComputerUseBackend` (Gemini Computer Use) |
| Fallback backend | — | `BrowserJsonBackend` |
| Video recording | — | `.browser-artifacts/<run_id>/session_video/session.webm` |

Backend selection via env vars: `OPERON_DESKTOP_BACKEND`, `OPERON_BROWSER_BACKEND`, `OPERON_BROWSER_FALLBACK_BACKEND`.

---

## 3. Action Types (21 total)

| Category | Actions |
|---|---|
| Pointer | `click`, `double_click`, `right_click`, `hover`, `drag`, `scroll` |
| Keyboard | `type`, `select`, `press_key`, `hotkey` |
| Navigation | `navigate`, `launch_app` |
| Flow control | `wait`, `wait_for_user`, `stop`, `batch` |
| Clipboard | `read_clipboard`, `write_clipboard` |
| Vision | `screenshot_region` |
| File upload | `upload_file`, `upload_file_native` |

**upload_file** — Playwright `expect_file_chooser` interception. Headless-safe.

**upload_file_native** — Deterministic OS picker macro (`src/executor/os_picker_macro.py`). After clicking the upload trigger: polls for native picker window by title keyword, types the absolute path via `pyautogui.write`, presses Enter, polls for close. Returns `PickerOutcome`: `SUCCESS`, `PICKER_NOT_DETECTED`, `FILE_NOT_REFLECTED`, `UNAVAILABLE`. Headed mode only — headless guard returns `EXECUTION_ERROR` immediately.

---

## 4. Perception

**What:** Vision-only. Screenshot → Gemini → typed `ScreenPerception` (visible elements, `PageHint`, focused element). No DOM, no XPath, no CSS selectors from the model.

**Tech:**
- `GeminiPerceptionService` — `src/agent/perception.py`
- `GeminiHttpClient` — `src/clients/gemini.py`
- Prompts: `prompts/perception_prompt.txt`, `prompts/desktop_perception_prompt.txt`

---

## 5. Policy Layer (3-tier)

### Tier 1 — Deterministic Rule Engine
`PolicyRuleEngine` (`src/agent/policy_rules.py`). 6 rules in priority order:

| Rule | Type | Trigger |
|---|---|---|
| `_login_page_guardrail` | Benchmark-specific | `page_hint == GOOGLE_SIGN_IN` + memory hint |
| `_form_success_stop_rule` | Engine primitive | `page_hint == FORM_SUCCESS` or success keywords in elements |
| `_avoid_identical_type_retry` | Engine primitive | Memory hint + repeated TYPE failure on same element |
| `_compose_already_visible_rule` | Benchmark-specific | Subgoal contains "compose" + compose input visible |
| `_submit_form_when_ready_rule` | Benchmark-specific | `page_hint == FORM_PAGE` + all required fields typed |
| `_focus_before_type_rule` | Engine primitive | Memory hint `click_before_type` + target not focused |

### Tier 2 — LLM Fallback
`GeminiPolicyService` (`src/agent/policy.py`). Renders prompt template + memory hints → Gemini. Prompts: `prompts/policy_prompt.txt`, `prompts/browser_combined_prompt.txt`.

### Tier 3 — Anthropic Planner (optional)
`AnthropicPolicyService` (`src/agent/anthropic_policy.py`). Claude-backed planning. Enabled via `OPERON_DESKTOP_PLANNER_PROVIDER=anthropic`.

**Coordinator:** `PolicyCoordinator` (`src/agent/policy_coordinator.py`) wraps all three tiers. Memory hints from `FileBackedMemoryStore` are injected into both rule engine and LLM prompt every step.

---

## 6. Target Resolution & Re-resolution

**What:** Targetable actions carry a serializable `TargetIntent` + `target_context` (original target signature, top candidate evidence, matched signals). On stale/shifted/lost target, the executor captures fresh perception and re-resolves deterministically — no LLM call.

**Tech:** `DeterministicTargetSelector.reresolve()` — `src/agent/selector.py`. Failures emit `target_reresolution_failed` or `target_reresolution_ambiguous`. Full trace (trigger reason, candidates considered, outcome) written to `execution_trace.json` as `reresolution_trace`.

---

## 7. Verification

### Deterministic Verifier
`DeterministicVerifierService` (`src/agent/verifier.py`). Checks expected outcome against new perception after every action.

### Video Verifier
Triggers when `screen_diff` detects no visual change after an idempotent action (`click`, `press_key`, `hotkey`, `launch_app`, `scroll`, `hover`). Records a 3-second clip via `ScreenRecorder`, re-executes the action, sends clip to Gemini for temporal analysis. Adds ~5–8s per uncertain step. Prompt: `prompts/video_verification_prompt.txt`.

### Critic (in progress)
`prompts/critic_prompt.txt` — critic-backed step evaluation. Selectable via `OPERON_DESKTOP_VERIFIER_PROVIDER`.

---

## 8. Self-Improving Memory

**What:** After terminal steps, `PostRunReflector` analyzes the run, extracts failure patterns, writes `MemoryRecord` entries per benchmark. On next run, hints are injected into both the rule engine and the LLM prompt.

**Tech:**
- `src/agent/reflector.py` — `PostRunReflector`
- `src/store/memory.py` — `FileBackedMemoryStore`
- `src/models/memory.py` — `MemoryRecord`, `MemoryHint`

---

## 9. Episodic Memory

**What:** Successful runs are compressed into reusable `Episode` objects — a trajectory cache of what worked, available as advisory context on future runs.

**Tech:** `src/models/episode.py`. Written by `PostRunReflector` on `SUCCEEDED` runs.

---

## 10. Redundancy Blocking

**What:** Prevents the agent from repeating the same action without screen progress.

**Tech:** `ProgressState` counters inside `AgentState`. `AgentLoop._block_redundant_action()` compares current action + screen diff against recent history. Emits `REPEATED_ACTION_WITHOUT_PROGRESS`, `REPEATED_TARGET_WITHOUT_PROGRESS`, or `REPEATED_FAILURE_LOOP`.

---

## 11. Recovery Manager

**What:** After a failed step, decides: continue / retry / stop.

**Tech:** `RuleBasedRecoveryManager` (`src/agent/recovery.py`). Maps `FailureCategory` → recovery strategy.

**Failure categories (40+):** Full enumeration in `src/models/common.py`. Includes perception failures, selector failures, target re-resolution failures, execution failures, progress failures, file upload failures (`PICKER_NOT_DETECTED`, `FILE_NOT_REFLECTED`), and terminal limits.

---

## 12. Unified Contract Layer

**What:** Every step wrapped in typed perception/planner/actor/critic bundles — clean separation of concerns between observation, planning, execution, and critique.

**Contracts:** `PerceptionOutput`, `PlannerOutput`, `ActorOutput`, `CriticOutput` in `src/core/contracts/`.

**Router:** `BROWSER_ACTIONS` / `DESKTOP_ACTIONS` sets. `is_cross_environment_action()` flags actions (like `upload_file_native`) that cross environment boundaries. `validate_plan_route()` enforces this at plan time.

**Orchestrator:** `UnifiedOrchestrator` — adaptation strategies per failure type (e.g. `PICKER_NOT_DETECTED` → `wait_then_retry`, `FILE_NOT_REFLECTED` → `reperceive_and_replan`).

---

## 13. Persistence & Observability

### Run Store
`FileBackedRunStore` — in-memory dict + `runs/<run_id>/state.json`. No database.

### Per-step artifacts
```
runs/<run_id>/
  state.json
  run.jsonl                   # StepLog per step (JSONL)
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
```

### Browser video
`ScreenRecorder` → `.browser-artifacts/<run_id>/session_video/session.webm`

### CLI tools
```
python -m src.store.replay <run_id>
python -m src.store.summary <run_id>
python -m src.store.summary runs
```

---

## 14. API

FastAPI app (`src/api/server.py`). All routes in `src/api/routes.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/run-task` | Create browser run |
| POST | `/step` | Advance browser run one step |
| POST | `/resume` | Resume paused browser run |
| GET | `/run/{id}` | Read browser run state |
| POST | `/desktop/run-task` | Create desktop run |
| POST | `/desktop/step` | Advance desktop run one step |
| POST | `/desktop/resume` | Resume paused desktop run |
| GET | `/desktop/run/{id}` | Read desktop run state |
| POST | `/desktop/cleanup` | Close apps launched by a run |
| GET | `/observer/api/runs` | List all runs |
| GET | `/observer/api/run/{id}` | Full run snapshot |
| GET | `/observer/api/artifact` | Serve step artifact (png, json, txt) |
| GET | `/health` | Health check |
| GET | `/` | Operon Pilot UI |

Input validation: `intent` 1–500 chars, stripped whitespace, strict Pydantic v2 models — no extra fields.

---

## 15. Benchmarks

| Benchmark | Entry point | What it measures |
|---|---|---|
| Form benchmark | `python -m src.agent.benchmark` | End-to-end web form fill success rate |
| Native upload benchmark | `src/evaluation/benchmark_native_upload.py` | `upload_file_native` reliability in headed mode |

---

## 16. Model Configuration

All model choices are env-configurable at runtime:

| Role | Default model | Env var |
|---|---|---|
| Desktop perception + policy | `gemini-3-flash-preview` | `OPERON_DESKTOP_MODEL` |
| Browser primary | `gemini-2.5-computer-use-preview-10-2025` | `OPERON_BROWSER_MODEL` |
| Browser fallback | `gemini-3-flash-preview` | `OPERON_BROWSER_FALLBACK_MODEL` |
| Desktop verifier | `gemini-3-flash-preview` | `OPERON_DESKTOP_VERIFIER_MODEL` |
| Browser verifier | `gemini-3-flash-preview` | `OPERON_BROWSER_VERIFIER_MODEL` |
| Planner (optional) | Claude (Anthropic) | `OPERON_DESKTOP_PLANNER_PROVIDER=anthropic` |

---

## 17. Known Gaps & Hardening Opportunities

- **Rule engine domain bleed** — 3 of 6 rules are Gmail/form benchmark-specific. A third benchmark is the forcing function to split into a base engine rule set + per-benchmark rule registrations.
- **`PageHint` enum** — 4 of 6 values are Gmail-specific. Should generalise.
- **`upload_file_native` integration test** — no test against a real headed browser with a live OS picker. CI-only coverage is unit-level.
- **Critic integration** — `critic_prompt.txt` exists but critic is not fully wired into the recovery loop.
- **Episodic memory retrieval** — episodes are written but retrieval/injection into prompt is not yet implemented.
- **Single benchmark default** — `benchmark_name_for_intent()` defaults unknown intents to `FORM_BENCHMARK`.
