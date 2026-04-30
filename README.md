# Operon

**Operon is a vision-only computer use agent — no DOM, no selectors. It sees the screen like a human does and acts on it.**

Closed loop: `capture → perceive → decide → execute → verify → recover`. Desktop (pyautogui) and browser (Playwright + Gemini Computer Use) share one loop.

> 🚧 Early stage — actively building in public. Failures included.

---

## Why Vision-Only?

Most automation frameworks anchor to the DOM — CSS selectors, XPath, element IDs. Operon deliberately doesn't.

- **Selectors are brittle.** A class name change, a UI redesign, a framework migration — your automation breaks.
- **The DOM is a human construct. AI shouldn't need it.** Humans look at interfaces; agents that generalise should too.
- **Vision-only works everywhere.** Web apps, desktop apps, internal tools, legacy systems. If a human can see it, Operon can use it.

---

## HLD

```text
FastAPI  →  AgentLoop  →  Capture   (mss / Playwright screenshot + burst velocity)
                       →  Perceive  (Gemini → ScreenPerception + coord smoothing + ghost detection)
                       →  Decide    PolicyCoordinator
                                      ├── PolicyRuleEngine  (deterministic, pre-LLM)
                                      ├── GeminiPolicyService (LLM fallback)
                                      └── _semantic_anchor_check (post-LLM, 15px guard)
                       →  Execute   DesktopExecutor (adaptive servo) | NativeBrowserExecutor
                       →  Verify    DeterministicVerifier
                                      ├── reaction check  (VideoVerifier, multi-image)
                                      ├── STABLE_WAIT     (200ms re-verify on UI motion)
                                      ├── PROGRESSING_STABLE (Gemini-confirmed reaction → advance)
                                      └── PENDING         (page loading, 2–8s backoff)
                       →  Recover   RuleBasedRecoveryManager
                       →  Reflect   PostRunReflector  (on terminal)
                       →  Persist   FileBackedRunStore + RollingElementBuffer + MemoryStore + Episodes

Unified Contract Layer  (observer on top of the loop)
    LegacyOperonContractAdapter → PerceptionOutput / PlannerOutput / ActorOutput / CriticOutput
    UnifiedOrchestrator  →  src/core/router.py  →  AgentRuntimeState
```

**What's inside:**

- **Vision only** — no DOM / a11y tree. Coordinates come from Gemini perception, not HTML.
- **Spatial persistence** — `RollingElementBuffer` (3-frame rolling cache) tracks coordinates across steps. Elements absent for >2 frames but on a stable screen become `GhostElement`s (occluded, not gone). Buffer cleared on `visual_velocity > 5%` to prevent cross-app contamination.
- **Coordinate smoothing** — sub-pixel jitter (< 3 px) snapped back to previous-frame values when velocity is low, keeping targets stable.
- **Adaptive servo threshold** — `DesktopExecutor` samples 5 random desktop crops at startup to establish an Idle Noise Floor and scales the visual-servo variance threshold accordingly (flat desktops keep 20 px²; textured ones scale up).
- **Semantic anchor check** — post-LLM guard in `PolicyCoordinator`. If a coordinate is > 15 px from every visible element's bounding box, the action is intercepted and the agent re-perceives with an anchor hint.
- **Reaction-check verification** — for CLICK/TYPE actions, `VideoVerifier.verify_reaction()` sends before+after frames to Gemini to detect micro-reactions (ripple, focus ring, loading spinner). Returns `PROGRESSING_STABLE` → advance immediately.
- **STABLE_WAIT** — deterministic screen-motion check (before→after pixel diff). If UI is actively animating post-action, wait 200 ms, re-capture, re-verify once before settling.
- **Rules first, LLM fallback** — `PolicyRuleEngine` → `GeminiPolicyService`. Rules are named and stamped on `PolicyDecision.rule_name`; the loop builds a Rule-Augmented Generation trace for the next LLM prompt.
- **Hardened targets** — `DeterministicTargetSelector.reresolve()` re-binds to original `TargetIntent` if the target moves.
- **Step-level logging granularity** — `StepLog` records `decision_source` (`[RULE] rule_name` or `[LLM] gemini`) and `visual_variance` from the servo check. Observer live-feed colours them green vs blue.
- **Self-improving** — `PostRunReflector` writes `MemoryHint`s; successful runs are compressed into reusable `Episode`s. Hints decay geometrically on failure and are pruned below weight 0.1.

---

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Copy `.env.example` → `.env` and set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).

## Run

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

- `/` — Pilot UI (desktop + browser)
- `/console` — Task Console (live log with `[RULE]`/`[LLM]` colouring)
- `/dashboard` — MTD metrics
- `/benchmarks` — Benchmark suite runner
- `POST /run-task`, `/step`, `/resume`, `/stop`, `/cleanup`
- `GET /run/{id}`, `/observer/api/runs`, `/observer/api/run/{id}`

## Key Env Vars

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | — | Gemini API access |
| `ANTHROPIC_API_KEY` | — | Claude planner/verifier (optional) |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use` or `json` |
| `OPERON_DESKTOP_BACKEND` | `json` | Desktop perception+policy mode |
| `OPERON_BROWSER_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_DESKTOP_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `BROWSER_HEADLESS` | `false` | Headless Playwright |
| `OPERON_TRACE` | — | `1` enables cyan `[TRACE]` loop events |
| `OPERON_TEST_SAFE_MODE` | `false` | Skip display baseline + servo calibration in tests |

## Tests

```powershell
# Safe default (no live server required)
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py

# Lint
ruff check src tests --select E,F,W,I --ignore E501
```

## Layout

```
src/
  agent/      loop.py, perception.py, policy_coordinator.py, policy_rules.py,
              policy.py, verifier.py, video_verifier.py, recovery.py,
              reflector.py, selector.py, capture.py, hitl.py,
              screen_diff.py, action_translation.py, backend.py
  executor/   desktop.py, browser_native.py, browser_adapter.py,
              desktop_adapter.py, os_picker_macro.py
  api/        server.py, routes.py, observer.py, runtime_config.py,
              static/ (landing, console, dashboard, benchmarks)
  clients/    gemini.py, anthropic.py, gemini_computer_use.py
  models/     state.py, perception.py, policy.py, execution.py,
              verification.py, recovery.py, memory.py, logs.py, common.py
  store/      run_store.py, memory.py, run_logger.py, background_writer.py
  runtime/    orchestrator.py, state.py, legacy_adapter.py
  core/       contracts/, router.py
prompts/      policy_prompt.txt, perception_prompt.txt, critic_prompt.txt,
              video_verification_prompt.txt, reaction_check_prompt.txt, …
runs/<run_id>/   state.json, run.jsonl, step_N/ (screenshots + all model I/O)
memory/          memory.jsonl, episodes.jsonl
```

See `docs/architecture.md` and `CLAUDE.md` for full detail.
