<div align="center">

<img src="assets/operon-loop-mark.svg" width="72" alt="Operon logo" />

# Operon

**A zero-abstraction, vision-first computer-use agent.**  
No DOM. No selectors. No accessibility tree. Just pixels.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Lint](https://github.com/vickey-kapoor/Operon/actions/workflows/ci.yml/badge.svg)](https://github.com/vickey-kapoor/Operon/actions)
[![Status: Active](https://img.shields.io/badge/status-active%20development-orange)](https://github.com/vickey-kapoor/Operon/pulls)

<img src="Mockup.png" width="860" alt="Operon Command Center UI" />

</div>

---

## What is Operon?

Operon is a clinical, execution-only computer-use engine. It perceives the world as raw pixels and operates exclusively in coordinate space — the same way a human does. Every run is a direct march toward a visually confirmed terminal state.

It works on **any surface**: web apps, desktop applications, internal tools, legacy systems. If a human can see it, Operon can use it.

```
capture → perceive → decide → execute → verify → recover
```

---

## Why Vision-Only?

| Traditional Automation | Operon |
|---|---|
| Anchors to CSS selectors, XPath, element IDs | Anchors to pixel coordinates |
| Breaks on UI redesigns or framework migrations | Works on any rendered surface |
| Requires DOM access — web only | Works on desktop, browser, legacy apps |
| Brittle to dynamic class names | Stable to visual layout |

---

## Key Capabilities

- **Pure vision perception** — Gemini perceives the screen and returns typed `ScreenPerception` objects (coordinates, page hints, element types). Zero DOM interaction.
- **Rule-Augmented Generation** — `PolicyRuleEngine` runs deterministic checks before every LLM call. Fired rule names are stamped and injected into the next prompt as a RAG trace.
- **Confidence-gated autonomy** — Set an autonomy threshold in the UI. When `PolicyDecision.confidence` drops below it, the agent pauses and surfaces the decision for human approval or correction.
- **Spatial persistence** — `RollingElementBuffer` (3-frame rolling cache) tracks coordinates across steps. Elements absent for >2 frames become `GhostElement`s — occluded, not gone.
- **Visual servo** — Before every click, a 100×100px crop is variance-checked. Uniform regions (target shifted or disappeared) are rejected and logged as `CoordDriftWarning`. Never fires blind.
- **Adaptive stall detection** — Screen-change ratio measured after each action. <1% change over 2+ consecutive steps triggers a subgoal reset and forces a new strategy.
- **Reaction-check verification** — `VideoVerifier` sends before/after frames to Gemini to detect micro-reactions (ripple, focus ring, spinner). Returns `PROGRESSING_STABLE` → advance immediately.
- **Observable mode** — Browser runs stream live JPEG frames (CDP `Page.startScreencast`) to the Command Center via WebSocket. `BrowserManager` (`src/browser/manager.py`) attaches to Chrome on port 9222, streams at ~15fps, and exposes interactive input injection.
- **Self-improving memory** — `PostRunReflector` writes `MemoryHint`s on every run. Successful trajectories are compressed into reusable `Episode`s. Hints decay geometrically on failure.
- **Human-in-the-loop** — CAPTCHA, login, 2FA, or bot-detection triggers a UI pause with a Windows/macOS/Linux notification, a live screenshot overlay, and a correction hint input.

---

## Architecture

```
FastAPI  ──▶  AgentLoop
                ├── Capture    mss / Playwright burst (3 frames, velocity-checked)
                ├── Perceive   Gemini → ScreenPerception + coord smoothing + ghost detection
                ├── Decide     PolicyCoordinator
                │               ├── PolicyRuleEngine   (8 deterministic rules, pre-LLM)
                │               ├── Episode replay     (cached optimal trajectories)
                │               ├── GeminiPolicyService / AnthropicPolicyService (LLM fallback)
                │               └── _semantic_anchor_check (post-LLM, 15px guard)
                ├── Execute    DesktopExecutor (adaptive servo) │ NativeBrowserExecutor
                ├── Verify     DeterministicVerifier
                │               ├── Terminal state check (visual predicate, fires first)
                │               ├── Reaction check  (VideoVerifier, multi-image Gemini)
                │               ├── STABLE_WAIT     (200ms re-verify on UI motion)
                │               └── PENDING         (page loading, 2–8s backoff)
                ├── Recover    RuleBasedRecoveryManager (5-rung escalation ladder)
                ├── Reflect    PostRunReflector  (terminal only)
                └── Persist    FileBackedRunStore + RollingElementBuffer + MemoryStore

Observable mode (browser):
    NativeBrowserExecutor  →  launches Chromium + --remote-debugging-port=9222
    BrowserManager         →  connect_over_cdp → CDP Page.startScreencast → JPEG frames
    ws_stream              →  broadcasts binary frames to all WebSocket clients (port 9001)

SSE  /command-center/api/run/{id}/stream ──▶ Command Center (live step log)
WebSocket (port 9001)                    ──▶ JPEG frames + step events + control messages
```

---

## Quick Start

**Requirements:** Python 3.11, Windows/macOS/Linux, Chrome (for browser mode)

```powershell
# 1. Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -e .[dev]
playwright install chromium

# 3. Configure API key
copy .env.example .env
# Set GOOGLE_API_KEY or GEMINI_API_KEY in .env

# 4. Start the server
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Open **http://localhost:8080** — redirects automatically to the split-pane Command Center.

---

## Command Center UI

Open **http://localhost:8080** — the Pilot UI (three-pane layout, React 19 + Zustand 5).

```
┌─────────────────────┬──────────────────────────────┬─────────────────────┐
│  TASK INTELLIGENCE  │  LIVE EXECUTION              │  SETTINGS           │
│  35%                │  50%                         │  15%                │
│                     │                              │                     │
│  ● Subgoal 1 ✓      │  ┌──────────────────────┐   │  Rule Manager       │
│  ◌ Subgoal 2 …      │  │  JPEG frame stream   │   │  toggle rules on/off│
│  ○ Subgoal 3        │  │  (CDP screencast)    │   │                     │
│                     │  │                      │   │  Session Mode       │
│  Thought Cards:     │  └──────────────────────┘   │  Fresh / Observable │
│  ├ step 3  0.92     │                              │                     │
│  │ clicked Submit   │  SVG element bounds overlay  │                     │
│  └ rule fired: …    │  ConfidenceSlider            │                     │
└─────────────────────┴──────────────────────────────┴─────────────────────┘
```

| Pane | What's in it |
|---|---|
| **Task Intelligence** (35%) | SubgoalTree (status icons per subgoal), Thought Cards reasoning log (timestamp, perception, confidence badge, rationale), ThinkingPulse SVG animation, HITL approval card |
| **Live Execution** (50%) | Canvas JPEG frame mirror (zero React re-render delivery via `registerFrameCallback`), SVG element bounds overlay, ConfidenceSlider (sends `set_confidence_threshold` over WS), HITL dim overlay |
| **Settings / Moat Builder** (15%) | Rule toggles (`set_disabled_rules`), session persistence mode (Fresh / Observable) |

In **Observable mode**, the Live Execution pane receives binary JPEG frames from the WebSocket (port 9001) streamed via CDP `Page.startScreencast` at ~15fps. Interactive clicks and keystrokes typed in the pane are sent back as `inject_input` control messages.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | — | Gemini API (required) |
| `ANTHROPIC_API_KEY` | — | Claude planner / verifier (optional) |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use` or `json` |
| `OPERON_DESKTOP_BACKEND` | `json` | Desktop perception + policy mode |
| `OPERON_BROWSER_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_DESKTOP_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `BROWSER_HEADLESS` | `false` | Headless Playwright |
| `BROWSER_WIDTH` | `1920` | Playwright viewport width |
| `BROWSER_HEIGHT` | `1080` | Playwright viewport height |
| `OPERON_TRACE` | — | `1` enables `[TRACE]` loop events |
| `OPERON_TEST_SAFE_MODE` | `false` | Skip display baseline + servo calibration in tests |

---

## API

```
# Browser run lifecycle
POST /run-task                                    Start a browser run
POST /step                                        Advance one step (browser)
POST /resume                                      Resume a HITL-paused run
POST /stop                                        Cancel active run (body: {run_id})
POST /run/{id}/stop                               Cancel by path
POST /run/{id}/pause                              Pause by path (→ WAITING_FOR_USER)
GET  /run/{id}                                    Read run state
POST /cleanup                                     Close Playwright resources
GET  /health                                      Health check

# Desktop run lifecycle
POST /desktop/run-task                            Start a desktop run
POST /desktop/step                                Advance one step (desktop)
POST /desktop/resume                              Resume paused desktop run
POST /desktop/cleanup                             Close apps launched by run
GET  /desktop/run/{id}                            Read desktop run state

# Command Center UI
GET  /                                            Pilot UI (React 19 three-pane)
GET  /command-center                              Command Center (HTML+SSE)
GET  /command-center/{run_id}                     Command Center with run pre-selected
GET  /command-center/api/run/{id}/stream          SSE step stream (tails run.jsonl)
GET  /command-center/api/run/{id}/screenshot      Latest step screenshot (polling)

# Observer / telemetry
GET  /observer/api/runs                           Recent runs list
GET  /observer/api/run/{id}                       Full run snapshot + artifacts
GET  /observer/api/artifact                       Serve step artifact files
GET  /observer/api/usage                          Token usage summary
GET  /observer/api/export/{id}                    Export run as ZIP
GET  /observer/api/live-browser/{id}              Live browser URL for run
GET  /console                                     Task Console (live log)
GET  /dashboard                                   MTD metrics dashboard

# Benchmarks
POST /benchmark/run-suite                         Start benchmark suite
POST /benchmark/run-task                          Run single task {task_id, max_steps, mode}
POST /benchmark/stop-suite/{id}                   Stop running suite
GET  /benchmark/tasks                             List available tasks
GET  /benchmark/suite/{id}                        Read suite status + per-task results
GET  /benchmarks                                  Benchmark runner UI
```

---

## Testing

```powershell
# Unit + integration tests (no live server required)
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests -q `
  --ignore=tests/test_e2e_quick_tasks.py `
  --ignore=tests/test_bug_fixes_verification.py

# Single file
python -m pytest tests/test_agent_loop.py -q

# Lint
ruff check src tests --select E,F,W,I --ignore E501
```

---

## Project Layout

```
src/
  agent/      loop.py, perception.py, policy_coordinator.py, policy_rules.py,
              policy.py, verifier.py, video_verifier.py, recovery.py,
              reflector.py, selector.py, capture.py, hitl.py,
              screen_diff.py, action_translation.py, backend.py,
              browser_computer_use.py, browser_json.py, combined.py,
              benchmark.py, screen_recorder.py
  browser/    manager.py  (BrowserManager: CDP attach + screencast + input injection)
  executor/   desktop.py, browser_native.py, browser_adapter.py,
              desktop_adapter.py, os_picker_macro.py
  api/        server.py, routes.py, observer.py, ws_stream.py, runtime_config.py,
              benchmark_suite.py, static/ (command-center, console, dashboard, benchmarks)
  clients/    gemini.py, anthropic.py, gemini_computer_use.py
  models/     state.py, perception.py, policy.py, execution.py,
              verification.py, recovery.py, memory.py, logs.py, common.py
  store/      run_store.py, memory.py, run_logger.py, background_writer.py,
              replay.py, summary.py, cleanup.py
  runtime/    orchestrator.py, state.py, legacy_adapter.py, benchmark_runner.py
  benchmarks/ registry.py  (BENCHMARK_REGISTRY for plugin rules)
  core/       contracts/, router.py
ui/           React 19 + Zustand 5 + react-resizable-panels (Vite, inline styles)
prompts/      policy_prompt.txt, perception_prompt.txt, critic_prompt.txt,
              browser_combined_prompt.txt, video_verification_prompt.txt,
              reaction_check_prompt.txt, desktop_combined_prompt.txt, …
runs/<run_id>/   state.json, run.jsonl, step_N/ (screenshots + all model I/O)
memory/          memory.jsonl, episodes.jsonl
.browser-artifacts/  Browser session video recordings (.webm)
```

---

## Benchmarks

| Suite | Entry point | Status |
|---|---|---|
| WebArena easy (13 tasks) | `/benchmarks` UI or `POST /benchmark/run-suite` | 7/13 at 100% on easy tasks |
| Form benchmark | `python -m src.agent.benchmark` | Live form-fill end-to-end |
| Native upload benchmark | `src/evaluation/benchmark_native_upload.py` | Headed Windows |

---

## Roadmap

- [ ] Claude Computer Use backend for browser perception
- [ ] WebArena medium/hard benchmarks at scale
- [ ] Prompt caching (Vertex AI context cache for static prefixes)
- [ ] Tauri desktop app packaging (`src-tauri/`)
- [ ] Multi-monitor support (currently primary display only)

---

<div align="center">

Built with [Gemini](https://deepmind.google/technologies/gemini/) · [FastAPI](https://fastapi.tiangolo.com/) · [Playwright](https://playwright.dev/) · [React 19](https://react.dev/)

</div>
