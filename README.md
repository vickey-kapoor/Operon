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
- **Self-improving memory** — `PostRunReflector` writes `MemoryHint`s on every run. Successful trajectories are compressed into reusable `Episode`s. Hints decay geometrically on failure.
- **Human-in-the-loop** — CAPTCHA, login, 2FA, or bot-detection triggers a UI pause with a Windows/macOS/Linux notification, a live screenshot overlay, and a correction hint input.

---

## Architecture

```
FastAPI  ──▶  AgentLoop
                ├── Capture    mss / Playwright burst (3 frames, velocity-checked)
                ├── Perceive   Gemini → ScreenPerception + coord smoothing + ghost detection
                ├── Decide     PolicyCoordinator
                │               ├── PolicyRuleEngine   (deterministic, named rules)
                │               ├── GeminiPolicyService (LLM fallback)
                │               └── SemanticAnchorCheck (15px post-LLM guard)
                ├── Execute    DesktopExecutor (adaptive servo) │ NativeBrowserExecutor
                ├── Verify     DeterministicVerifier
                │               ├── Terminal state check (visual predicate, fires first)
                │               ├── Reaction check  (VideoVerifier, multi-image)
                │               ├── STABLE_WAIT     (200ms re-verify on UI motion)
                │               └── PENDING         (page loading, 2–8s backoff)
                ├── Recover    RuleBasedRecoveryManager
                ├── Reflect    PostRunReflector  (terminal only)
                └── Persist    FileBackedRunStore + RollingElementBuffer + MemoryStore

WebSocket (port 9001) ──▶ Command Center UI (React 19 + Zustand 5)
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

Open **http://localhost:8080** — the Command Center UI loads automatically.

---

## Command Center UI

The UI streams live over WebSocket and gives you full control of every run.

| Pane | Purpose |
|---|---|
| **Task Intelligence** (35%) | Subgoal tree, reasoning log (Thought Cards), HITL approval card |
| **Live Execution** (50%) | Real-time browser mirror, element bounds overlay, autonomy controls |
| **Moat Builder** | Toggle engine rules on/off, configure CDP port, set session mode |

### Autonomy Threshold

Drag the slider in the Live Execution pane to set a confidence floor. When the agent's confidence drops below it, the run pauses — you see the pending action and can either **Proceed** or inject a **Correction Hint** that flows into the next LLM call.

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
| `OPERON_TRACE` | — | `1` enables `[TRACE]` loop events |
| `OPERON_TEST_SAFE_MODE` | `false` | Skip display baseline in tests |

---

## API

```
POST /run-task        Start a new run
POST /step            Advance one step
POST /resume          Resume a HITL-paused run
POST /stop            Cancel an active run
GET  /run/{id}        Read run state
POST /connect-cdp     Attach to Chrome via CDP + start screencast
GET  /health          Health check

GET  /observer/api/runs         Recent runs list
GET  /observer/api/run/{id}     Full run snapshot
GET  /observer/api/artifact     Serve step artifact files
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
              benchmark.py, screen_recorder.py
  executor/   desktop.py, browser_native.py, browser_adapter.py,
              desktop_adapter.py, os_picker_macro.py
  api/        server.py, routes.py, observer.py, runtime_config.py,
              static/ (landing, console, dashboard, benchmarks)
  clients/    gemini.py, anthropic.py, gemini_computer_use.py
  models/     state.py, perception.py, policy.py, execution.py,
              verification.py, recovery.py, memory.py, logs.py, common.py
  store/      run_store.py, memory.py, run_logger.py, background_writer.py
  runtime/    orchestrator.py, state.py, legacy_adapter.py, benchmark_runner.py
  benchmarks/ registry.py, form_plugin.py, webarena.py, …
  core/       contracts/, router.py
ui/           React 19 Command Center (Vite + Zustand 5 + react-resizable-panels)
prompts/      policy_prompt.txt, perception_prompt.txt, critic_prompt.txt,
              video_verification_prompt.txt, reaction_check_prompt.txt, …
runs/<run_id>/   state.json, run.jsonl, step_N/ (screenshots + all model I/O)
memory/          memory.jsonl, episodes.jsonl
```

---

## Roadmap

- [ ] Claude Computer Use backend for browser perception
- [ ] WebArena medium/hard benchmark suite
- [ ] Prompt caching (Vertex AI context cache for static prefixes)
- [ ] Tauri desktop app packaging
- [ ] Multi-monitor support

---

<div align="center">

Built with [Gemini](https://deepmind.google/technologies/gemini/) · [FastAPI](https://fastapi.tiangolo.com/) · [Playwright](https://playwright.dev/) · [React 19](https://react.dev/)

</div>
