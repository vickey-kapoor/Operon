<div align="center">

<img src="assets/operon-loop-mark.svg" width="72" alt="Operon logo" />

# Operon

**A zero-abstraction, vision-first computer-use agent.**

No DOM targeting, no CSS selectors, no XPath. Operon observes pixels, chooses coordinate-based actions, executes them, verifies the visual result, and recovers when progress stalls.

</div>

---

## What It Does

Operon runs browser and desktop automation through the same control loop:

```text
capture -> perceive -> decide -> execute -> verify -> recover
```

Core runtime path:

```text
FastAPI routes
  -> AgentLoop
  -> ScreenCaptureService
  -> Gemini Computer Use or JSON perception/policy backend
  -> PolicyCoordinator and PolicyRuleEngine
  -> NativeBrowserExecutor or DesktopExecutor
  -> DeterministicVerifierService
  -> RuleBasedRecoveryManager
  -> FileBackedRunStore
```

## Key Capabilities

- Vision-only targeting with `UIElement` coordinates from perception output.
- Gemini Computer Use browser backend by default, with optional JSON fallback.
- Desktop JSON backend for native app control.
- Deterministic policy rules before LLM fallback.
- Rolling spatial memory for recent elements and ghost-element handling.
- Adaptive click servo checks before browser/desktop clicks.
- Human-in-the-loop pause/resume for uncertain or blocked states.
- Run artifacts under `runs/<run_id>/`, including screenshots, model I/O, policy decisions, execution traces, and logs.
- Observable browser mode via CDP screencast and `/ws/stream`.

## Quick Start

Requirements: Python 3.14 for local development and Chrome/Chromium for browser runs. Package metadata remains compatible with Python 3.11+.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
copy .env.example .env
```

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`, then start the API:

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

## API Surface

The active FastAPI routes are defined in `src/api/routes.py`.

```text
# Browser run lifecycle
POST /run-task
POST /step
POST /resume
POST /stop
POST /run/{run_id}/stop
POST /run/{run_id}/pause
GET  /run/{run_id}
POST /cleanup
GET  /health

# Desktop run lifecycle
POST /desktop/run-task
POST /desktop/step
POST /desktop/resume
POST /desktop/cleanup
GET  /desktop/run/{run_id}

# Observer / telemetry
GET  /observer/api/runs
GET  /observer/api/run/{run_id}
GET  /observer/api/artifact
GET  /observer/api/usage
GET  /observer/api/export/{run_id}
GET  /observer/api/live-browser/{run_id}

# Live stream
WS   /ws/stream
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | unset | Gemini API key |
| `ANTHROPIC_API_KEY` | unset | Required only when using Anthropic planner/verifier |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use`, `json`, or `browserbase` |
| `OPERON_BROWSER_FALLBACK_BACKEND` | `json` | Optional browser fallback backend |
| `OPERON_DESKTOP_BACKEND` | `json` | Desktop backend |
| `OPERON_BROWSER_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_DESKTOP_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_BROWSER_MODEL` | `gemini-2.5-computer-use-preview-10-2025` | Browser primary model |
| `OPERON_DESKTOP_MODEL` | `gemini-2.5-flash` | Desktop primary model |
| `BROWSER_HEADLESS` | `false` | Browser headed/headless behavior |
| `BROWSER_WIDTH` | `1920` | Browser viewport width |
| `BROWSER_HEIGHT` | `1080` | Browser viewport height |
| `OPERON_TRACE` | unset | Set `1` for loop trace logging |
| `OPERON_TEST_SAFE_MODE` | `false` | Skips display baseline and servo calibration in tests |

## Project Layout

```text
src/
  agent/      loop, perception, policy, policy rules, recovery, verifier,
              browser backends, action translation, targeting, progress tracking
  api/        FastAPI app, routes, observer helpers, runtime config, websocket stream
  browser/    CDP BrowserManager for observable mode
  clients/    Gemini, Gemini Computer Use, and Anthropic HTTP clients
  core/       contract models and route validation used by executor adapters
  executor/   browser, desktop, Browserbase, and native upload executors
  models/     Pydantic schemas for state, perception, policy, execution, logs, memory
  store/      run persistence, memory, replay, summaries, cleanup, background writer
  tools/      optional file transfer helpers

ui/           Optional React/Tauri command center UI
src-tauri/    Optional Tauri shell for the UI
prompts/      Model prompt templates
tests/        Pytest suite
docs/         Architecture and product notes
```

## Testing

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests -q `
  --ignore=tests/test_e2e_quick_tasks.py `
  --ignore=tests/test_bug_fixes_verification.py
```

Lint:

```powershell
ruff check src tests --select E,F,W,I --ignore E501
```

Some tests are intentionally opt-in because they require a live server, headed Windows browser session, Browserbase, or Google Drive credentials.

## Optional Components

These are not required for the core computer-use agent:

- `ui/` and `src-tauri/`: command center UI and desktop packaging shell.
- `src/executor/browserbase_native.py`: Browserbase backend.
- `src/tools/file_porter.py`: Google Drive file-transfer helper.
- `docs/substack_drafts/`: historical writeups.
