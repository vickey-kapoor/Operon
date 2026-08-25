<div align="center">

# Operon

[![CI](https://github.com/vickey-kapoor/Operon/actions/workflows/ci.yml/badge.svg)](https://github.com/vickey-kapoor/Operon/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

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
- Run artifacts under `.var/runs/<run_id>/` by default, including screenshots, model I/O, policy decisions, execution traces, and logs.
- Observable browser mode via CDP screencast and `/ws/stream`.

## Quick Start

Requirements: Python 3.11 or newer, and Chrome/Chromium for browser runs. CI tests
against 3.11 and 3.14.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
playwright install chromium
copy .env.example .env
```

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`, then start the API:

```bash
python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080/console for the Command Center UI, or drive the API
directly:

```bash
curl -X POST http://127.0.0.1:8080/desktop/run-task \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "Open notepad and type hello world in go programming language"}'
```

> **Desktop runs control your actual mouse and keyboard.** Run them on a machine
> you can afford to have clicked on, or inside a VM. Browser runs are contained
> to the Chromium instance Operon launches.

## API Surface

The active FastAPI routes are registered from `src/operon/api/routes/`.

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
POST /desktop/stop
POST /desktop/run/{run_id}/stop
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
| `API_KEYS` | unset | Comma-separated API keys. When set, every HTTP route except `/health` requires a matching `X-API-Key` header, and the `/ws/stream` WebSocket requires the key via `?api_key=` (or an `x-api-key` subprotocol). **Unset = no auth** — only safe on a trusted host, since the API can drive a real browser/desktop. |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use` or `json` |
| `OPERON_BROWSER_FALLBACK_BACKEND` | `json` | Optional browser fallback backend |
| `OPERON_DESKTOP_BACKEND` | `json` | Desktop backend |
| `OPERON_BROWSER_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_DESKTOP_PLANNER_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `OPERON_BROWSER_MODEL` | `gemini-2.5-computer-use-preview-10-2025` | Browser primary model |
| `OPERON_DESKTOP_MODEL` | `gemini-2.5-flash` | Desktop primary model |
| `BROWSER_HEADLESS` | `false` | Browser headed/headless behavior |
| `BROWSER_WIDTH` | `1920` | Browser viewport width |
| `BROWSER_HEIGHT` | `1080` | Browser viewport height |
| `OPERON_RUNTIME_ROOT` | `.var` | Root for generated runtime artifacts |
| `OPERON_RUNS_ROOT` | `.var/runs` | Override run state and step artifact storage |
| `OPERON_BROWSER_ARTIFACTS_ROOT` | `.var/browser-artifacts` | Override browser recording/session artifacts |
| `OPERON_DESKTOP_ARTIFACTS_ROOT` | `.var/desktop-artifacts` | Override desktop recording/session artifacts |
| `OPERON_TEST_ARTIFACTS_ROOT` | `.var/test-artifacts` | Override test artifact output |
| `OPERON_TRACE` | unset | Set `1` for loop trace logging |
| `OPERON_TEST_SAFE_MODE` | `false` | Skips display baseline and servo calibration in tests |
| `OPERON_BESTOFN_N` | `1` | Experimental (RFC 0001 Move 3): sample N candidate actions at uncertain steps and execute the best-scored one. `1` disables (default). |
| `OPERON_BESTOFN_CONFIDENCE` | `1.0` | Experimental: only sample when a step's policy confidence is below this ceiling (lower to restrict Best-of-N to uncertain steps). |
| `OPERON_GROUNDER` | `deterministic` | Experimental (RFC 0001 Move 2): grounder backend. `deterministic` (default, baseline) or `snap` (snap a near-miss raw coordinate onto the nearest interactable element). |
| `OPERON_TRUST_GATE` | `off` | Experimental (RFC 0001 Move 5): `on` enables the pre-execution deny/confirm policy gate. Off (default) = no behavior change. |
| `OPERON_TRUST_DENY` | unset | Comma-separated phrases that **block** an action (the run pauses for a human; the action never runs autonomously). Matched case-insensitively against the action text, URL, and target-element name. |
| `OPERON_TRUST_CONFIRM` | built-in set | Comma-separated phrases that require **human approval** before running. Defaults to a conservative high-risk set (e.g. `delete account`, `place order`) when the gate is on and this is unset. |
| `OPERON_TRUST_ALLOW_DOMAINS` | unset | Comma-separated domains (subdomains included). When set, navigation is **restricted to these domains** — any other URL is denied (fail-closed). Unset = no domain restriction. |

Generated runtime output is intentionally kept under `.var/` and ignored by git. Legacy `runs/`, `.browser-artifacts/`, and `.desktop-artifacts/` paths may still appear in old logs or tests, but new default runtime output uses the `.var/` tree unless overridden by environment variables.

## Project Layout

```text
src/
  agent/      loop plus focused actions, artifacts, backends, perception, and policy packages
  api/        FastAPI app, split routes, runtime construction, observer helpers, websocket stream
  browser/    CDP BrowserManager for observable mode
  clients/    Gemini, Gemini Computer Use, and Anthropic HTTP clients
  core/       shared Environment enum used to select the execution path
  executor/   browser, desktop, and native upload executors
  models/     Pydantic schemas for state, perception, policy, execution, logs, memory
  store/      run persistence, memory, replay, cleanup, background writer

prompts/      Model prompt templates
tests/        Pytest suite
docs/         Architecture and product notes
```

## Testing

```bash
GEMINI_API_KEY=fake-test-key python -m pytest tests -q
```

On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests -q
```

That runs the full offline suite — 552 tests, about 30 seconds, no API calls and
no browser. No `--ignore` flags needed: `addopts = "-m 'not live_server'"` in
`pyproject.toml` already deselects the suites that need a running server.

A further 350 tests are marked `live_server` and are **not** covered by CI. They
need a uvicorn instance on `localhost:8080` and, in some cases, a headed Windows
session. To opt in:

```bash
OPERON_RUN_LIVE_SERVER_TESTS=true python -m pytest tests -m live_server
```

Lint:

```bash
ruff check src tests --select E,F,W,I --ignore E501
```

