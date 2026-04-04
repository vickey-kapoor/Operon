# Operon

![Operon loop mark](assets/operon-loop-mark.svg)

Operate desktop apps and browser sessions through a closed loop: observe, choose the next action, execute it, verify the result, and store the run.

License: Apache-2.0

## Overview

Operon has two execution paths:

- Desktop mode uses full-screen automation with `pyautogui` and `mss`.
- Browser mode uses a native Playwright executor with Gemini Computer Use and a JSON fallback backend.

Both paths share the same agent loop, verifier, recovery logic, and local run store.

## Architecture

```text
FastAPI routes
    |
    +-- AgentLoop
          |
          +-- capture
          +-- backend selection
          |     - desktop: combined JSON perception + policy
          |     - browser: Computer Use or JSON fallback
          +-- executor
          |     - DesktopExecutor
          |     - NativeBrowserExecutor
          +-- verifier + recovery
          +-- local run persistence
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/agent/` | Main loop, backend adapters, verifier, policy coordinator, recovery |
| `src/api/` | FastAPI routes, Pilot UI, observer endpoints |
| `src/executor/` | Desktop and native browser execution |
| `src/clients/` | Gemini HTTP clients, including Computer Use |
| `src/models/` | Pydantic schemas for actions, runs, perception, and execution |
| `src/store/` | Local run history, memory, replay, and summaries |
| `prompts/` | Prompt templates for desktop, browser JSON, and browser Computer Use |
| `tests/` | Unit, route, and browser-focused regression coverage |

## Browser Runtime

The browser path is built around:

- `BrowserComputerUseBackend` for Gemini Computer Use turns
- `BrowserJsonBackend` as a fallback backend
- `NativeBrowserExecutor` for Playwright execution
- mandatory browser-session video recording under `.browser-artifacts/`
- observer support to review saved browser recordings from the run detail page

Computer Use actions are translated to the documented browser protocol, including normalized coordinate handling and multi-call turns.

## Setup

Python 3.11 is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Copy `.env.example` to `.env` and set your Gemini key:

```env
GOOGLE_API_KEY=your-gemini-api-key
```

Useful runtime settings:

- `BROWSER_HEADLESS=true`
- `OPERON_BROWSER_BACKEND=computer_use`
- `OPERON_BROWSER_FALLBACK_BACKEND=json`
- `OPERON_DESKTOP_BACKEND=json`

## Run

Start the API:

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Main endpoints:

- `POST /run-task`
- `POST /step`
- `POST /resume`
- `GET /run/{run_id}`
- `GET /observer/api/runs`
- `GET /observer/api/run/{run_id}`

Open `http://127.0.0.1:8080/` for the Pilot UI and observer.

## Artifacts

Operon stores run state under `runs/`. Browser recordings are saved under `.browser-artifacts/` and linked back into the run snapshot for the UI.

Typical artifacts include:

- screenshots before and after each step
- execution traces
- verifier outputs
- reflection summaries
- browser session video for browser runs

## Tests

```powershell
python -m pytest tests -q
ruff check src tests --select E,F,W,I --ignore E501
```

If PowerShell process launching breaks on Windows:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```
