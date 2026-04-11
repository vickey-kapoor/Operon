# Operon

Vision-driven computer-use engine. Closed loop: capture → perceive → decide → execute → verify → recover. Desktop (pyautogui) and browser (Playwright + Gemini Computer Use) share one loop.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Put `GOOGLE_API_KEY` in `.env`.

## Run

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

- `/` — Pilot UI
- `/console` — Task Console
- `POST /run-task`, `/step`, `/resume`, `/stop`, `/cleanup`
- `GET /run/{id}`, `/observer/api/runs`, `/observer/api/run/{id}`

## Env

- `OPERON_BROWSER_BACKEND=computer_use` (primary) / `json` (fallback)
- `OPERON_DESKTOP_BACKEND=json`
- `BROWSER_HEADLESS=true`
- `OPERON_TRACE=1` — cyan `[TRACE]` lines from the loop

## Tests

```powershell
python -m pytest tests -q
ruff check src tests
```

## Layout

- `src/agent/` — loop, policy, selectors, verifier, recovery, reflector
- `src/executor/` — desktop + native browser executors
- `src/api/` — FastAPI + Pilot/Console UIs
- `src/store/` — run store, memory, episodes
- `src/models/` — Pydantic schemas
- `core/`, `runtime/`, `executors/` — unified contract layer (observer on top of the loop)
- `prompts/`, `docs/`, `examples/contracts/`
- Artifacts: `runs/<run_id>/`, `.browser-artifacts/`

See `CODEBASE_OVERVIEW.md` and `CLAUDE.md` for details.
