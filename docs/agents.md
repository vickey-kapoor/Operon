# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the application code. Key packages are `src/agent/` for the closed-loop orchestration (combined perception+policy service, video verification, screen diff, post-run reflector), `src/api/` for the FastAPI server and Operon Pilot UI, `src/executor/` for desktop automation (pyautogui/mss, screen recording via OpenCV), `src/clients/` for the async Gemini HTTP client (httpx with HTTP/2, supports image + video payloads), `src/models/` for Pydantic schemas, and `src/store/` for local run persistence, memory, and replay tools. Tests live in `tests/` and generally mirror module boundaries. Prompts are stored in `prompts/`, static assets in `assets/`, and environment repair helpers in `scripts/`.

## Build, Test, and Development Commands
Use Python 3.11 locally.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Common commands:

```powershell
\.\venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
\.\venv\Scripts\python.exe -m pytest tests\ -q
\.\venv\Scripts\python.exe -m pytest tests\test_agent_loop.py -q
ruff check src tests --select E,F,W,I --ignore E501
\.\venv\Scripts\python.exe -m src.agent.benchmark
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints, small focused modules, and Pydantic models for structured data. Use `snake_case` for functions, variables, and test files; `PascalCase` for classes; and keep FastAPI route handlers and model names explicit. CI enforces Ruff rules `E,F,W,I`; import ordering matters, while long lines are tolerated (`E501` ignored).

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). Name tests `test_<behavior>` and keep one responsibility per test. Run the full suite before opening a PR. Use the repo virtualenv interpreter (`.\.venv\Scripts\python.exe`), not system `python`, because several dependencies are only installed in `.venv`. Use the real `GEMINI_API_KEY` from `.env` - do not use fake or mock API keys. Add or update tests whenever changing agent flow, API contracts, persistence, or execution behavior.

### Safe Test Execution For Agents
Before running broad pytest commands, scan the test tree for live-server or real-desktop patterns:

```powershell
rg -n "127.0.0.1:8080|/desktop/run-task|/desktop/step|/desktop/cleanup|/desktop/resume|requests\.Session\(|pyautogui|mss" tests
```

Rules:

- Do not run live-server tests unless explicitly requested.
- Treat any test that talks to `http://127.0.0.1:8080` or posts to `/desktop/*` as potentially dangerous.
- Prefer browser `/run-task` over `/desktop/run-task` for API-contract tests unless the test is specifically about desktop executor behavior.
- Keep `OPERON_TEST_SAFE_MODE=true` for normal test runs.
- The live-server modules `tests/test_e2e_quick_tasks.py` and `tests/test_bug_fixes_verification.py` are opt-in only. Exclude them from broad runs unless the user explicitly asks for live-server validation.

Safe default broad run:

```powershell
\.\venv\Scripts\python.exe -m pytest tests\ -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py
```

Intentional live-server run:

```powershell
$env:OPERON_RUN_LIVE_SERVER_TESTS='true'
\.\venv\Scripts\python.exe -m pytest tests\test_e2e_quick_tasks.py -q --live
\.\venv\Scripts\python.exe -m pytest tests\test_bug_fixes_verification.py -q --live
```

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects with prefixes such as `Fix:`, `Docs:`, `CI:`, `Chore:`, and `Refactor:`. Keep commit titles specific, scoped, and under one line. PRs should describe the user-visible or architectural change, list validation performed (`pytest`, `ruff`, benchmark/manual API checks), and include screenshots only when the UI or desktop behavior changes. Link related issues when applicable.

## Configuration & Security Tips
Keep secrets in `.env`; never commit real credentials. `runs/` stores local artifacts and screenshots, so review generated files before sharing. If PowerShell process launching breaks on Windows, use `scripts/repair-process-env.ps1 -PersistForSession`.
