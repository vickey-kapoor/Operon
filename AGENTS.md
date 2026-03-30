# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the application code. Key packages are `src/agent/` for the closed-loop orchestration (including the combined perception+policy service), `src/api/` for the FastAPI server and Operon Pilot UI, `src/executor/` for desktop automation (pyautogui/mss), `src/clients/` for the async Gemini HTTP client (httpx with HTTP/2), `src/models/` for Pydantic schemas, and `src/store/` for local run persistence, memory, and replay tools. Tests live in `tests/` and generally mirror module boundaries. Prompts are stored in `prompts/`, static assets in `assets/`, and environment repair helpers in `scripts/`.

## Build, Test, and Development Commands
Use Python 3.11 locally.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Common commands:

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
python -m pytest tests\ -q
python -m pytest tests\test_agent_loop.py -q
ruff check src tests --select E,F,W,I --ignore E501
python -m src.agent.benchmark
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints, small focused modules, and Pydantic models for structured data. Use `snake_case` for functions, variables, and test files; `PascalCase` for classes; and keep FastAPI route handlers and model names explicit. CI enforces Ruff rules `E,F,W,I`; import ordering matters, while long lines are tolerated (`E501` ignored).

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). Name tests `test_<behavior>` and keep one responsibility per test. Run the full suite before opening a PR. Use the real `GEMINI_API_KEY` from `.env` — do not use fake/mock API keys. Add or update tests whenever changing agent flow, API contracts, persistence, or execution behavior.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects with prefixes such as `Fix:`, `Docs:`, `CI:`, `Chore:`, and `Refactor:`. Keep commit titles specific, scoped, and under one line. PRs should describe the user-visible or architectural change, list validation performed (`pytest`, `ruff`, benchmark/manual API checks), and include screenshots only when the UI or desktop behavior changes. Link related issues when applicable.

## Configuration & Security Tips
Keep secrets in `.env`; never commit real credentials. `runs/` stores local artifacts and screenshots, so review generated files before sharing. If PowerShell process launching breaks on Windows, use `scripts/repair-process-env.ps1 -PersistForSession`.
