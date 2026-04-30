# Repository Guidelines

## Project Structure & Module Organization
Operon uses a Python `src/` layout. Core agent logic lives in `src/agent/`, API routes and static UIs in `src/api/`, executors in `src/executor/`, Gemini clients in `src/clients/`, shared schemas in `src/models/`, and persistence in `src/store/`. Tests live in `tests/` and generally mirror the package they cover. Supporting material is kept in `prompts/`, `docs/`, `assets/`, `examples/contracts/`, and `scripts/`.

## Build, Test, and Development Commands
Use Python 3.11 locally.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Run the API locally with `python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080`.

Run the default test suite with `python -m pytest tests -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py`.

Lint with `ruff check src tests --select E,F,W,I --ignore E501`.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, type hints, `snake_case` for modules/functions/variables, and `PascalCase` for classes and Pydantic models. Keep modules focused and prefer explicit names such as `browser_native.py` or `policy_coordinator.py`. Ruff enforces import sorting and core lint rules; long lines are tolerated.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). Name files and functions as `test_<behavior>`. Add or update tests whenever changing agent flow, API contracts, persistence, or executor behavior. CI runs Python 3.11 and 3.12 plus Ruff. There is no fixed coverage threshold in the repo today, so aim for targeted regression coverage.

Live-server and real-environment tests are opt-in. Do not include `tests/test_e2e_quick_tasks.py` or `tests/test_bug_fixes_verification.py` in broad local runs unless you intend to validate against a live server.
`tests/test_live_execution.py` is also opt-in and is intended to fail on real capability regressions, not to act as a smoke-only demo.
`tests/test_upload_file_native_integration.py` is a headed Windows-only manual gate for the native OS file picker path.
`tests/test_browserbase_integration.py` is an env-gated smoke test for the real Browserbase backend.
`tests/test_file_porter_integration.py` is an env-gated smoke test for real Google Drive upload via `FILE_PORTER`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects with prefixes like `Fix:`, `Docs:`, `CI:`, `Chore:`, and `Refactor:`. Keep commit titles specific and one line. PRs should summarize the behavioral change, list validation performed, link related issues, and include screenshots only for UI or desktop-behavior changes.

## Configuration & Security Tips
Store secrets in `.env` and keep `.env.example` in sync when adding new settings. Review `runs/` and `.browser-artifacts/` before sharing logs because they may contain screenshots, prompts, and execution traces.
