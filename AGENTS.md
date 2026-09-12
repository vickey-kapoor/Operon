# Repository Guidelines
_For AI coding agents and contributors._

## Project Structure & Module Organization
Operon uses a Python `src/` layout. Core orchestration lives in `src/operon/agent/`, with focused `actions/`, `artifacts/`, `backends/`, `perception/`, and `policy/` subpackages. API routes live in `src/operon/api/routes/`, shared API runtime construction in `src/operon/api/runtime/`, CDP-based browser observation in `src/operon/browser/`, executors in `src/operon/executor/`, Gemini clients in `src/operon/clients/`, shared schemas in `src/operon/models/`, and persistence in `src/operon/store/`. Tests live in `tests/` and generally mirror the package they cover. Supporting material is kept in `prompts/` and `docs/`.

## Build, Test, and Development Commands
Use Python 3.14 locally. Package metadata remains compatible with Python 3.11+.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Run the API locally with `.venv\Scripts\python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080`.

Run the default test suite (no live server required):

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
.venv\Scripts\python -m pytest tests -q
```

No `--ignore` flags needed: `addopts = "-m 'not live_server'"` in `pyproject.toml` already deselects the suites that require a running server.

Lint with `ruff check src tests --select E,F,W,I --ignore E501`.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, type hints, `snake_case` for modules/functions/variables, and `PascalCase` for classes and Pydantic models. Keep modules focused and prefer explicit names such as `browser_native.py` or `policy_coordinator.py`. Ruff enforces import sorting and core lint rules; long lines are tolerated.

## Key Architecture Points
Operon is vision-only: no DOM, CSS selectors, XPath, or Playwright `locator()` in the perception/policy path — all targeting uses `UIElement` coordinates from perception output. Deterministic logic belongs in `PolicyRuleEngine` before any LLM call.

The full set of load-bearing invariants (spatial buffer TTLs, verification states, atomic TYPE, visual servo, observable-mode import rules) lives in [CLAUDE.md](CLAUDE.md#invariants). Read it before changing agent flow.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). Name files and functions as `test_<behavior>`. Add or update tests whenever changing agent flow, API contracts, persistence, or executor behavior. CI runs Python 3.11 and 3.14 plus Ruff.

Live-server and real-environment tests are opt-in and excluded from the default CI path:
- `tests/test_e2e_quick_tasks.py`, `tests/test_bug_fixes_verification.py` - require a live server
- `tests/test_live_execution.py` - capability regression gate
- `tests/test_upload_file_native_integration.py` - headed Windows-only, native OS file picker

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects with prefixes like `Fix:`, `Docs:`, `CI:`, `Chore:`, `Refactor:`, and `Feat:`. Keep commit titles specific and one line. PRs should summarize the behavioral change, list validation performed, link related issues, and include screenshots only for UI or desktop-behavior changes.

## Configuration & Security Tips
Store secrets in `.env` and keep `.env.example` in sync when adding new settings. Review `.var/runs/`, `.var/browser-artifacts/`, and `.var/desktop-artifacts/` before sharing logs because they may contain screenshots, prompts, and execution traces. Never commit anything under `.claude/`; it is gitignored and personal.
