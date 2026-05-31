# Repository Guidelines
_For AI coding agents and contributors._

## Project Structure & Module Organization
Operon uses a Python `src/` layout. Core agent logic lives in `src/operon/agent/`, API routes in `src/operon/api/`, CDP-based browser observation in `src/operon/browser/`, executors in `src/operon/executor/`, Gemini clients in `src/operon/clients/`, shared schemas in `src/operon/models/`, and persistence in `src/operon/store/`. Tests live in `tests/` and generally mirror the package they cover. Supporting material is kept in `prompts/`, `docs/`, `assets/`, `examples/contracts/`, and `scripts/`. The optional React 19 Command Center UI lives in `ui/`.

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
.venv\Scripts\python -m pytest tests\ -q --ignore=tests/test_e2e_quick_tasks.py --ignore=tests/test_bug_fixes_verification.py
```

Lint with `ruff check src tests --select E,F,W,I --ignore E501`.

## Key Architecture Points
- **Vision-only**: no DOM, no CSS selectors, no XPath, no Playwright `locator()` in the policy/perception path. All targeting uses `UIElement` coordinates from perception output.
- **Rules before LLM**: deterministic logic belongs in `PolicyRuleEngine`. Post-LLM guards, such as `_semantic_anchor_check`, live on `PolicyCoordinator`.
- **Spatial persistence**: `RollingElementBuffer` tracks elements across steps. Ghost elements have TTL=2 and are auto-purged. Buffer clears on `visual_velocity > 5%`.
- **Verification states**: SUCCESS / FAILURE / UNCERTAIN / PENDING / PROGRESSING_STABLE / STABLE_WAIT. `STABLE_WAIT` triggers a 200ms re-verify; `PROGRESSING_STABLE` advances immediately.
- **Atomic TYPE**: executor merges focus+type into one call. Never emit a CLICK immediately followed by TYPE on the same element from policy.
- **Visual servo**: `_region_has_content()` runs before every click with an adaptively calibrated threshold. Never bypass it.
- **Observable mode**: `BrowserManager` (`src/operon/browser/manager.py`) connects via `connect_over_cdp`, streams JPEG frames via CDP `Page.startScreencast`, and exposes `inject_input()` for interactive control. Never import `ws_stream` at module load; use the lazy `_ws_stream()` accessor to avoid circular imports.
- **Benchmark hooks**: benchmark datasets and the local runner are optional. Do not add benchmark-specific logic to core policy rules unless a dedicated plugin registry is restored.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). Name files and functions as `test_<behavior>`. Add or update tests whenever changing agent flow, API contracts, persistence, or executor behavior. CI runs Python 3.11 and 3.14 plus Ruff.

Live-server and real-environment tests are opt-in and excluded from the default CI path:
- `tests/test_e2e_quick_tasks.py`, `tests/test_bug_fixes_verification.py` - require a live server
- `tests/test_live_execution.py` - capability regression gate
- `tests/test_upload_file_native_integration.py` - headed Windows-only, native OS file picker
- `tests/test_browserbase_integration.py` - env-gated, real Browserbase backend
- `tests/test_file_porter_integration.py` - env-gated, real Google Drive upload

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects with prefixes like `Fix:`, `Docs:`, `CI:`, `Chore:`, `Refactor:`, and `Feat:`. Keep commit titles specific and one line. PRs should summarize the behavioral change, list validation performed, link related issues, and include screenshots only for UI or desktop-behavior changes.

## Configuration & Security Tips
Store secrets in `.env` and keep `.env.example` in sync when adding new settings. Review `runs/` and `.browser-artifacts/` before sharing logs because they may contain screenshots, prompts, and execution traces. Never commit anything under `.claude/`; it is gitignored and personal.
