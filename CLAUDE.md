# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Operon is a vision-first computer-use agent: it drives browsers and desktop apps
from pixels only. See `AGENTS.md` for contributor conventions and `docs/architecture.md`
for the long-form design. `ui/CLAUDE.md` covers the React frontend.

## Environment

Always use the repo virtualenv — never bare `python`/`pip`:

```powershell
.venv\Scripts\python      # Windows
.venv/bin/python          # WSL / Linux / macOS
```

Setup: `python -m venv .venv`, then `.venv\Scripts\pip install -e ".[dev]"` and
`.venv\Scripts\playwright install chromium`. Dependencies are locked in `uv.lock`;
CI installs with `uv sync --locked --extra dev`, so update deps via `uv lock --upgrade`
(editing `pyproject.toml` alone makes CI fail on a stale lock).

## Commands

```powershell
# Run the API
.venv\Scripts\python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080

# Offline test suite (~554 tests, ~30s, no network, no browser)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests -q

# Single test / file
.venv\Scripts\python -m pytest tests/agent/test_loop.py::test_name -q

# Lint (exactly what CI runs)
.venv\Scripts\ruff check src tests --select E,F,W,I --ignore E501

# Frontend (from ui/)
npm run dev      # Vite dev server on :5173
npm run build    # tsc type-check + production build — the only automated gate
```

`addopts = "-m 'not live_server'"` in `pyproject.toml` auto-deselects the ~350
`live_server` tests. **Do not run them unless explicitly asked** — they need a
uvicorn instance on `localhost:8080` and, in some cases, a headed Windows session
that will drive the real mouse and keyboard:

```powershell
$env:OPERON_RUN_LIVE_SERVER_TESTS = "true"; .venv\Scripts\python -m pytest tests -m live_server
```

Opt-in suites: `test_e2e_quick_tasks.py`, `test_bug_fixes_verification.py` (live
server), `test_live_execution.py` (capability regression gate),
`test_upload_file_native_integration.py` (headed Windows, native OS file picker).

## Architecture

One control loop serves both environments:

```
capture -> perceive -> decide -> execute -> verify -> recover
```

`AgentLoop` (`src/operon/agent/loop.py`, ~1800 lines) is the orchestrator. Each
`step_run()` call is one iteration; the API drives it, so runs are resumable and
pausable between steps. Wiring is assembled in `src/operon/api/runtime/services.py`
(`build_browser_services` / `build_desktop_services`) — that is where backends,
executors, verifiers, and stores are chosen per environment. `src/operon/core/paths.py`
resolves every artifact directory and honors the `OPERON_*_ROOT` env overrides.

Layer map:

| Package | Role |
|---|---|
| `agent/perception/` | Screen capture, geometry, screen-diff, perception service |
| `agent/backends/` | Model backends: `browser_computer_use` (Gemini CU, default), `browser_json`, `anthropic_policy`, plus `fallback`/`combined` wrappers |
| `agent/policy/` | `rules.py` (deterministic engine), `coordinator.py` (post-LLM guards), `verifier.py`, `recovery.py`, `trust_gate.py`, `best_of_n.py` |
| `agent/actions/` | Grounding, target selection, action translation, retry hardening |
| `executor/` | `browser_native.py`, `desktop.py`, `os_picker_macro.py` + adapters to the unified contract |
| `browser/manager.py` | CDP `BrowserManager` for observable mode (screencast + `inject_input`) |
| `api/routes/` | `browser.py`, `desktop.py`, `observer.py`; `ws_stream.py` serves `/ws/stream` |
| `store/` | Run persistence, memory, replay, cleanup, background writer |
| `models/` | Pydantic schemas shared across every layer |

Browser modes: **batch** launches an isolated Chromium (no CDP); **observable**
owns CDP on port 9222 and streams JPEG frames over `/ws/stream`.

## Invariants

These are load-bearing — violating them silently breaks the agent.

- **Vision-only.** No DOM, CSS selectors, XPath, accessibility tree, or Playwright
  `locator()` in the perception/policy path. All targeting goes through `UIElement`
  coordinates from perception output. Improving perception is the intended fix for
  bad targeting.
- **Rules before LLM.** Anything decidable deterministically belongs in
  `PolicyRuleEngine` (`agent/policy/rules.py`). Post-LLM guards (e.g.
  `_semantic_anchor_check`) live on `PolicyCoordinator`.
- **No scenario hardcoding.** Never branch on a specific site, task, or benchmark
  in core policy. Site-specific behavior belongs in `agent/policy/site_adapters.py`.
- **Atomic TYPE.** The executor merges focus+type into one call. Policy must never
  emit a CLICK immediately followed by a TYPE on the same element.
- **Visual servo.** `_region_has_content()` runs before every click with an
  adaptively calibrated threshold. Never bypass it.
- **Spatial persistence.** `RollingElementBuffer` tracks elements across steps;
  ghost elements have TTL=2 and are auto-purged; the buffer clears when
  `visual_velocity > 5%`.
- **Verification states.** SUCCESS / FAILURE / UNCERTAIN / PENDING /
  PROGRESSING_STABLE / STABLE_WAIT. `STABLE_WAIT` triggers a 200ms re-verify;
  `PROGRESSING_STABLE` advances immediately.
- **Lazy `ws_stream`.** Never import `ws_stream` at module load in `browser/manager.py`
  — use the `_ws_stream()` accessor to avoid a circular import.
- **Experimental work ships behind an env flag defaulting to off.** Follow the
  `OPERON_BESTOFN_*`, `OPERON_GROUNDER`, `OPERON_TRUST_*` pattern; larger proposals
  get an RFC in `docs/rfcs/`.
- **`.env.example` is a contract.** Every variable in it must actually be read by
  code in `src/`, and new settings must be added there.

## Runtime artifacts

Runs write to `.var/runs/<run_id>/` (screenshots, model I/O, policy decisions,
execution traces, logs) — gitignored, and **screenshots capture the whole screen**,
so review before sharing. Legacy `runs/`, `.browser-artifacts/`, `.desktop-artifacts/`
paths still appear in old logs and tests.

Never commit anything under `.claude/` — it is gitignored and personal.

## Testing conventions

`pytest` with `pytest-asyncio` in `asyncio_mode = "auto"`; tests mirror the package
they cover; name things `test_<behavior>`. New behavior needs a test that runs
without a live server or a real screen — if that's genuinely impossible, mark it
`@pytest.mark.live_server` and keep the code path thin, since CI won't cover it.
`OPERON_TEST_SAFE_MODE=true` skips display baseline and servo calibration.

## Commits

Short imperative subjects with a prefix: `Fix:`, `Feat:`, `Docs:`, `CI:`, `Chore:`,
`Refactor:`. One line, specific.
