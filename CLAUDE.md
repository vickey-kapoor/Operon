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

# Offline test suite (~860 tests, ~45s, no network, no browser)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests -q

# Offline eval: scripted scenarios through the real loop (evals/); report table
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m evals

# Live smoke: a few real browser tasks against a running API (real GEMINI_API_KEY)
.venv\Scripts\python -m evals.live_smoke

# Live benchmark: 44 real web tasks graded from the final screen; saves to .var/bench/
# Stops once a run has spent --max-cost-usd / $MAX_RUN_COST_USD (default 5; 0 = no cap)
.venv\Scripts\python -m evals.live_bench [--category forms] [--only NAME]

# Hard subset repeated 3 times: pass@1 with a CI, pass^3 and flaky tasks (--repeat works on any suite)
.venv\Scripts\python -m evals.live_bench --suite hard --repeat 3

# Grounding eval: ScreenSpot-v2/Pro samples, one model call each through the real backend, click scored vs the box
.venv\Scripts\python -m evals.grounding [--per-set 100] [--set v2-web] [--backend computer_use] [--list]

# Live safety suite: prompt-injection and overreach tasks on a local site, graded from its request log
.venv\Scripts\python -m evals.live_bench --suite safety

# Live apps suite: 11 tasks on local fake shop, mail, account and forms, graded from their state (/state)
.venv\Scripts\python -m evals.live_bench --suite apps
.venv\Scripts\python -m evals.local_apps   # serve the local site on :8765 to look at it

# Judge check: compare the benchmark judge with hand labels (judge calls only, no agent runs)
.venv\Scripts\python -m evals.judge_check template .var\bench\full-<stamp>.json > labels.jsonl
.venv\Scripts\python -m evals.judge_check run labels.jsonl [--repeat 3]

# Live desktop benchmark: 10 tasks on stock Windows apps; drives the real mouse and keyboard
# (in CI: hosted windows-latest, or our Azure VM via `/bench desktop azure`; docs/azure-desktop-runner.md)
.venv\Scripts\python -m evals.live_desktop_bench [--suite smoke] [--keep-screens]

# Live benchmarks from a PR comment (owners/collaborators; runs on the PR head, never on push):
#   /bench [browser|desktop] [thin|full] [smoke|all|safety|apps|hard] [x2|x3|x5] [opus|sonnet|haiku] [check-sonnet|check-haiku] [azure] [zoom]
#   /bench grounding [opus|sonnet|haiku|gemini] [zoom]
# Repo secrets: ANTHROPIC_API_KEY, GEMINI_API_KEY, DATABASE_URL, BLOB_READ_WRITE_TOKEN, SLACK_WEBHOOK_URL,
# AZURE_CREDENTIALS (fallback); variables AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID (OIDC),
# optional AZURE_VM_RESOURCE_GROUP, AZURE_VM_NAME. Keep Vercel Authentication (All Deployments) on for the dashboard. Benchmarks cost real credits: run only when asked.

# Eval dashboard (dashboard/, Vercel + Neon Postgres): write reports to the DB, then serve locally
# (the import needs `pip install "psycopg[binary]"`; the "Dashboard data" workflow does it after each benchmark)
$env:DATABASE_URL = "postgres://..."; .venv\Scripts\python -m evals.dashboard import .var\bench\full-<stamp>.json
cd dashboard; npm install; npm run dev   # page + API on :3000, reads $env:DATABASE_URL

# Single test / file
.venv\Scripts\python -m pytest tests/agent/test_loop.py::test_name -q

# Lint (exactly what CI runs)
.venv\Scripts\ruff check src tests evals --select E,F,W,I --ignore E501

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

Opt-in suites (in `tests/integration/`): `test_e2e_quick_tasks.py`, `test_bug_fixes_verification.py` (live
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
| `agent/backends/` | Model backends: `browser_claude_computer_use` / `desktop_claude_computer_use` (Claude, default with `ANTHROPIC_API_KEY`), `browser_computer_use` (Gemini CU), `browser_json`, plus `fallback`/`combined` wrappers |
| `agent/policy/` | `rules.py` (deterministic engine), `coordinator.py` (post-LLM guards), `verifier.py`, `recovery.py`, `trust_gate.py`, `best_of_n.py` |
| `agent/actions/` | Grounding, target selection, action translation, retry hardening |
| `executor/` | `browser_native.py`, `desktop.py`, `os_picker_macro.py` + adapters to the unified contract |
| `browser/manager.py` | CDP `BrowserManager` for observable mode (screencast + `inject_input`) |
| `api/routes/` | `browser.py`, `desktop.py`, `observer.py`; `ws_stream.py` serves `/ws/stream` |
| `store/` | Run persistence, memory, replay, cleanup, background writer |
| `models/` | Pydantic schemas shared across every layer |
| `prompts/` | Model prompt templates (package data, loaded via `core.paths.prompts_dir()`) |

Outside the package: `evals/` holds the benchmarks and eval tools (run as
`python -m evals.<tool>`), `tests/` the pytest suite (eval-tool tests in `tests/evals/`,
opt-in live suites in `tests/integration/`), `docs/reports/` the benchmark write-ups,
`scripts/` maintenance scripts, `ui/` the React frontend and `dashboard/` the Vercel eval
dashboard (Vercel's root directory, so don't move it).

Browser modes: **batch** launches an isolated Chromium (no CDP, no debug port); **observable**
owns CDP on port 9222 and streams JPEG frames over `/ws/stream`.

## Invariants

These are load-bearing — violating them silently breaks the agent.

- **Vision-only.** No DOM, CSS selectors, XPath, accessibility tree, or Playwright
  `locator()` in the perception/policy path. All targeting goes through `UIElement`
  coordinates from perception output. Improving perception is the intended fix for
  bad targeting.
- **Rules before LLM (full harness).** Anything decidable deterministically belongs in
  `PolicyRuleEngine` (`agent/policy/rules.py`). Post-LLM guards (e.g.
  `_semantic_anchor_check`) live on `PolicyCoordinator`. The thin harness
  (`agent/policy/thin.py`, the default for `claude_computer_use`; `OPERON_HARNESS_PROFILE=full`
  opts out) runs none of them: Claude's
  action is never replaced, retried or reset by the loop; fix failures there by
  improving what the model sees or the executor, not by adding rules.
- **No scenario hardcoding.** Never branch on a specific site, task, or benchmark
  in core policy. Site-specific behavior belongs in `agent/policy/site_adapters.py`.
- **Atomic TYPE.** The executor merges focus+type into one call. Policy must never
  emit a CLICK immediately followed by a TYPE on the same element.
- **Visual servo.** `_region_has_content()` runs before every click with an
  adaptively calibrated threshold. Never bypass it.
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
