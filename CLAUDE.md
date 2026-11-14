# CLAUDE.md

Operon is a vision-first computer-use agent: it drives browsers and desktop apps
from pixels only. See `AGENTS.md` for contributor conventions and `docs/architecture.md`
for the long-form design.

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
# Front door (KAN-89): API + task page link (/app#key=...); one task here, or sent to a running API
.venv\Scripts\operon setup     # KAN-140: save Gemini key to ~/.operon, install Chromium, self-check (--check only checks)
.venv\Scripts\operon serve
.venv\Scripts\operon run "task" [--desktop] [--url URL] [--mode ask|risky|auto]
.venv\Scripts\operon run --request "task"
.venv\Scripts\operon envs      # KAN-80: list desktop run environments and whether each is ready here
# Release (KAN-36): push tag v<version> matching pyproject.toml; release.yml uploads operon-agent to PyPI (docs/releasing.md)

# Offline test suite (no network, no browser)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests -q

# Offline eval: scripted scenarios through the real loop (evals/); report table
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m evals

# Live smoke: a few real browser tasks against a running API (real GEMINI_API_KEY)
.venv\Scripts\python -m evals.live_smoke

# Benchmarks, seasons, judge check, /bench PR-comment syntax, repo secrets and the eval dashboard:
# see evals/CLAUDE.md. Benchmarks cost real credits: run only when asked, and name the run by cell
# when asking Vickey: "Gemini · Browser · Thin · once, about $X".

# Lint (exactly what CI runs)
.venv\Scripts\ruff check src tests evals --select E,F,W,I --ignore E501

# Type check (KAN-159; CI "Type check" job). Baseline modules are listed in [tool.mypy] in
# pyproject.toml: fix one, delete it from the list. Coverage: add --cov to the pytest command.
.venv\Scripts\mypy
```

`addopts = "-m 'not live_server'"` in `pyproject.toml` auto-deselects the ~350
`live_server` tests. **Do not run them unless explicitly asked** — they need a
uvicorn instance on `localhost:8080` and, in some cases, a headed Windows session
that will drive the real mouse and keyboard:

```powershell
$env:OPERON_RUN_LIVE_SERVER_TESTS = "true"; .venv\Scripts\python -m pytest tests -m live_server
```

## Architecture

One control loop serves both environments:

```
capture -> perceive -> decide -> execute -> verify -> recover
```

`AgentLoop` (`src/operon/agent/loop.py`, ~2300 lines) is the orchestrator. Each
`step_run()` call is one iteration; the API drives it, so runs are resumable and
pausable between steps. A step is a chain of stage methods (`_perceive_stage`,
`_decide_stage`, `_gate_stage`, `_execute_and_verify_stage`, `_recover_stage`, ...)
over a typed `StepContext` (`agent/step_context.py`, KAN-155). Broad `except` blocks in the
loop go through `AgentLoop._recovered()`, which logs and counts them per run
(`recovered_errors`); `OPERON_STRICT_ERRORS` (on in tests) re-raises bugs (KAN-158). Wiring is assembled in `src/operon/api/runtime/services.py`
(`build_browser_services` / `build_desktop_services`) — that is where backends,
executors, verifiers, and stores are chosen per environment. The desktop machine itself
comes from an environment provider (`src/operon/environments/`, `OPERON_ENVIRONMENT`,
default `live`; KAN-80, see `docs/environments.md`). `src/operon/core/paths.py`
resolves every artifact directory and honors the `OPERON_*_ROOT` env overrides. Settings
are one typed object, `get_settings()` in `src/operon/core/settings.py` (KAN-157): read fields
there, never `os.getenv`; it is checked at startup and saved per run as `settings.json`.

`dashboard/` is Vercel's root directory, so don't move it.

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
  `_reject_premature_stop`) live on `PolicyCoordinator`. Gemini Computer Use reports no
  element list or page type, so a rule must work without them or it never fires (KAN-161). The thin harness
  (`agent/policy/thin.py`, opt in with `OPERON_HARNESS_PROFILE=thin`) runs none of them: the model's
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
  `OPERON_BESTOFN_*`, `OPERON_GROUNDER` pattern (the trust gate graduated to on by default); larger proposals
  get an RFC in `docs/rfcs/`.
- **`.env.example` is a contract.** Every variable in it must actually be read by
  code in `src/`, and new settings must be added there (and to `Settings`).

## Runtime artifacts

Runs write to `.var/runs/<run_id>/` (`~/.operon/runs/<run_id>/` when installed with pip) (screenshots, model I/O, policy decisions,
execution traces, logs) — gitignored, and **screenshots capture the whole screen**,
so review before sharing. Legacy `runs/`, `.browser-artifacts/`, `.desktop-artifacts/`
paths still appear in old logs and tests.

Never commit anything under `.claude/` — it is gitignored and personal.

## Testing conventions

New behavior needs a test that runs
without a live server or a real screen — if that's genuinely impossible, mark it
`@pytest.mark.live_server` and keep the code path thin, since CI won't cover it.
`OPERON_TEST_SAFE_MODE=true` skips display baseline and servo calibration.

## Commits

Short imperative subjects with a prefix: `Fix:`, `Feat:`, `Docs:`, `CI:`, `Chore:`,
`Refactor:`. One line, specific.
