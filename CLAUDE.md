# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Operon is a vision-first computer-use agent: it drives browsers and desktop apps
from pixels only by default. See `AGENTS.md` for contributor conventions and `docs/architecture.md`
for the long-form design. Standing rules, infrastructure and current open items for a
local Claude Code session are in `docs/CLAUDE-CODE-REFERENCE.md`: read it at the start of a session.

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
# Front door (KAN-89): starts the API; one task here, or sent to a running API
.venv\Scripts\operon serve
.venv\Scripts\operon run "task" [--desktop] [--url URL] [--mode ask|risky|auto]
.venv\Scripts\operon run --request "task"
# KAN-103 part 2 (OPERON_SECRETS=on): the model types {{secret:NAME}} and never sees the value; the value is asked for hidden
.venv\Scripts\operon secrets set NAME     # also: operon secrets list (names only) / operon secrets rm NAME; see docs/secrets.md
.venv\Scripts\operon envs      # KAN-80: list desktop run environments and whether each is ready here

# Offline test suite (no network, no browser)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests -q
# One file or one test (tests/conftest.py sets the safe-mode, strict-errors and no-auth env defaults)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/api/test_ws_browser_frames.py -q -k parallel

# Free desktop end-to-end (KAN-166; CI job "Desktop (Linux container)"): real loop + real XFCE desktop
# in Docker, scripted model ($0). Needs Docker; deselected by default.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/desktop_e2e -m desktop_container -o addopts=""

# Free Windows e2e (KAN-93): real loop + real Windows desktop in Windows Sandbox (~2 min boot), scripted model ($0).
# Needs Windows Sandbox turned on; deselected by default. On an 8 GB PC run it alone (no other heavy
# processes) with $env:OPERON_SANDBOX_MEMORY_MB = "2048", or the sandbox closes before its guest starts.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/desktop_e2e_windows -m windows_sandbox -o addopts=""
# Free browser end-to-end: real loop + real NativeBrowserExecutor on headless Chromium against a local
# fixture site, scripted model ($0, ~3 min). Needs Playwright's Chromium; deselected by default.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/browser_e2e -m browser_e2e -o addopts=""

# Offline eval: scripted scenarios through the real loop (evals/); report table
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m evals

# Live smoke: a few real browser tasks against a running API (real GEMINI_API_KEY)
.venv\Scripts\python -m evals.live_smoke

# Paid runs are public benchmarks only (KAN-184): OSWorld and grounding. Benchmarks, judge check, /bench PR-comment syntax, repo secrets and the eval dashboard:
# see evals/CLAUDE.md. Benchmarks cost real credits: run only when asked, and name the run by cell
# when asking Vickey: "Gemini · OSWorld · 25 tasks · once, about $X".

# Lint (exactly what CI runs)
.venv\Scripts\ruff check src tests evals --select E,F,W,I --ignore E501

# Type check (KAN-159; CI "Type check" job). Baseline modules are listed in [tool.mypy] in
# pyproject.toml: fix one, delete it from the list. Coverage: add --cov to the pytest command.
.venv\Scripts\mypy

# Security checks (the Security workflow, KAN-34); each is a CI check on every PR
uvx zizmor==1.32.0 --offline --no-progress --min-severity medium .github/workflows
uv export --locked --all-extras --no-hashes --no-emit-project -o $env:TEMP\requirements.txt   # then:
uvx pip-audit==2.10.1 -r $env:TEMP\requirements.txt --no-deps --disable-pip
uvx --python 3.14 bandit==1.9.4 -r src -ll -q -b .bandit-baseline.json -f txt   # baseline compare needs Linux/WSL (forward-slash paths)
(cd dashboard; npm audit --package-lock-only --audit-level=high)
gitleaks git --no-banner --redact .      # history; `gitleaks dir --no-banner --redact .` scans the working tree
```

`addopts` in `pyproject.toml` auto-deselects the `live_server`, `desktop_container`,
`windows_sandbox` and `browser_e2e` tests (run those with `-o addopts=""` as above), including the ~350
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

`AgentLoop` (`src/operon/agent/loop.py`, ~3000 lines) is the orchestrator. Each
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
there, never `os.getenv`; it is checked at startup and saved per run as `settings.json`. Process
state (the agent loops, desktop provider, live-view references, CDP browser manager, operator
controls) lives on one `OperonRuntime` (`src/operon/core/runtime.py`, `current_runtime()`), and
each run's own state (override hint, auto-run task, start/finish times) on its `RunContext`
there (KAN-156): don't add module-level globals; tests patch `current_runtime()` fields.

`dashboard/` is Vercel's root directory, so don't move it.

Browser modes: **batch** launches an isolated Chromium (no CDP, no debug port); **observable**
owns CDP on port 9222 and streams JPEG frames over `/ws/stream`.

## Invariants

These are load-bearing — violating them silently breaks the agent.

- **Vision-only by default.** With default settings, no DOM, CSS selectors, XPath,
  accessibility tree, or Playwright `locator()` in the perception/policy path. All
  targeting goes through `UIElement` coordinates from perception output. Improving
  perception is the intended fix for bad targeting.
  Exception: the opt-in perception modes (`OPERON_PERCEPTION=a11y|dom`, default off)
  may read the accessibility tree or DOM, only inside their own perception backend.
  They must still emit `UIElement` coordinates, and no code outside that backend may
  depend on them.
- **One harness: the thin harness (KAN-183).** The Full harness (rules, guards, step critic,
  retries) was removed. `agent/policy/thin.py` is the only policy: the model's action is never
  replaced, retried or reset by the loop, and its own "done" ends the run. Fix failures by
  improving what the model sees or the executor, not by adding rules or guards around the model.
  The safety gates (trust gate, OCR trust gate, file guard, permission modes, confidence gate,
  coordinate-thrash pause, max steps, secret-type gate, visual servo) are not part of the harness and stay.
- **No scenario hardcoding.** Never branch on a specific site, task, or benchmark
  in core policy.
- **Atomic TYPE.** The executor merges focus+type into one call. Policy must never
  emit a CLICK immediately followed by a TYPE on the same element.
- **Visual servo.** On the desktop executors (`DesktopExecutor` and `LinuxDesktopExecutor`;
  the browser executor has none; OSWorld runs its own fixed-threshold check), `_region_has_content()` runs before every click with an
  adaptively calibrated threshold. Never bypass it.
- **Verification states.** SUCCESS / FAILURE / UNCERTAIN / PENDING /
  PROGRESSING_STABLE / STABLE_WAIT. The thin harness's `OwnDoneVerifier` emits SUCCESS, FAILURE
  and UNCERTAIN; PENDING, PROGRESSING_STABLE and STABLE_WAIT are no longer emitted by any
  verifier, and their handlers were removed. The values are kept only so old reports load.
- **Lazy `ws_stream`.** Never import `ws_stream` at module load in `browser/manager.py`
  — use the `_ws_stream()` accessor to avoid a circular import.
- **Experimental work ships behind an env flag defaulting to off.** Follow the
  `OPERON_SETTLE_DETECT`, `OPERON_CHANGE_NOTE` pattern (the trust gate graduated to on by default); larger proposals
  get an RFC in `docs/rfcs/`.
- **`.env.example` is a contract.** Every variable in it must actually be read by
  code in `src/`, and new settings must be added there (and to `Settings`).

## Dev workflow (Opus plans, Haiku scans, Sonnet codes, Opus verifies)

For any code change, follow the `pipeline` skill (`/pipeline <task>`): Opus plans, Haiku
scanners find, Sonnet coders fix in isolated worktrees, a fresh Opus verifier checks.
The main session does not edit code. Never merge until Vickey writes "merge" for that PR.

Worktrees have no `.venv` of their own and the editable install points at the main checkout's
`src/`, so in a worktree run the main checkout's venv by absolute path, from the worktree root, with `PYTHONPATH=src` (PowerShell:
`$env:PYTHONPATH = "src"`) so tests import the worktree's code.
Coder worktrees branch from the default branch (`main`), not from the session's current branch, so for work on an
unmerged branch tell the coder to check that branch out in its worktree first.

## Runtime artifacts

Runs write to `.var/runs/<run_id>/` (`~/.operon/runs/<run_id>/` when installed with pip) (screenshots, model I/O, policy decisions,
execution traces, logs) — gitignored, and **screenshots capture the whole screen**,
so review before sharing. Legacy `runs/`, `.browser-artifacts/`, `.desktop-artifacts/`
paths still appear in old logs and tests.

Never commit anything under `.claude/` except `.claude/agents/` (the shared subagent definitions, see Dev workflow) —
the rest is gitignored and personal.

## Testing conventions

New behavior needs a test that runs
without a live server or a real screen — if that's genuinely impossible, mark it
`@pytest.mark.live_server` and keep the code path thin, since CI won't cover it.
`OPERON_TEST_SAFE_MODE=true` skips display baseline and servo calibration.

## Commits

Short imperative subjects with a prefix: `Fix:`, `Feat:`, `Docs:`, `CI:`, `Chore:`,
`Refactor:`. One line, specific.