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
.venv\Scripts\operon run --tasks tasks.yaml [--junit report.xml] [--only NAME] [--fail-fast]   # KAN-86: task file suite, docs/task-files.md
.venv\Scripts\operon mcp       # KAN-96: MCP server over stdio (pip install operon-agent[mcp]); docs/mcp.md
# KAN-103 part 2 (OPERON_SECRETS=on): the model types {{secret:NAME}} and never sees the value; the value is asked for hidden
.venv\Scripts\operon secrets set NAME     # also: operon secrets list (names only) / operon secrets rm NAME; see docs/secrets.md
.venv\Scripts\operon envs      # KAN-80: list desktop run environments and whether each is ready here
# Release (KAN-36): push tag v<version> matching pyproject.toml; release.yml uploads operon-agent to PyPI (docs/releasing.md)

# Offline test suite (no network, no browser)
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests -q

# Free desktop end-to-end (KAN-166; CI job "Desktop (Linux container)"): real loop + real XFCE desktop
# in Docker, scripted model ($0). Needs Docker; deselected by default.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/desktop_e2e -m desktop_container -o addopts=""

# Free Windows e2e (KAN-93): real loop + real Windows desktop in Windows Sandbox (~2 min boot), scripted model ($0).
# Needs Windows Sandbox turned on; deselected by default.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/desktop_e2e_windows -m windows_sandbox -o addopts=""
# Free browser end-to-end: real loop + real NativeBrowserExecutor on headless Chromium against a local
# fixture site, scripted model ($0, ~3 min). Needs Playwright's Chromium; deselected by default.
$env:GEMINI_API_KEY = "fake-test-key"; .venv\Scripts\python -m pytest tests/browser_e2e -m browser_e2e -o addopts=""

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
- **Visual servo.** On the desktop executors (`DesktopExecutor` and `LinuxDesktopExecutor`;
  the browser executor has none; OSWorld runs its own fixed-threshold check), `_region_has_content()` runs before every click with an
  adaptively calibrated threshold. Never bypass it.
- **Verification states.** SUCCESS / FAILURE / UNCERTAIN / PENDING /
  PROGRESSING_STABLE / STABLE_WAIT. `STABLE_WAIT` triggers a 200ms re-verify.
  `PROGRESSING_STABLE` is handled (the loop would advance immediately) but reserved: no
  verifier currently emits it.
- **Lazy `ws_stream`.** Never import `ws_stream` at module load in `browser/manager.py`
  — use the `_ws_stream()` accessor to avoid a circular import.
- **Experimental work ships behind an env flag defaulting to off.** Follow the
  `OPERON_BESTOFN_*`, `OPERON_GROUNDER` pattern (the trust gate graduated to on by default); larger proposals
  get an RFC in `docs/rfcs/`.
- **`.env.example` is a contract.** Every variable in it must actually be read by
  code in `src/`, and new settings must be added there (and to `Settings`).

## Dev workflow (Opus plans, Haiku scans, Sonnet codes, Opus verifies)

For code work in Claude Code, the main session (Opus) runs this pipeline. The subagents are
defined in `.claude/agents/` with their model pinned in frontmatter (scanner and verifier also have a
restricted tool list).

1. **Plan (main session, Opus, max effort).** Write the plan as a script: the question, the
   files or areas to scan, and how each fix will be checked. The main session does not edit code.
2. **Scan (`scanner`, Haiku, read-only).** Launch several in parallel, one per area, to find
   candidates with file:line evidence. Scanners only read and search; they never edit.
3. **Code (`coder`, Sonnet).** One coder per fix, each in its own isolated git worktree
   (`isolation: worktree`). Coders write all code and tests and run the relevant tests.
4. **Verify (`verifier`, Opus, independent).** Every finding and every fix gets a fresh verifier
   that did not write it: confirm the issue is real, review the diff against the invariants
   above, and run lint, type check and the relevant tests. Findings that fail verification
   are dropped, not passed on.
5. **Sign off (main session, Opus).** Review only verified findings and fixes, bring the
   worktree changes onto the working branch, run the full offline suite once, and sign off
   before anything is pushed. Codex reviews every PR; never merge until Vickey writes "merge"
   for that PR.

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
