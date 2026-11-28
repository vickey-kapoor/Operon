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
.venv\Scripts\operon keys set NAME        # KAN-186: model API key in the OS credential store (Windows Credential Manager); also: keys list / keys rm NAME / keys import; see docs/secrets.md
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

# Red test (KAN-280; CI "Tests fail without the fix"): the PR's changed tests must fail on main's code
.venv\Scripts\python scripts/red_test_check.py --base origin/main

# Type check (KAN-159; CI "Type check" job). Baseline modules are listed in [tool.mypy] in
# pyproject.toml: fix one, delete it from the list. Coverage: add --cov to the pytest command.
.venv\Scripts\mypy

# Security checks (the Security workflow, KAN-34); each is a CI check on every PR
uvx zizmor==1.32.0 --offline --no-progress --min-severity medium .github/workflows
uv export --locked --all-extras --no-hashes --no-emit-project -o $env:TEMP\requirements.txt   # then:
uvx pip-audit==2.10.1 -r $env:TEMP\requirements.txt --no-deps --disable-pip
uvx --python 3.14 bandit==1.9.4 -r src -ll -q -b .bandit-baseline.json -f txt   # baseline compare needs Linux/WSL (forward-slash paths)
npm audit --prefix dashboard --package-lock-only --audit-level=high
gitleaks git --no-banner --redact .      # git mode is the authoritative check (what CI runs); `gitleaks dir` skips allowlisted paths whole and scans .claude/worktrees and __pycache__ copies, so don't rely on it
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
  coordinate-thrash pause, max steps, word-only stop, secret-type gate, visual servo) are not part of the harness and stay.
- **No scenario hardcoding.** Never branch on a specific site, task, or benchmark
  in core policy.
- **Atomic TYPE.** The executor merges focus+type into one call. Policy must never
  emit a CLICK immediately followed by a TYPE on the same element.
- **Visual servo.** On the desktop executors (`DesktopExecutor` and `LinuxDesktopExecutor`;
  the browser executor has none; OSWorld runs its own fixed-threshold check), `_region_has_content()` runs before every click and every drag start with an
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

## Lessons

Mistakes caught in review, by Codex, or by a failed run, one line each. A PR that fixes such a mistake adds
its lesson here, or, if code can check it, a guard in `tests/test_invariants.py`. Lines marked [guarded: ...] name that guard.

- Linux TYPE pasted with ctrl+v and terminals ignored it -> type short printable ASCII key by key; paste only what has no key (KAN-166, #280).
- Scroll amounts are pixels but pyautogui takes wheel clicks, so pages jumped hundreds of notches -> convert with `wheel_clicks()` in `executor/scroll_units.py` (KAN-171, #346) [guarded: `test_scroll_goes_through_wheel_clicks`].
- NAVIGATE and `start_url` accepted `file:` and other schemes -> every path that opens a URL checks `is_web_url()` first (KAN-191, #359) [guarded for `.goto()`: `test_goto_is_guarded_by_is_web_url`].
- `run_shell` inherited API keys from the environment -> every process the model can start (shell, launched apps) gets `env=launch_env()` (KAN-181, #356) [guarded for `Popen`/`create_subprocess_*` in the executors and tool runner: `test_spawned_processes_use_launch_env`].
- `http_get` could reach private addresses through DNS, redirects, the env proxy and IPv4-mapped IPv6 -> check every resolved IP, `trust_env=False`, `follow_redirects=False` (KAN-181, #356) [client flags guarded: `test_tool_http_client_is_locked_down`].
- The API key was printed, and serve and CORS were open -> never print a key, bind to loopback, refuse `CORS_ORIGINS=*` (KAN-180, #352).
- A key file was readable by other users, and the fix missed files that already existed -> create secret files owner-only and tighten existing ones on update (#336).
- Model clients gave up on 429 and 5xx, and a failed turn stayed in the conversation -> honor `Retry-After`, back off, roll the failed turn back (KAN-173, #348).
- Browser confirm dialogs skipped the trust gate -> a dialog is an action: gate it, and bind the approval to that one dialog (KAN-172, #350).
- Run-cost readers counted a step with no usage file as $0 -> report missing usage as a gap; readers of run files accept null fields (KAN-177, #358).
- A new pre-execution skip ignored a person's gate approval, which the gate stage had already used up -> any check that can stop an action reads the approval the gate stage captured (KAN-207, #381).
- A fresh-look screenshot was awaited after the gate stage's cancel check, so a stop could still let the click run -> any await added between the gate's cancel check and execution rechecks `_run_cancelled` after it (KAN-207, #381).
- A timing test had no margin and failed by 0.03 s on CI -> size test delays at several times the worst measured sample, and note the measurement next to the constant (KAN-211, #364).
- A workflow checkout kept its token in `.git/config` -> `persist-credentials: false` on every `actions/checkout`; run zizmor before pushing workflow edits (#351).
- A test that shells out to bash failed on Windows, where bash may be WSL with no distro -> skip such tests on Windows; offline checks must pass there too (KAN-195, #360).
- A stop step's run.jsonl row named an after screenshot that was never written -> `after_artifact_path` and `executed_action.artifact_path` name only a screenshot the executor actually took; null otherwise (KAN-218, #368; KAN-219).
- `/bench` PR comments run `main`'s copy of the workflow file, not the PR's -> a workflow change takes effect for `/bench` only after merge (found during the #191 restructure).
- The PR-gate label job runs main's copy of `scripts/pr_gate.py`, so a new flag failed the check -> pass new flags only when the base script supports them, until merged (#374).
- A settle frame grab could launch the browser inside the wait's time limit, where a slow start is cancelled half-way -> a frame grab only reads a session that already exists; it never opens one (KAN-203, #380).
- A changed default (no Win+D on your own PC) silently took the clean screen from the live desktop benchmarks -> when a default changes, find every caller that relied on the old one and make it ask explicitly (KAN-205, #379).
- A Google workload identity mapping read an OIDC claim (`job_workflow_ref`) that a directly started run may not carry -> map an optional claim as `"claim" in assertion ? assertion.claim : <fallback>`, and check a direct start and a called run both sign in at apply (KAN-224, #386).
- The pipeline skill was named here but never committed, so cloud story threads planned and coded everything in one Opus session (PRs 369-386) -> the skill lives in `.claude/skills/pipeline/` and every PR body records who did each step (#387, 2026-11-27).
- A reusable workflow's job with `environment:` got an empty key because the caller didn't pass it -> callers pass environment secrets by name (never `secrets: inherit`) (KAN-282, #391) [guarded: `test_every_caller_of_a_paid_workflow_passes_the_gemini_key`, `test_no_caller_inherits_secrets`].

## Dev workflow (Opus plans, Haiku scans, Sonnet codes, Opus verifies)

For any code change, small ones too, follow the `pipeline` skill (`.claude/skills/pipeline/SKILL.md`, `/pipeline <task>`):
Opus plans, Haiku scanners find, Sonnet coders fix in isolated worktrees, a fresh Opus verifier checks.
The main session does not edit code. Each PR body names who did each step ("Who did what").

**Merging (Vickey, 2026-11-26: "Every PR can auto-merge but no paid runs without my explicit approval").** This is the
full rule; AGENTS.md and docs/CLAUDE-CODE-REFERENCE.md point here.
- A PR may merge once its checks are green, every Codex finding is answered, and a fresh verifier has passed on the
  head commit (the main session then posts `<!-- operon-verifier: PASS <full head sha> -->` on the PR).
- The PR gate (`.github/workflows/pr-gate.yml`, `scripts/pr_gate.py`, KAN-216/KAN-227) labels each PR `low-risk`,
  `risky` or `needs-vickey` (the PRs that wait for Vickey's own "merge": every workflow and action file, the paid-run machinery and the merge check, see the Paid-run exception below), and merges ready `low-risk` and `risky` PRs itself when the repo variable `OPERON_AUTO_MERGE`
  is `on` (it is). The gate does the merging; agents never merge by hand, except that a story thread may run
  `gh pr merge --squash --match-head-commit <sha>` only while no open `auto-work-paused` issue exists, the gate has posted its "would merge" note
  (`<!-- operon-gate-trial: <head sha> -->`) for that exact head, the PR is not labelled `needs-vickey`, and the thread
  has first run main's `scripts/pr_gate.py classify` on the PR's files itself and got `low-risk` or `risky` (a same-repo
  PR's own CI could post a fake note or remove labels, so the note and labels alone prove nothing). A fork PR is
  never merged by the gate. Never enable GitHub's own auto-merge (it is off repo-wide): it only waits for required
  checks and skips the markers.
- While an open issue labelled `auto-work-paused` exists (Vickey wrote "pause"), the gate merges nothing and posts no would-merge notes, so no PR merges, and no agent merges by hand, even with an earlier would-merge note.
- A `risky` PR also needs a review by the top model available in this app (today Fable 5.2), in a fresh session that is
  not the coordinator and did not write or verify the change, so someone checks on Vickey's behalf. Post its verdict on
  the PR verbatim, with the marker, in its own comment. Fix what it finds; once it passes on the head, post
  `<!-- operon-top-review: PASS <full head sha> -->` in that comment; the gate ignores a top-review marker in the same
  comment as the verifier marker. A new commit needs a new review.
- **Paid-run exception (label `needs-vickey`).** Every GitHub workflow or action file waits for Vickey (Vickey,
  2026-11-26), because a workflow can start a paid run on its own; the content of a workflow is not inspected. A PR
  waits for Vickey to write "merge" for that PR, and the other files merge after a Fable 5.2 review.
  - Waits for Vickey (`needs-vickey`): `.github/workflows/**` and `.github/actions/**` (a rename into or out of them
    too); any `.claude/` folder at any depth or file in it other than a plain agent definition `.claude/agents/<name>.md`, and any `.mcp.json`; a plain agent definition also waits if it is not an ordinary file (e.g. a link), if it can't be read or is over 64 KB, if its frontmatter has anything beyond name, description, model, tools, isolation, color and effort (so no hooks or MCP servers; `tools` must be a plain comma-separated list of tool names), if its `name` is not its file name, if a body line is `---` or starts with `hooks:` or `mcpServers:`, and `.claude/agents/verifier.md` always waits (its PASS is a merge marker); the merge check (`scripts/pr_gate.py`, `tests/ci/test_pr_gate.py`); `evals/**` except
    JSON recordings under `evals/recordings/` (so `evals/CLAUDE.md` waits too); `scripts/**` (bench runners and VM
    scripts); cost caps, model prices, ids, defaults and settings (`src/operon/models/usage.py`,
    `src/operon/core/settings.py`, `src/operon/agent/backends/model_adapter/registry.py`,
    `src/operon/agent/backends/model_adapter/models.yaml`, `src/operon/api/runtime_config.py`,
    `src/operon/api/runtime/services.py`, `src/operon/clients/**`,
    `src/operon/agent/backends/model_adapter/transports/**`, `src/operon/api/observer.py`); a package with the same
    name as any listed `.py` file (`usage/` next to `usage.py`); any PR with 3000 or more changed files (the file list
    may be cut short).
  - Risky (Fable 5.2 review), not Vickey: other GitHub settings files (`dependabot.yml`, `CODEOWNERS`, issue and PR
    templates) and the rules files (`CLAUDE.md`, `AGENTS.md`, `docs/CLAUDE-CODE-REFERENCE.md`, any `CLAUDE.md` or
    `AGENTS.md` in a subfolder), plain agent definitions `.claude/agents/<name>.md`.
  - When unsure, it waits. A `low-risk` label never overrides this; keep such a PR as a draft until Vickey says "merge".
- **No paid run without Vickey's explicit OK for that run and its cost:** no `/bench` comment, no manual start of
  `benchmark.yml`, `grounding.yml` or `osworld-bench.yml`, no local run with a real model key, and no PR whose own CI
  would use a model key. Approval is per PR and per run: an earlier "merge", a "yes" to another question, a green check,
  a clean Codex review or a reviewer PASS do not count. Every paid job runs in the `paid-runs` environment and waits for
  Vickey's Approve in GitHub (KAN-282); an agent never approves a waiting `paid-runs` run (no pending_deployments call, no
  bypass), even with her OK for the run. Agents act as Vickey's account, so the Approve button is a hard stop against a
  run starting by accident and a notification to her, but an agent's token could still approve through the API; the
  no-approve rule is what stops that. An agent never changes the `paid-runs` environment (reviewers, branch rule, bypass,
  secrets) and never adds a model key as a repo secret.
- Marker comments come from the same account that opens PRs, so they are only as trustworthy as the review behind them.
  Gate merges use `GITHUB_TOKEN`, so push-triggered CI on `main` does not run for them.

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

Never commit anything under `.claude/` except `.claude/agents/` and `.claude/skills/pipeline/` (the shared subagent definitions and the pipeline skill, see Dev workflow) —
the rest is gitignored and personal.

## Testing conventions

New behavior needs a test that runs
without a live server or a real screen — if that's genuinely impossible, mark it
`@pytest.mark.live_server` and keep the code path thin, since CI won't cover it.
`OPERON_TEST_SAFE_MODE=true` skips display baseline and servo calibration.

## Commits

Short imperative subjects with a prefix: `Fix:`, `Feat:`, `Docs:`, `CI:`, `Chore:`,
`Refactor:`. One line, specific.