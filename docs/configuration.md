# Configuration

Operon reads its settings from environment variables, usually through a `.env` file.
[`.env.example`](../.env.example) lists every setting the server reads, with comments; copy
it to `.env` to start.

## Model keys

- `ANTHROPIC_API_KEY` makes Claude computer use the browser backend, running the thin
  harness. Set `OPERON_DESKTOP_BACKEND=claude_computer_use` to use Claude on the desktop too.
- `GEMINI_API_KEY` runs the Gemini backends: the browser backend when there is no Anthropic
  key, the browser fallback, and the default desktop backend. The benchmark judge also uses
  it. The Gemini JSON backends also accept `GOOGLE_API_KEY`; Gemini Computer Use reads
  `GEMINI_API_KEY` only.

## Harness profiles

`OPERON_HARNESS_PROFILE` sets how much scaffolding surrounds the model.

- **thin** (the default for `claude_computer_use`): Claude makes every decision. No rule
  engine or post-model guard replaces its action, failures go back to Claude as the result
  of its tool call, and Claude checks the outcome once against the task when it says it is
  done. The trust gate, allowed file roots, coordinate-thrash pause and step budget still
  apply.
- **full** (always used for Gemini backends; `OPERON_HARNESS_PROFILE=full` opts Claude in):
  deterministic rules before the model, post-model guards, a per-step verifier and a
  recovery ladder.

On the live browser benchmark the thin harness matched or beat the full one for Claude
(42/44 at 28.6 s per task against 42/44 at 31 s). [architecture.md](architecture.md) has
the details.

## All settings

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Gemini API key (Gemini Computer Use reads `GEMINI_API_KEY` only) |
| `ANTHROPIC_API_KEY` | unset | Anthropic key; when set, browser runs default to Claude computer use |
| `API_KEYS` | unset | Comma-separated API keys. Every HTTP route except `/health` requires a matching `X-API-Key` header, and the `/ws/stream` WebSocket requires it as an `x-api-key` subprotocol (`["x-api-key", "<key>"]`). Unset = the server generates one key on first start in `.var/api-key` (mode 0600), which the UI and the benchmark harness read. |
| `OPERON_ALLOW_UNAUTHENTICATED` | `false` | With `API_KEYS` unset, `true` turns authentication off. Only safe on a throwaway machine, since the API can drive a real browser/desktop. |
| `OPERON_ALLOWED_HOSTS` | unset | Extra host names the API answers to. Loopback names are always allowed; any other `Host` gets 400, which blocks DNS rebinding. |
| `OPERON_BROWSER_BACKEND` | `claude_computer_use` with `ANTHROPIC_API_KEY`, else `computer_use` | `claude_computer_use` (Claude), `computer_use` (Gemini) or `json` |
| `OPERON_BROWSER_FALLBACK_BACKEND` | `json` | Optional browser fallback backend |
| `OPERON_DESKTOP_BACKEND` | `json` | Desktop backend |
| `OPERON_BROWSER_MODEL` | `gemini-3.5-flash` | Browser primary model |
| `OPERON_BROWSER_THINKING` | `low` | Computer Use thinking level (Gemini 3) |
| `OPERON_BROWSER_VERIFIER_THINKING` | `low` | Browser critic thinking level |
| `OPERON_BROWSER_CLAUDE_MODEL` | `claude-opus-5-5` | Model for `claude_computer_use` |
| `OPERON_BROWSER_CLAUDE_EFFORT` | `low` | Claude effort level for `claude_computer_use` |
| `OPERON_DESKTOP_MODEL` | `gemini-2.5-flash` | Desktop primary model (Gemini JSON backend) |
| `OPERON_DESKTOP_CLAUDE_MODEL` | `claude-opus-5-5` | Model for the desktop `claude_computer_use` backend |
| `OPERON_DESKTOP_CLAUDE_EFFORT` | `low` | Claude effort level on the desktop |
| `OPERON_HARNESS_PROFILE` | `thin` for Claude | `full` switches a `claude_computer_use` backend back to the full harness. Other backends always run `full`. |
| `OPERON_FINAL_CHECK_MODEL` | the acting Claude model | Model for the thin harness's completion check; a cheaper model (e.g. `claude-sonnet-5-5`) cuts its cost |
| `BROWSER_HEADLESS` | `false` | Browser headed/headless behavior |
| `BROWSER_WIDTH` | `1920` | Browser viewport width |
| `BROWSER_HEIGHT` | `1080` | Browser viewport height |
| `OPERON_RUNTIME_ROOT` | `.var` | Root for generated runtime artifacts |
| `OPERON_RUNS_ROOT` | `.var/runs` | Override run state and step artifact storage |
| `OPERON_BROWSER_ARTIFACTS_ROOT` | `.var/browser-artifacts` | Override browser recording/session artifacts |
| `OPERON_DESKTOP_ARTIFACTS_ROOT` | `.var/desktop-artifacts` | Override desktop recording/session artifacts |
| `OPERON_TEST_ARTIFACTS_ROOT` | `.var/test-artifacts` | Override test artifact output |
| `OPERON_TRACE` | unset | Set `1` for loop trace logging |
| `OPERON_TEST_SAFE_MODE` | `false` | Skips display baseline and servo calibration in tests |
| `OPERON_BESTOFN_N` | `1` | Experimental (RFC 0001 Move 3): sample N candidate actions at uncertain steps and execute the best-scored one. `1` disables (default). |
| `OPERON_BESTOFN_CONFIDENCE` | `1.0` | Experimental: only sample when a step's policy confidence is below this ceiling (lower to restrict Best-of-N to uncertain steps). |
| `OPERON_GROUNDER` | `deterministic` | Experimental (RFC 0001 Move 2): grounder backend. `deterministic` (default, baseline) or `snap` (snap a near-miss raw coordinate onto the nearest interactable element). |
| `OPERON_TRUST_GATE` | `off` | Experimental (RFC 0001 Move 5): `on` enables the pre-execution deny/confirm policy gate. Off (default) = no behavior change. When it pauses a run, **Resume** (or Proceed) approves that exact action once: the next step runs it if the policy picks it again. Resuming with a correction hint does not approve it. |
| `OPERON_TRUST_DENY` | unset | Comma-separated phrases that **block** an action (the run pauses for a human; the action never runs autonomously). Matched case-insensitively against the action text, URL, and target-element name. |
| `OPERON_TRUST_CONFIRM` | built-in set | Comma-separated phrases that require **human approval** before running. Defaults to a conservative high-risk set (e.g. `delete account`, `place order`) when the gate is on and this is unset. |
| `OPERON_TRUST_ALLOW_DOMAINS` | unset | Comma-separated domains (subdomains included). When set, navigation is **restricted to these domains** — any other URL is denied (fail-closed). Unset = no domain restriction. |
| `OPERON_ALLOWED_FILE_ROOTS` | unset | Extra directories that `read_text` / `upload_file` may touch. The file guard is always on: any path not named in the task or inside the run directory pauses the run for a human. |
| `OPERON_SITE_ADAPTERS` | unset | Built-in site adapters to load (`wikipedia`, `github`, or `all`) |
| `OPERON_RUNS_RETAIN_DAYS` | `14` | Days of run directories kept by the startup sweep; `-1` disables it |
| `HITL_WEBHOOK_URL` | unset | POST endpoint called when a run pauses for a human |

`.env.example` lists every setting the server reads, with comments.

Generated runtime output is intentionally kept under `.var/` and ignored by git. Legacy `runs/`, `.browser-artifacts/`, and `.desktop-artifacts/` paths may still appear in old logs or tests, but new default runtime output uses the `.var/` tree unless overridden by environment variables.
