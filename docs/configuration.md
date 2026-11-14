# Configuration

Operon reads its settings from environment variables, usually through a `.env` file.
[`.env.example`](../.env.example) lists every setting the server reads, with comments; copy
it to `.env` to start.

## Model keys

`GEMINI_API_KEY` runs Gemini, Operon's only model for now: the browser and desktop backends
(Gemini computer use), the browser fallback (full harness only), the step critic and the
benchmark judge. The Gemini JSON backends also accept `GOOGLE_API_KEY`; Gemini Computer Use
reads `GEMINI_API_KEY` only. Other models will come back later through a simpler model
onboarding ([roadmap](roadmap.md)).

## Harness profiles

`OPERON_HARNESS_PROFILE` sets how much scaffolding surrounds the model.

- **full** (the default, and the only profile for `json` backends): deterministic rules
  before the model, post-model guards, a per-step critic and a recovery ladder.
- **thin** (`OPERON_HARNESS_PROFILE=thin` with a `computer_use` backend, browser or desktop):
  Gemini makes every decision. No rule engine or guard replaces its action, and its own
  "done" ends the run, so the benchmark graders decide success. The trust gate, allowed file
  roots, coordinate-thrash pause and step budget still apply.
- **same-model critic** (`OPERON_CRITIC=same`, full harness only): the per-step critic calls
  the acting model instead of the `OPERON_*_VERIFIER_MODEL`. Reports record the critic as
  `critic` (`same:<model>` or `gemini:<model>`).

## All settings

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Gemini API key (Gemini Computer Use reads `GEMINI_API_KEY` only) |
| `OPERON_CONFIG_DIR` | `~/.operon` | Per-user folder for `operon.env`, where `operon setup` saves the key. Every `operon` command and the API read it after `.env`; a value in `.env` or the environment wins. |
| `GEMINI_API_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/models` | Base URL for Gemini calls, e.g. a proxy. The Gemini Computer Use client always uses it; the JSON Gemini client ignores it when `OPERON_USE_VERTEX=true` and calls Vertex AI instead. |
| `API_KEYS` | unset | Comma-separated API keys. Every HTTP route except `/health` requires a matching `X-API-Key` header, and the `/ws/stream` WebSocket requires it as an `x-api-key` subprotocol (`["x-api-key", "<key>"]`). Unset = the server generates one key on first start in `.var/api-key` (mode 0600), which the benchmark harness reads. |
| `OPERON_ALLOW_UNAUTHENTICATED` | `false` | With `API_KEYS` unset, `true` turns authentication off. Only safe on a throwaway machine, since the API can drive a real browser/desktop. |
| `OPERON_ALLOWED_HOSTS` | unset | Extra host names the API answers to. Loopback names are always allowed; any other `Host` gets 400, which blocks DNS rebinding. |
| `CORS_ORIGINS` | empty | Comma-separated exact origins (`scheme://host:port`) allowed to call the API from a browser, e.g. `http://localhost:3000`. `http://127.0.0.1:3000` is a different origin and needs its own entry. Also gates `/ws/stream`, which otherwise accepts only clients that send no `Origin` and loopback pages served from the API's own host and port. `*` allows any origin over HTTP but not on `/ws/stream`. Empty = no cross-origin access. |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use` (Gemini computer use) or `json` (Gemini JSON) |
| `OPERON_BROWSER_FALLBACK_BACKEND` | `json` | Browser fallback backend. Ignored under the thin harness |
| `OPERON_DESKTOP_BACKEND` | `computer_use` | `computer_use` (Gemini computer use) or `json` (the older Gemini JSON path) |
| `OPERON_ENVIRONMENT` | `live` | Where desktop runs happen (KAN-80). `live` is this machine's own screen; `operon envs` lists every environment and whether it can run here, and `operon run --env NAME` picks one for a single local run. See [environments.md](environments.md). |
| `OPERON_BROWSER_MODEL` / `OPERON_DESKTOP_MODEL` | `gemini-3.5-flash` | Primary model |
| `OPERON_BROWSER_FALLBACK_MODEL` / `OPERON_DESKTOP_FALLBACK_MODEL` | `gemini-3.5-flash` | Fallback (JSON) model |
| `OPERON_BROWSER_THINKING` / `OPERON_DESKTOP_THINKING` | `low` | Thinking level (Gemini 3 levels) |
| `OPERON_BROWSER_VERIFIER_MODEL` / `OPERON_DESKTOP_VERIFIER_MODEL` | `gemini-3.5-flash` | Full-harness step critic model |
| `OPERON_BROWSER_VERIFIER_THINKING` / `OPERON_DESKTOP_VERIFIER_THINKING` | `low` | Critic thinking level |
| `OPERON_HARNESS_PROFILE` | `full` | `thin` switches a `computer_use` backend to the thin harness (judged by its own "done"). `json` backends always run `full`. |
| `OPERON_CRITIC` | (unset) | `same`: the full harness's step critic runs on the acting model instead of the `OPERON_*_VERIFIER_MODEL` |
| `BROWSER_HEADLESS` | `false` | Browser headed/headless behavior |
| `BROWSER_WIDTH` | `1920` | Browser viewport width |
| `BROWSER_HEIGHT` | `1080` | Browser viewport height |
| `OPERON_RUNTIME_ROOT` | `.var` (checkout) or `~/.operon` (pip install) | Root for generated runtime artifacts: runs, screenshots, the local API key. A source checkout uses its `.var/`; a pip install uses `OPERON_CONFIG_DIR` |
| `OPERON_RUNS_ROOT` | `.var/runs` | Override run state and step artifact storage |
| `OPERON_BROWSER_ARTIFACTS_ROOT` | `.var/browser-artifacts` | Override browser recording/session artifacts |
| `OPERON_DESKTOP_ARTIFACTS_ROOT` | `.var/desktop-artifacts` | Override desktop recording/session artifacts |
| `OPERON_TEST_ARTIFACTS_ROOT` | `.var/test-artifacts` | Override test artifact output |
| `OPERON_TRACE` | unset | Set `1` for loop trace logging |
| `OPERON_TEST_SAFE_MODE` | `false` | Skips display baseline and servo calibration in tests |
| `OPERON_BESTOFN_N` | `1` | Experimental (RFC 0001 Move 3): sample N candidate actions at uncertain steps and execute the best-scored one. `1` disables (default). |
| `OPERON_BESTOFN_CONFIDENCE` | `1.0` | Experimental: only sample when a step's policy confidence is below this ceiling (lower to restrict Best-of-N to uncertain steps). |
| `OPERON_GROUNDER` | `deterministic` | Experimental (RFC 0001 Move 2): grounder backend. `deterministic` (default, baseline) or `snap` (snap a near-miss raw coordinate onto the nearest interactable element). |
| `OPERON_TOOLS` | `on` | On by default since 2026-11-14 ([RFC 0002](rfcs/0002-tool-action-channel.md), KAN-79): tools the Gemini computer-use model may call next to its screen actions, each answered with its output. `off` = no tools; `on` = `read_file`, `list_dir`, `write_file`, `http_get`; `all` adds `run_shell`; or a comma list of tool names. Relative paths and the shell's working folder are the home folder. File tools obey the file guard (`OPERON_ALLOWED_FILE_ROOTS`), `http_get` the domain allowlist, and with the trust gate on every `run_shell` needs approval. |
| `OPERON_ZOOM` | `off` | Experimental (KAN-82): zoom before clicking. The model's proposed click is refined in one extra one-shot request that shows it only a crop around that point (with what it said it was clicking); its click on the crop is mapped back to the screen. `dense` zooms on screens whose longer side is at least `OPERON_ZOOM_MIN_SIDE` pixels, `always` on every click (for measuring), `off` never. A failed or empty zoom request keeps the original click. Gemini computer-use backends (browser and desktop). Measure with `python -m evals.grounding --zoom dense`. |
| `OPERON_ZOOM_MIN_SIDE` | `2000` | With `OPERON_ZOOM=dense`: the screen's longer side, in pixels, from which a screen counts as dense. |
| `OPERON_ZOOM_FACTOR` | `2.0` | The zoom crop is 1/factor of the screen on each side (clamped to 1.5 to 6). |
| `OPERON_PROGRESS_REVISIT` | `off` | Experimental ([RFC 0003](rfcs/0003-repeated-action-progress.md)), full harness only: a repeated action counts as progress only if its after-screen differs (by at least 1%) from the screens the same action produced earlier in the run, so a model relaunching the same app is stopped by the existing loop guards instead of running to `max_steps`. A missing screenshot never counts as the same screen. |
| `OPERON_TRUST_GATE` | `on` | Pre-execution deny/confirm policy gate (RFC 0001 Move 5). On by default with the built-in confirm list, an empty deny list and no domain allowlist, so the run pauses for approval before a high-risk action such as `place order`. `off` (also `false`, `0`, `no`) allows every action; the live benchmarks and their CI workflows set it to `off` so scores stay comparable with earlier runs, and a benchmark report records it as `trust_gate`. When it pauses a run, **Resume** (or Proceed) approves that exact action once: the next step runs it if the policy picks it again. Resuming with a correction hint does not approve it. |
| `OPERON_TRUST_DENY` | unset | Comma-separated phrases that **block** an action (the run pauses for a human; the action never runs autonomously). Matched case-insensitively against the action text, URL, and target-element name. |
| `OPERON_TRUST_CONFIRM` | built-in set | Comma-separated phrases that require **human approval** before running. Defaults to a conservative high-risk set (`delete account`, `permanently delete`, `deactivate`, `close account`, `confirm payment`, `place order`, `buy now`, `complete purchase`, `send money`, `transfer funds`, `uninstall`) when this is unset. |
| `OPERON_TRUST_ALLOW_DOMAINS` | unset | Comma-separated domains (subdomains included). When set, navigation is **restricted to these domains** — any other URL is denied (fail-closed). Unset = no domain restriction. |
| `OPERON_PERMISSION_MODE` | `risky` | Permission mode for a run that does not choose one (the task page and `operon run --mode` choose per run). `ask` approves every action that does something, even with the gate off. `risky` uses the gate's confirm list. `auto` approves nothing. Deny rules and the domain allowlist block in every mode. |
| `OPERON_STREAM_TOKEN_TTL_S` | `600` | Lifetime of the short-lived `/ws/stream` tokens from `POST /app/api/stream-token` (10 to 86400 s). Checked at connect, so an open stream is not cut off. |
| `OPERON_API_URL` | `http://127.0.0.1:8080` | Where `operon run --request` sends the task. |
| `OPERON_ALLOWED_FILE_ROOTS` | unset | Extra directories that `read_text` / `upload_file` may touch. The file guard is always on: any path not named in the task or inside the run directory pauses the run for a human. |
| `OPERON_SITE_ADAPTERS` | unset | Built-in site adapters to load (`wikipedia`, `github`, or `all`) |
| `OPERON_RUNS_RETAIN_DAYS` | `14` | Days of run directories kept by the startup sweep; `-1` disables it |
| `HITL_WEBHOOK_URL` | unset | POST endpoint called when a run pauses for a human |

`.env.example` lists every setting the server reads, with comments.

Generated runtime output is intentionally kept under `.var/` and ignored by git. Legacy `runs/`, `.browser-artifacts/`, and `.desktop-artifacts/` paths may still appear in old logs or tests, but new default runtime output uses the `.var/` tree unless overridden by environment variables.
