# Configuration

Operon reads its settings from environment variables, usually through a `.env` file.
[`.env.example`](../.env.example) lists every setting the server reads, with comments; copy
it to `.env` to start.

Inside Operon, every setting is parsed in one place, `operon.core.settings` (KAN-157): code
reads `get_settings().<field>` instead of calling `os.getenv`. Startup (`operon serve`, the
API, and `operon run`) checks the settings once. A value Operon cannot use at all, such as
`BROWSER_WIDTH=wide` or `OPERON_BROWSER_BACKEND=conputer_use`, stops startup with every
problem listed. A value that falls back to its default, such as `OPERON_ZOOM=maybe`, is logged
as a warning. `python -m operon.api.runtime_config` prints the same checks. Each run saves the
settings it started with in `settings.json` in its run folder, with keys shown only as
`"<set>"`. A new setting is a field on `Settings` plus a line in `.env.example`;
`tests/test_settings.py` checks both.

## Model keys

`GEMINI_API_KEY` runs Gemini, Operon's default model: the browser and desktop backends
(Gemini computer use), the browser fallback (full harness only), the step critic and the
benchmark judge. The Gemini JSON backends also accept `GOOGLE_API_KEY`; Gemini Computer Use
reads `GEMINI_API_KEY` only.

### Other models, side by side with Gemini (KAN-168)

Off by default. `OPERON_BROWSER_BACKEND=model` (or `OPERON_DESKTOP_BACKEND=model`) runs the
profile named by `OPERON_MODEL_PROFILE`, a key of `src/operon/agent/backends/model_adapter/models.yaml`: `claude-sonnet-5-5`
(needs `ANTHROPIC_API_KEY`) or `openai-computer-use` (needs `OPENAI_API_KEY`). A model run is
always thin and the model's own "done" ends it, the same as Gemini's thin runs, so scores compare.
Manual batches and `/bench batch` take the cell word `model=<profile>`; the workflows give a
cell only its own provider's key. Startup refuses a missing key or profile.

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
| `OPERON_CONFIG_DIR` | `~/.operon` | Per-user folder for `operon.env`, an optional file that can hold the key. Every `operon` command and the API read it after `.env`; a value in `.env` or the environment wins. |
| `GEMINI_API_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/models` | Base URL for Gemini calls, e.g. a proxy. The Gemini Computer Use client always uses it; the JSON Gemini client ignores it when `OPERON_USE_VERTEX=true` and calls Vertex AI instead. |
| `API_KEYS` | unset | Comma-separated API keys. Every HTTP route except `/health` requires a matching `X-API-Key` header, and the `/ws/stream` WebSocket requires it as an `x-api-key` subprotocol (`["x-api-key", "<key>"]`). Unset = the server generates one key on first start in `.var/api-key` (mode 0600), which the benchmark harness reads. |
| `OPERON_ALLOW_UNAUTHENTICATED` | `false` | With `API_KEYS` unset, `true` turns authentication off. Only safe on a throwaway machine, since the API can drive a real browser/desktop. |
| `OPERON_ALLOWED_HOSTS` | unset | Extra host names the API answers to. Loopback names are always allowed; any other `Host` gets 400, which blocks DNS rebinding. |
| `CORS_ORIGINS` | empty | Comma-separated exact origins (`scheme://host:port`) allowed to call the API from a browser, e.g. `http://localhost:3000`. `http://127.0.0.1:3000` is a different origin and needs its own entry. Also gates `/ws/stream`, which otherwise accepts only clients that send no `Origin` and loopback pages served from the API's own host and port. `*` allows any origin over HTTP but not on `/ws/stream`. Empty = no cross-origin access. |
| `OPERON_BROWSER_BACKEND` | `computer_use` | `computer_use` (Gemini computer use) or `json` (Gemini JSON) |
| `OPERON_BROWSER_FALLBACK_BACKEND` | `json` | Browser fallback backend. Ignored under the thin harness |
| `OPERON_DESKTOP_BACKEND` | `computer_use` | `computer_use` (Gemini computer use) or `json` (the older Gemini JSON path) |
| `OPERON_ENVIRONMENT` | `live` | Where desktop runs happen (KAN-80). `live` is this machine's own screen; `operon envs` lists every environment and whether it can run here, and `operon run --env NAME` picks one for a single local run. `windows-sandbox` runs desktop tasks in a throwaway Windows Sandbox, `linux-container` in a Linux desktop in Docker. See [environments.md](environments.md). |
| `OPERON_SANDBOX_MEMORY_MB` / `OPERON_SANDBOX_START_TIMEOUT` | `4096` / `240` | Windows Sandbox memory, and seconds to wait for its guest agent to start |
| `OPERON_CONTAINER_IMAGE` / `OPERON_CONTAINER_START_TIMEOUT` / `OPERON_CONTAINER_CA_BUNDLE` | built from `docker/linux-desktop` / `120` / unset | Linux container: a prebuilt image instead of building one, seconds to wait for it, and a CA bundle for pip during the build behind a TLS-inspecting proxy |
| `OPERON_OSWORLD_URL` | unset | OSWorld run environment: the task VM's server, e.g. `http://127.0.0.1:5000` (KAN-169) |
| `OPERON_OSWORLD_POST_ACTION_WAIT_S` | `0` (off) | OSWorld run environment: seconds to wait after each input action or app launch (once per batch, also a part-failed one) before the next screenshot, so a closing dialog is gone from it (max `10`) |
| `OPERON_GUEST_TOKEN` | unset | Inside a run environment only: the token `operon guest` requires (else `--token-file`) |
| `OPERON_BROWSER_MODEL` / `OPERON_DESKTOP_MODEL` | `gemini-3.5-flash` | Primary model |
| `OPERON_BROWSER_FALLBACK_MODEL` / `OPERON_DESKTOP_FALLBACK_MODEL` | `gemini-3.5-flash` | Fallback (JSON) model |
| `OPERON_BROWSER_THINKING` / `OPERON_DESKTOP_THINKING` | `low` | Thinking level (Gemini 3 levels) |
| `OPERON_BROWSER_VERIFIER_MODEL` / `OPERON_DESKTOP_VERIFIER_MODEL` | `gemini-3.5-flash` | Full-harness step critic model |
| `OPERON_BROWSER_VERIFIER_THINKING` / `OPERON_DESKTOP_VERIFIER_THINKING` | `low` | Critic thinking level |
| `OPERON_HARNESS_PROFILE` | `full` | `thin` switches a `computer_use` backend to the thin harness (judged by its own "done"). `json` backends always run `full`. |
| `OPERON_CRITIC` | (unset) | `same`: the full harness's step critic runs on the acting model instead of the `OPERON_*_VERIFIER_MODEL` |
| `BROWSER_HEADLESS` | `false` | `true` runs Chromium headless; the default shows the browser window. Set `true` on servers and CI. |
| `BROWSER_WIDTH` | `1920` | Browser viewport width |
| `BROWSER_HEIGHT` | `1080` | Browser viewport height |
| `OPERON_RUNTIME_ROOT` | `.var` (checkout) or `~/.operon` (pip install) | Root for generated runtime artifacts: runs, screenshots, the local API key. A source checkout uses its `.var/`; a pip install uses `OPERON_CONFIG_DIR` |
| `OPERON_RUNS_ROOT` | `.var/runs` | Override run state and step artifact storage |
| `OPERON_BROWSER_ARTIFACTS_ROOT` | `.var/browser-artifacts` | Override browser recording/session artifacts |
| `OPERON_DESKTOP_ARTIFACTS_ROOT` | `.var/desktop-artifacts` | Override desktop recording/session artifacts |
| `OPERON_TEST_ARTIFACTS_ROOT` | `.var/test-artifacts` | Override test artifact output |
| `OPERON_TRACE` | unset | Set `1` for loop trace logging |
| `OPERON_TEST_SAFE_MODE` | `false` | Skips display baseline and servo calibration in tests |
| `OPERON_STRICT_ERRORS` | `false` | Re-raise bugs (`AttributeError`, `NameError`, `AssertionError`, `ImportError`) at the agent loop's recovery sites instead of carrying on; on in the offline test suite. Off, every recovery is logged with its traceback and counted in the run's `recovered_errors` (KAN-158) |
| `OPERON_BESTOFN_N` | `1` | Experimental (RFC 0001 Move 3): sample N candidate actions at uncertain steps and execute the best-scored one. `1` disables (default). |
| `OPERON_BESTOFN_CONFIDENCE` | `1.0` | Experimental: only sample when a step's policy confidence is below this ceiling (lower to restrict Best-of-N to uncertain steps). |
| `OPERON_PARALLEL_RUNS` | `1` | Experimental (KAN-81): how many batch-mode browser runs may step at once. Above `1`, each batch run gets its own executor and isolated Chromium (and its own backend, so its conversation is its own); a start beyond the cap is refused with HTTP 429 and `Retry-After`. Observable (live view) browser runs and all desktop runs stay serialized: they share one CDP port (9222) or one real screen and mouse. `1` is today's behaviour. |
| `OPERON_SETTLE_DETECT` | `off` | Experimental (KAN-110): after each action, watch small local frames until the screen stops changing instead of sleeping a fixed time. No model call. `on` enables. |
| `OPERON_AUTO_CONFIRM` | `off` | Benchmark runs only (KAN-169): `operon run` answers the model's own "require confirmation" pause yes (at most 10 per run), since nobody can click Resume. Trust-gate approvals are never answered. `evals/osworld.py` sets it for its runs; report scores as run with it. |
| `OPERON_SETTLE_MAX_S` | `3.0` | Longest a settle wait may run, in seconds (clamped to 0.5 to 15). Only read when settle detection is on. |
| `OPERON_CHANGE_NOTE` | `off` | Experimental (KAN-111): while the screen settles after an action, write one line of text about what appeared and went away, or is still changing, and send it to the model as a "Note from Operon" with the next screenshot. Text only (no extra images), no model call. Needs the same frame sampling as `OPERON_SETTLE_DETECT`, so `on` turns that on too. |
| `OPERON_FOCUS_GUARD` | `off` | Experimental (KAN-92), Windows desktop only: record the window in front at each screenshot; if a window from another process has taken focus by the time the first action after that screenshot runs (focus that changed while the model was deciding), skip the action and report `focus_stolen` so the next step re-observes. Windows the agent opened itself (same process) are fine. Nothing is switched back. |
| `OPERON_DPI_AWARE` | `off` | Experimental (KAN-90), Windows desktop only: set the process to per-monitor-v2 DPI awareness before `pyautogui` is imported (pyautogui otherwise calls `SetProcessDPIAware` on import, which locks the process at system-aware and virtualizes mixed-DPI or non-system-scale monitors). Capture, monitor rects and clicks are then all physical pixels and nothing is converted. Needs the executor module to load first in the process (the API server and `operon run` do). If something else already fixed the awareness, one warning says so. Independent of the flag, startup warns once when the process is DPI-unaware, or system-aware while a monitor's true DPI differs from the system's, and the run's `displays.json` records the awareness plus each monitor's true rect, DPI and scale. |
| `OPERON_PLANNER` | `off` | Experimental (KAN-112): at task start a cheaper model writes a short plan (2 to 8 steps) from the task text alone, and the acting model gets it as a "Note from Operon" on its first step. One extra text-only call per run, no re-planning; the plan and its cost are saved as `plan.json` in the run folder. Any failure means no plan. |
| `OPERON_TASK_NOTE` | `off` | Experimental (KAN-102): a running note for long tasks, written by the loop with no model call: the task, the plan steps (when `OPERON_PLANNER` ran), the last few actions each with its outcome ("typed 'hello' at (412,220) -> success"), and the step count. It goes to the model as a "Note from Operon" with every step, thin harness included (it changes what the model sees, not what it does), and is saved as `task_note.txt` in each step folder. Capped at about 1200 characters, oldest lines dropped first. |
| `OPERON_TASK_NOTE_MAX_ITEMS` | `8` | How many recent actions the task note lists (1 to 50). |
| `OPERON_CHECKPOINTS` | `off` | Experimental (KAN-80): resume runs after a restart. When the API starts, runs the last process left running, pending or waiting are marked interrupted (status `waiting_for_user`, stop reason `interrupted`). Nothing resumes by itself: `POST /run/{id}/resume-interrupted`, `operon runs --interrupted` and `operon resume <run_id>` continue one from the current screen, with the saved steps as a hint instead of the lost conversation. The step budget continues from the saved step count. See [api.md](api.md). |
| `OPERON_PLANNER_MODEL` | `gemini-3.5-flash-lite` | The model that writes the plan. |
| `OPERON_EXPECT_CHECK` | `off` | Experimental (KAN-116): the model is asked to write one line, `Expect: <none/focus/content/navigation/dialog> - <what it will see>`, before each action. After the action runs, that is compared with how much of the screen changed (for example, content or navigation expected but the screen did not change), and a mismatch is sent to the model as a "Note from Operon" with the next screenshot. Text only, no model call; it never changes a verification result, a retry or an action. The result is saved as `expect_check.json` in the step folder for each checked step; a step is not checked when its action failed or does not touch the screen (stop, wait, tool calls), a JS dialog was auto-accepted, the run was stopped, the step was retried, a rule (not the model) chose the action, or a screenshot is missing. The JSON (full harness) backend already returns `expected_change`, so it uses that field instead of the `Expect:` line and is not asked for one. |
| `OPERON_GROUNDER` | `deterministic` | Experimental (RFC 0001 Move 2): grounder backend. `deterministic` (default, baseline) or `snap` (snap a near-miss raw coordinate onto the nearest interactable element). |
| `OPERON_TOOLS` | `on` | On by default since 2026-11-14 ([RFC 0002](rfcs/0002-tool-action-channel.md), KAN-79): tools the Gemini computer-use model may call next to its screen actions, each answered with its output. `off` = no tools; `on` = `read_file`, `list_dir`, `write_file`, `http_get`; `all` adds `run_shell`; or a comma list of tool names. Relative paths and the shell's working folder are the home folder. File tools obey the file guard (`OPERON_ALLOWED_FILE_ROOTS`), `http_get` the domain allowlist, and with the trust gate on every `run_shell` needs approval. |
| `OPERON_ZOOM` | `dense` | KAN-82: zoom before clicking, on by default for dense screens only (ScreenSpot-Pro grounding 50% to 68%; screens under `OPERON_ZOOM_MIN_SIDE` are untouched). The benchmark workflows pass `off` unless a run asks for zoom, so scores stay comparable. The model's proposed click is refined in one extra one-shot request that shows it only a crop around that point (with what it said it was clicking); its click on the crop is mapped back to the screen. `dense` (default) zooms only on screens whose longer side is at least `OPERON_ZOOM_MIN_SIDE` pixels (2000 by default, so a 1920x1080 screen is never zoomed), `always` on every click (for measuring), `off` never. A failed or empty zoom request keeps the original click. Gemini computer-use backends (browser and desktop). Measure with `python -m evals.grounding --zoom dense`. |
| `OPERON_ZOOM_MIN_SIDE` | `2000` | With `OPERON_ZOOM=dense`: the screen's longer side, in pixels, from which a screen counts as dense. |
| `OPERON_ZOOM_FACTOR` | `2.0` | The zoom crop is 1/factor of the screen on each side (clamped to 1.5 to 6). |
| `OPERON_PROGRESS_REVISIT` | `off` | Experimental ([RFC 0003](rfcs/0003-repeated-action-progress.md)), full harness only: a repeated action counts as progress only if its after-screen differs (by at least 1%) from the screens the same action produced earlier in the run, so a model relaunching the same app is stopped by the existing loop guards instead of running to `max_steps`. A missing screenshot never counts as the same screen. |
| `OPERON_TRUST_GATE` | `on` | Pre-execution deny/confirm policy gate (RFC 0001 Move 5). On by default with the built-in confirm list, an empty deny list and no domain allowlist, so the run pauses for approval before a high-risk action such as `place order`. `off` (also `false`, `0`, `no`) allows every action; the live benchmarks and their CI workflows set it to `off` so scores stay comparable with earlier runs, and a benchmark report records it as `trust_gate`. When it pauses a run, **Resume** (or Proceed) approves that exact action once: the next step runs it if the policy picks it again. Resuming with a correction hint does not approve it. |
| `OPERON_TRUST_DENY` | unset | Comma-separated phrases that **block** an action (the run pauses for a human; the action never runs autonomously). Matched case-insensitively against the action text, URL, and target-element name. |
| `OPERON_TRUST_CONFIRM` | built-in set | Comma-separated phrases that require **human approval** before running. Defaults to a conservative high-risk set (`delete account`, `permanently delete`, `deactivate`, `close account`, `confirm payment`, `place order`, `buy now`, `complete purchase`, `send money`, `transfer funds`, `uninstall`) when this is unset. |
| `OPERON_TRUST_ALLOW_DOMAINS` | unset | Comma-separated domains (subdomains included). When set, navigation is **restricted to these domains** — any other URL is denied (fail-closed). Unset = no domain restriction. |
| `OPERON_TRUST_CATEGORIES` | `off` | Experimental (KAN-103). When on, a click whose element label names a risk category (computer-use backends report no element label, so for them see the stated-reason rule below) pauses for approval, and the approval request carries a `category` (`waiting.category` on the front-door status). Categories: `payment` (pay, buy, purchase, checkout, place order, subscribe, transfer, wire, donate), `send` (send, post, publish, submit, reply all, share), `delete` (delete, remove, erase, wipe, uninstall, format disk, format drive, empty trash, permanently), `account` (sign out of all, change password, close account, revoke), `install` (install, run as administrator, allow, grant access). Matching is case-insensitive on word boundaries (so "Pay now" matches, "Payday", "Payments history" and "Resend code" do not) a label is judged only when it is at most 5 words, and labels starting with don't / do not / cancel never match. With no label, the model's stated reason is judged only for a quoted target of at most 5 words (`Click 'Pay now'`, judged like a label) or a multi-word phrase (place order, buy now, pay now, confirm payment, send money, wire transfer, transfer funds, empty trash, delete permanently, permanently delete, delete account, reply all, sign out of all, change password, close account, grant access, run as administrator, install now); bare words in prose never fire, and press_key/hotkey consult the reason only for Enter/Return. Typed text and URLs are never judged. Like the confirm list, it is skipped in `auto` mode and when the gate is off (the benchmarks). |
| `OPERON_TRUST_CATEGORY_LIST` | all five | Comma-separated categories enforced when `OPERON_TRUST_CATEGORIES` is on (`payment,send,delete,account,install`); unknown names are ignored. |
| `OPERON_SECRETS` | `off` | Experimental (KAN-103). When on, the model may type `{{secret:NAME}}` and the executor fills in the value stored with `operon secrets set NAME`; the model is told the names, never the values, and typing one asks for approval (not skipped in `auto` mode; skipped only on a page whose domain is on `OPERON_TRUST_ALLOW_DOMAINS`). Off, a placeholder fails the action instead of being typed. See `docs/secrets.md`. |
| `OPERON_CLICK_TEXT` | `off` | Experimental (KAN-163): under Computer Use a click is only coordinates, so the trust gate cannot see what it presses. With this on, before a click that names no element the gate crops about 240x60 px around the click point from the step screenshot and reads it with local OCR (RapidOCR on the CPU, about 10 ms, no model call). If that text holds a deny or confirm phrase, the run pauses for approval; it never blocks outright, since the crop can catch a neighbouring label. Skipped in `auto` mode and for clicks inside a batch. Needs `pip install -e ".[ocr]"`; without it the gate works as before and logs one warning. |
| `OPERON_PERMISSION_MODE` | `risky` | Permission mode for a run that does not choose one (the task page and `operon run --mode` choose per run). `ask` approves every action that does something, even with the gate off. `risky` uses the gate's confirm list. `auto` approves nothing. Deny rules and the domain allowlist block in every mode. |
| `OPERON_STREAM_TOKEN_TTL_S` | `600` | Lifetime of the short-lived `/ws/stream` tokens from `POST /app/api/stream-token` (10 to 86400 s). Checked at connect, so an open stream is not cut off. |
| `OPERON_API_URL` | `http://127.0.0.1:8080` | Where `operon run --request` sends the task. |
| `OPERON_ALLOWED_FILE_ROOTS` | unset | Extra directories that `read_text` / `upload_file` may touch. The file guard is always on: any path not named in the task or inside the run directory pauses the run for a human. |
| `OPERON_SITE_ADAPTERS` | unset | Built-in site adapters to load (`wikipedia`, `github`, or `all`) |
| `OPERON_RUNS_RETAIN_DAYS` | `14` | Days of run directories kept by the startup sweep; `-1` disables it |
| `HITL_WEBHOOK_URL` | unset | POST endpoint called when a run pauses for a human |

`.env.example` lists every setting the server reads, with comments.

Generated runtime output is intentionally kept under `.var/` and ignored by git. Legacy `runs/`, `.browser-artifacts/`, and `.desktop-artifacts/` paths may still appear in old logs or tests, but new default runtime output uses the `.var/` tree unless overridden by environment variables.
