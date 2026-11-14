# HTTP API

The API is a FastAPI app (`operon.api.server:app`). Start it with:

```bash
python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080
```

Every HTTP route except `/health` needs an `X-API-Key` header, and `/ws/stream` takes the key
as an `x-api-key` subprotocol (never in the URL, which is logged). The key is one of `API_KEYS`,
or, when that is unset, the one the server generates on first start: `~/.operon/api-key` when
Operon is installed with pip, `.var/api-key` in a source checkout (see `OPERON_RUNTIME_ROOT`):

```bash
curl -H "X-API-Key: $(cat ~/.operon/api-key)" http://127.0.0.1:8080/run-task ...
```

Any `Host` other than loopback or `OPERON_ALLOWED_HOSTS` gets 400, which blocks DNS rebinding.
`OPERON_ALLOW_UNAUTHENTICATED=true` turns authentication off, which is only safe on a throwaway
machine. See [configuration.md](configuration.md).

## Task page and `operon run`

First time, run `operon setup`: it asks for your Gemini key (hidden, checked with a free
model-list call, saved to `~/.operon/operon.env` readable only by you), installs Chromium and
runs a self-check. `operon setup --check` reruns only the self-check; `--new-key` replaces the
key; `--offline` and `--skip-browser` skip the key check and the browser.

`operon serve` starts the API and prints the task page's link, with the key in the URL
fragment (`#key=...`, which the browser never sends to the server). On
`http://127.0.0.1:8080/app` you type a task, pick Browser or Desktop and a permission mode,
then watch it run:

- **Live view.** Browser runs stream frames over `/ws/stream`; desktop runs show each step's
  screenshot.
- **Approvals.** When the run pauses, the page shows why. **Approve** runs the paused action
  once. **Decline** refuses it, and an optional note tells the model what to do instead.
- **Take over.** This pauses the run. While you are in control, Operon takes no screenshots
  and makes no model calls. A step already in flight finishes its action first. For a
  browser run, you can click, scroll and type on the live view, or use the browser window
  itself. For a desktop run, you use the real mouse and keyboard. **Hand back** resumes the
  run, and the model is told the screen may have changed. Handing back never approves an
  action that was waiting before you took over.
- **Permission modes.** `ask` approves every action that does something. `risky` (the
  default, or `OPERON_PERMISSION_MODE`) asks only before the trust gate's confirm list.
  `auto` asks for nothing. Deny rules and the domain allowlist block in every mode, and so
  does a vendor safety confirmation. `ask` asks even when `OPERON_TRUST_GATE=off`.
- **Screens.** **Home** takes a task and lists recent runs; **Live run** shows the live view,
  any approval, the steps so far and Pause, Take over and Stop; **Run recording** replays a
  finished run step by step: the screenshot, what it did, why, and how it was checked.
- **Report.** When the run ends, the page shows the outcome, steps, approvals and takeovers,
  and links to its recording, which can be downloaded as a zip.

The page talks to `/app/api/...`, which needs the API key like every other route. The live
stream uses a short-lived token from `POST /app/api/stream-token` instead of the key. The
token is sent as the subprotocol pair `["x-stream-token", token]` and expires after
`OPERON_STREAM_TOKEN_TTL_S` seconds (default 600). A `view` token gets frames and events only.
A `control` token can also send control messages. The page itself is served without the key,
because it holds no secret.

From a terminal:

```bash
operon run "Search Wikipedia for Alan Turing" --url https://www.wikipedia.org --mode ask
operon run --desktop "Open Notepad and type hello"
operon run --request "Search Wikipedia for Alan Turing"   # send to a running API (OPERON_API_URL)
```

`operon run` prints each step and asks in the terminal when the run waits for you: `y`
approves, `n` declines, any other text declines with that note, `t` takes over and `s`
stops. Press Ctrl+C once to take over, and twice to quit. It ends with the report and exits
0 when the task succeeded. With `--request` and no terminal (a script, for example), a run
that waits is left paused for the page and the command exits 2.

## Routes

The active FastAPI routes are registered from `src/operon/api/routes/`.

```text
# Browser run lifecycle
POST /run-task
POST /step
POST /resume
POST /stop
POST /run/{run_id}/stop
POST /run/{run_id}/pause
GET  /run/{run_id}
POST /cleanup
GET  /health

# Desktop run lifecycle
POST /desktop/run-task
POST /desktop/step
POST /desktop/resume
POST /desktop/stop
POST /desktop/run/{run_id}/stop
POST /desktop/cleanup
GET  /desktop/run/{run_id}

# Observer / telemetry
GET  /observer/api/runs
GET  /observer/api/run/{run_id}
GET  /observer/api/artifact
GET  /observer/api/usage
GET  /observer/api/settings          # {"trust_gate": "on"|"off"}, read by the benchmarks
GET  /observer/api/export/{run_id}
GET  /observer/api/live-browser/{run_id}

# Task page (KAN-89)
GET  /app                              # the page; served without the key
POST /app/api/runs                     # {intent, environment, start_url, permission_mode, max_steps, headless}
GET  /app/api/runs?limit=20            # newest runs on this PC (Home and History)
GET  /app/api/runs/{run_id}            # status, what it waits on, recent steps
GET  /app/api/runs/{run_id}/report
POST /app/api/runs/{run_id}/approve
POST /app/api/runs/{run_id}/decline    # {note}
POST /app/api/runs/{run_id}/pause      # pause between steps; approve continues
POST /app/api/runs/{run_id}/takeover
POST /app/api/runs/{run_id}/handback   # {note}
POST /app/api/runs/{run_id}/input      # your click/keys during a takeover (browser)
POST /app/api/runs/{run_id}/stop
POST /app/api/stream-token             # {scope: view|control}

# Live stream
WS   /ws/stream
```

`POST /run-task` creates a browser run and runs it in the background up to `max_steps`
(default 25); poll `GET /run/{run_id}` for its state. `POST /desktop/run-task` only creates
the run, and each `POST /desktop/step` advances it by one step, so a client can pause,
resume or stop it between steps. `/run-task` with `"mode": "observable"` attaches to Chrome
over CDP and streams JPEG frames on `/ws/stream`.

## Examples

```bash
# Browser: starts and runs in the background
curl -X POST http://127.0.0.1:8080/run-task \
  -H 'Content-Type: application/json' \
  -d '{"intent": "Search Wikipedia for Alan Turing and open his article", "start_url": "https://www.wikipedia.org"}'

# Desktop: create the run, then step it
curl -X POST http://127.0.0.1:8080/desktop/run-task \
  -H 'Content-Type: application/json' \
  -d '{"intent": "Open Notepad and type hello world"}'
curl -X POST http://127.0.0.1:8080/desktop/step \
  -H 'Content-Type: application/json' \
  -d '{"run_id": "<run_id from the response>"}'
```

Request fields for both `run-task` routes: `intent` (required), `start_url`, `headless`,
`mode` (`batch` or `observable`) and `max_steps` (1 to 200).
