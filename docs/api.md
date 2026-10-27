# HTTP API

The API is a FastAPI app (`operon.api.server:app`). Start it with:

```bash
python -m uvicorn operon.api.server:app --host 127.0.0.1 --port 8080
```

Every HTTP route except `/health` needs an `X-API-Key` header, and `/ws/stream` takes the key
as an `x-api-key` subprotocol (never in the URL, which is logged). The key is one of `API_KEYS`,
or, when that is unset, the one the server generates on first start in `.var/api-key`:

```bash
curl -H "X-API-Key: $(cat .var/api-key)" http://127.0.0.1:8080/run-task ...
```

Any `Host` other than loopback or `OPERON_ALLOWED_HOSTS` gets 400, which blocks DNS rebinding.
`OPERON_ALLOW_UNAUTHENTICATED=true` turns authentication off, which is only safe on a throwaway
machine. See [configuration.md](configuration.md).

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
GET  /observer/api/export/{run_id}
GET  /observer/api/live-browser/{run_id}

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
