# UI Navigator MVP

UI Navigator is a browser-only, reliability-first computer-use agent focused on one benchmark workflow:

- Gmail draft creation only
- Stop before send
- Local-only persistence
- No dashboard
- No desktop mode
- No extension runtime
- No cloud storage

The control loop is the center of the system:

`capture -> perceive -> update state -> choose action -> execute -> verify -> recover`

## Runtime

Use Python `3.11` for this repo.

Recommended local virtual environment:

- [`.venv311sys`](./.venv311sys)

Gemini credentials must be provided through environment variables or a local `.env` file. The benchmark path and FastAPI app both load `.env` before creating `GeminiHttpClient`.

## Windows Playwright Setup

Use repo-local temp and browser cache paths when running Playwright:

```powershell
$env:TEMP=".\.tmp"
$env:TMP=".\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH=".\.ms-playwright"
```

Install Chromium for the repo-local Playwright environment:

```powershell
.\.venv311sys\Scripts\python.exe -m playwright install chromium
```

Quick browser launch check:

```powershell
.\.venv311sys\Scripts\python.exe -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); page=b.new_page(); page.goto('https://example.com'); print(page.title()); b.close(); p.stop()"
```

Expected output:

```text
Example Domain
```

## Run The API

Start the FastAPI app served from [server.py](./src/api/server.py):

```powershell
.\.venv311sys\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Available routes:

- `POST /run-task`
- `POST /step`
- `GET /run/{id}`
- `GET /health`

Example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/run-task `
  -ContentType "application/json" `
  -Body '{"intent":"Create a Gmail draft and stop before send."}'
```

## Run The Benchmark

Run the local Gmail benchmark entry point from [benchmark.py](./src/agent/benchmark.py):

```powershell
$env:TEMP=".\.tmp"
$env:TMP=".\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH=".\.ms-playwright"
.\.venv311sys\Scripts\python.exe -m src.agent.benchmark
```

This runs until one terminal condition is reached:

- stop-before-send
- retry limit reached
- max step limit reached

## Inspect Stored Runs

Replay one run from [replay.py](./src/store/replay.py):

```powershell
.\.venv311sys\Scripts\python.exe -m src.store.replay <run_id>
```

Summarize one run or a directory of runs from [summary.py](./src/store/summary.py):

```powershell
.\.venv311sys\Scripts\python.exe -m src.store.summary <run_id>
```

```powershell
.\.venv311sys\Scripts\python.exe -m src.store.summary runs
```

Run data is stored locally under `runs/<run_id>/` and includes:

- `state.json`
- `run.jsonl`
- per-step screenshots
- perception prompt/raw/parsed artifacts
- policy prompt/raw/parsed artifacts

## Browser Debug Mode

The Playwright executor in [browser.py](./src/executor/browser.py) supports environment-controlled debug launch settings.

Defaults preserve current behavior:

- `BROWSER_HEADLESS=true`
- `BROWSER_SLOW_MO_MS=0`
- `BROWSER_DEVTOOLS=false`

Enable visible debug mode:

```powershell
$env:BROWSER_HEADLESS="false"
$env:BROWSER_SLOW_MO_MS="250"
$env:BROWSER_DEVTOOLS="true"
```

`BROWSER_DEVTOOLS=true` opens Chromium devtools using the launch arg `--auto-open-devtools-for-tabs` because the installed Playwright version does not accept a `devtools=` launch kwarg.

## Tests

Run the browser executor tests:

```powershell
$env:TEMP=".\.tmp"
$env:TMP=".\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH=".\.ms-playwright"
.\.venv311sys\Scripts\python.exe -m pytest tests\test_browser_executor.py -q
```

Run the full MVP test suite you are working on as needed with the same Playwright env variables.
