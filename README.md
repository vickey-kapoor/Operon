# Operon

![Operon loop mark](assets/operon-loop-mark.svg)

**Operate any interface.**

Operon is a vision-driven computer-use engine that interacts with software the way a human does: by observing the screen, deciding the next action, executing it, and verifying the outcome. It supports both **browser automation** (via Playwright) and **full-screen desktop automation** (via pyautogui/mss).

License: Apache-2.0

## What It Is

Operon is not a chatbot and not a traditional browser automation script. It is a closed-loop interaction system built for UI-native automation:

- Perception from pixels (Gemini vision)
- Structured action selection (rule engine + LLM fallback)
- Real execution on browser or desktop
- Verification and recovery
- Replayable runs with persistent learning signals
- Local-only persistence

## Architecture

```
HTTP Request
    |
+-- ROUTES (src/api/routes.py) ---------------------------+
|                                                          |
|  Browser:             Desktop:          UI:              |
|  POST /run-task       POST /desktop/*   GET / (Pilot)    |
|  POST /step                                              |
|  POST /resume                                            |
|  GET  /run/{id}                                          |
+----------------------------------------------------------+
    |
+-- AGENT LOOP (src/agent/loop.py) -----------------------+
|                                                          |
|  1. CAPTURE    -- screenshot (Playwright or mss)         |
|  2. PERCEIVE   -- Gemini vision -> ScreenPerception      |
|  3. CHOOSE     -- PolicyRuleEngine -> GeminiPolicyService|
|  4. EXECUTE    -- executor.execute(AgentAction)          |
|  5. VERIFY     -- DeterministicVerifierService           |
|  6. RECOVER    -- RuleBasedRecoveryManager               |
|                                                          |
+----------------------------------------------------------+
    |
+-- EXECUTORS --------------------------------------------+
|                                                          |
|  PlaywrightBrowserExecutor    DesktopExecutor            |
|  (web pages)                  (full screen)              |
|                                                          |
|  click, type, select,         click, double_click,       |
|  press_key, navigate,         right_click, type,         |
|  wait                         press_key, hotkey,         |
|                               launch_app, drag, scroll,  |
|                               hover, read_clipboard,     |
|                               write_clipboard,           |
|                               screenshot_region, wait    |
+----------------------------------------------------------+
    |
+-- STORAGE (src/store/) ---------------------------------+
|  runs/{run_id}/state.json                                |
|  runs/{run_id}/run.jsonl                                 |
|  runs/{run_id}/step_N/before.png, after.png, ...         |
+----------------------------------------------------------+
```

### Module Map

| Directory | Purpose |
|-----------|---------|
| `src/agent/` | Core loop, perception, policy, selector, verifier, recovery |
| `src/api/` | FastAPI routes, Operon Pilot UI (unified desktop + browser) |
| `src/executor/` | BrowserExecutor (Playwright) and DesktopExecutor (pyautogui/mss) |
| `src/models/` | Pydantic schemas: actions, perception, execution, state, memory |
| `src/clients/` | GeminiHttpClient (vision + text generation) |
| `src/store/` | FileBackedRunStore, FileBackedMemoryStore, replay, summary |
| `prompts/` | LLM prompt templates for perception and policy (browser + desktop) |
| `tests/` | 22 test files, 195+ tests |

### Policy Layer

The `PolicyCoordinator` runs a deterministic `PolicyRuleEngine` first. If no rule fires, it falls back to `GeminiPolicyService` (LLM prompt). This minimizes Gemini calls and ensures predictable behavior for known patterns.

### Action Types (18 total)

| Action | Browser | Desktop | Description |
|--------|:-------:|:-------:|-------------|
| `click` | x | x | Left-click at element/coordinates |
| `double_click` | | x | Double-click at coordinates |
| `right_click` | | x | Right-click (context menu) |
| `type` | x | x | Type text into a field |
| `select` | x | | Select dropdown option |
| `press_key` | x | x | Press a single key |
| `hotkey` | | x | Key combination (e.g. ctrl+c) |
| `navigate` | x | | Go to URL |
| `launch_app` | | x | Launch desktop app by name |
| `drag` | | x | Drag from (x,y) to (x_end,y_end) |
| `scroll` | | x | Scroll at position |
| `hover` | | x | Move mouse without clicking |
| `read_clipboard` | | x | Read clipboard contents |
| `write_clipboard` | | x | Copy text to clipboard |
| `screenshot_region` | | x | Capture a screen region |
| `wait` | x | x | Wait for duration |
| `wait_for_user` | x | x | Pause for human input |
| `stop` | x | x | Mark task complete |

## Runtime

Python **3.11** required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Gemini credentials via `.env` file or environment:

```
GEMINI_API_KEY=your-key-here
```

## Windows Playwright Setup

Set repo-local temp and browser cache paths:

```powershell
$env:TEMP = (Join-Path $PWD ".tmp")
$env:TMP = (Join-Path $PWD ".tmp")
$env:PLAYWRIGHT_BROWSERS_PATH = (Join-Path $PWD ".ms-playwright")
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
New-Item -ItemType Directory -Force $env:PLAYWRIGHT_BROWSERS_PATH | Out-Null
```

Install Chromium:

```powershell
python -m playwright install chromium
```

## Run The API

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

### Routes

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` or `/desktop-pilot` | Operon Pilot UI |
| GET | `/health` | Health check |
| POST | `/run-task` | Create a browser automation run |
| POST | `/step` | Advance a browser run one step |
| POST | `/resume` | Resume browser run from user wait |
| GET | `/run/{id}` | Get browser run state |
| POST | `/desktop/run-task` | Create a desktop automation run |
| POST | `/desktop/step` | Advance a desktop run one step |
| POST | `/desktop/resume` | Resume desktop run from user wait |
| GET | `/desktop/run/{id}` | Get desktop run state |

### Operon Pilot UI

Open `http://127.0.0.1:8080/` for the unified automation UI. Toggle between **Desktop** and **Browser** modes. Quick tasks are organized by category:

**Desktop mode:**
- Launch Apps, Mouse Actions, Drag & Drop, Clipboard, Screen Analysis

**Browser mode:**
- Form Filling, Navigation, Interaction, Verification

Example API calls:

```powershell
# Desktop task
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/desktop/run-task `
  -ContentType "application/json" `
  -Body '{"intent":"Open Notepad and type Hello World"}'

# Browser task
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/run-task `
  -ContentType "application/json" `
  -Body '{"intent":"Fill out the contact form and submit it"}'
```

## Run The Benchmark

```powershell
$env:FORM_BENCHMARK_URL = "https://practice-automation.com/form-fields/"
python -m src.agent.benchmark
```

Terminal conditions: `form_submitted_success`, retry limit, max step limit.

## Inspect Stored Runs

```powershell
python -m src.store.replay <run_id>
python -m src.store.summary <run_id>
python -m src.store.summary runs
```

Run data layout:

```
runs/<run_id>/
  state.json
  run.jsonl
  step_1/
    before.png
    after.png
    perception_prompt.txt
    perception_raw.txt
    perception_parsed.json
    policy_prompt.txt
    policy_raw.txt
    policy_decision.json
    execution_trace.json
    progress_trace.json
```

## Browser Debug Mode

```powershell
$env:BROWSER_HEADLESS = "false"
$env:BROWSER_SLOW_MO_MS = "250"
$env:BROWSER_DEVTOOLS = "true"
```

## Tests

```powershell
$env:GEMINI_API_KEY = "fake-test-key"
python -m pytest tests\ -q
```

Single test file:

```powershell
python -m pytest tests\test_desktop_executor.py -q
```

## Windows Shell Troubleshooting

If Python or other external processes fail to launch from PowerShell with a COM+ error:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```
