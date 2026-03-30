# Operon

![Operon loop mark](assets/operon-loop-mark.svg)

**Operate any interface.**

Operon is a vision-driven computer-use engine that interacts with software the way a human does: by observing the screen, deciding the next action, executing it, and verifying the outcome. It uses **full-screen desktop automation** (via pyautogui/mss) to control any application — no browser drivers, no DOM access, no accessibility APIs.

License: Apache-2.0

## What It Is

Operon is not a chatbot and not a traditional automation script. It is a closed-loop interaction system built for UI-native automation:

- Perception from pixels (Gemini vision)
- Structured action selection (rule engine + LLM fallback)
- Real execution on the desktop via pyautogui
- Verification and recovery
- Replayable runs with persistent learning signals
- Local-only persistence

## Architecture

```
HTTP Request
    |
+-- ROUTES (src/api/routes.py) ---------------------------+
|                                                          |
|  POST /desktop/run-task   GET / (Pilot UI)               |
|  POST /desktop/step       GET /health                    |
|  POST /desktop/resume     GET /observer/api/*            |
|  POST /desktop/cleanup                                   |
+----------------------------------------------------------+
    |
+-- AGENT LOOP (src/agent/loop.py) -----------------------+
|                                                          |
|  1. CAPTURE    -- full-screen screenshot (mss)           |
|  2. PERCEIVE   -- Gemini vision -> ScreenPerception      |
|  3. CHOOSE     -- PolicyRuleEngine -> CombinedService    |
|  4. EXECUTE    -- DesktopExecutor (pyautogui)            |
|  5. VERIFY     -- DeterministicVerifierService           |
|  6. RECOVER    -- RuleBasedRecoveryManager               |
|                                                          |
+----------------------------------------------------------+
    |
+-- EXECUTOR (src/executor/desktop.py) -------------------+
|                                                          |
|  click, double_click, right_click, type, press_key,      |
|  hotkey, launch_app, drag, scroll, hover,                |
|  read_clipboard, write_clipboard, screenshot_region,     |
|  wait, wait_for_user, stop                               |
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
| `src/agent/` | Core loop, perception, policy, combined service, selector, verifier, recovery |
| `src/api/` | FastAPI routes, Operon Pilot UI, Observer API |
| `src/executor/` | DesktopExecutor (pyautogui/mss) |
| `src/models/` | Pydantic schemas: actions, perception, execution, state, progress, memory |
| `src/clients/` | GeminiHttpClient (async httpx with HTTP/2 + connection pooling) |
| `src/store/` | FileBackedRunStore, FileBackedMemoryStore, replay, summary |
| `prompts/` | LLM prompt templates (combined, perception, policy) |
| `tests/` | 204+ unit tests |

### Combined Perception+Policy

The `CombinedPerceptionPolicyService` sends a single Gemini call per step that returns both screen perception AND the next action. This eliminates one full LLM round-trip (~1-3s/step). The `PolicyCoordinator` still runs deterministic rules first and can override the LLM's action choice.

### Action Types (18 total)

| Action | Description |
|--------|-------------|
| `click` | Left-click at coordinates |
| `double_click` | Double-click at coordinates |
| `right_click` | Right-click (context menu) |
| `type` | Click to focus, then type text |
| `press_key` | Press a single key (enter, tab, etc.) |
| `hotkey` | Key combination (ctrl+c, win+r, etc.) |
| `launch_app` | Launch desktop app by name |
| `drag` | Drag from (x,y) to (x_end,y_end) |
| `scroll` | Scroll at position |
| `hover` | Move mouse without clicking |
| `read_clipboard` | Read clipboard contents |
| `write_clipboard` | Copy text to clipboard |
| `screenshot_region` | Capture a screen region |
| `wait` | Wait for duration |
| `wait_for_user` | Pause for human input |
| `stop` | Mark task complete |

### Progress Detection

The agent tracks forward progress by monitoring subgoal changes. If the model returns the same `active_subgoal` for 3+ consecutive steps, the no-progress streak increments and:

- Repeated actions (same type + same target region) are blocked
- Stagnation warnings are injected into the prompt
- After enough stagnation, the run terminates

Coordinate-based targets are bucketed to a 50px grid so nearby clicks are detected as repeats.

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

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | (required) | Gemini API key |
| `GEMINI_PERCEPTION_MODEL` | `gemini-2.0-flash` | Model for perception (faster) |
| `GEMINI_POLICY_MODEL` | `gemini-2.5-flash` | Model for policy (smarter) |
| `CORS_ORIGINS` | (disabled) | Comma-separated allowed origins |

## Run The API

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/` for the Operon Pilot UI.

### Routes

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` or `/desktop-pilot` | Operon Pilot UI |
| GET | `/health` | Health check |
| POST | `/desktop/run-task` | Create a desktop automation run |
| POST | `/desktop/step` | Advance a run one step |
| POST | `/desktop/resume` | Resume run from user wait |
| POST | `/desktop/cleanup` | Close apps launched during a run |
| GET | `/desktop/run/{id}` | Get run state |
| GET | `/observer/api/runs` | List recent runs |
| GET | `/observer/api/run/{id}` | Run snapshot with step details |

### Example

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/desktop/run-task `
  -ContentType "application/json" `
  -Body '{"intent":"Open Notepad and type Hello World"}'
```

## Run The Benchmark

```powershell
$env:FORM_BENCHMARK_URL = "https://practice-automation.com/form-fields/"
python -m src.agent.benchmark
```

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
    combined_prompt.txt
    combined_raw.txt
    combined_parsed.json
    execution_trace.json
    progress_trace.json
```

## Tests

```powershell
python -m pytest tests\ -q
```

## Windows Shell Troubleshooting

If Python or other external processes fail to launch from PowerShell with a COM+ error:

```powershell
. .\scripts\repair-process-env.ps1 -PersistForSession
```
