# UI Navigator

AI agent that controls browsers and desktops by analysing screenshots with **Gemini 2.5 Flash** and executing actions through **Playwright** (browser), **pyautogui** (desktop), or a **Chrome Extension** (real tabs). Exposed as a **FastAPI** REST + WebSocket service, deployable to **Google Cloud Run**.

**Version:** 1.4.0

---

## Three Execution Modes

| Mode | How it works | Endpoint |
|---|---|---|
| **Browser Mode** | Headless Chromium via Playwright | `POST /navigate`, `WS /ws/{task_id}` |
| **WebPilot Extension** | Chrome sidebar controls real browser tabs | `WS /webpilot/ws/{session_id}` |
| **Desktop Mode** | Full-screen control via mss + pyautogui | `WS /desktop/ws/{session_id}`, `POST /desktop/start` |

All three share the same core loop: **screenshot → Gemini 2.5 Flash → action plan → execute → repeat**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Clients                                      │
│  REST API   │  WebSocket   │  Chrome Extension  │  Desktop UI    │
└──────┬──────┴──────┬───────┴────────┬───────────┴───────┬───────┘
       │             │                │                   │
┌──────▼─────────────▼────────────────▼───────────────────▼───────┐
│              FastAPI Server  (src/api/server.py)                  │
│  ├── REST endpoints (navigate, tasks, sessions, health)          │
│  ├── WebSocket endpoints (browser, webpilot, desktop)            │
│  └── Static dashboards (/ui/*)                                   │
└──────┬──────────────┬───────────────┬───────────────────────────┘
       │              │               │
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
│ Browser     │ │ WebPilot   │ │ Desktop        │
│ Executor    │ │ Handler    │ │ Executor       │
│ (Playwright)│ │ (Gemini)   │ │ (mss+pyautogui)│
└─────────────┘ └────────────┘ └────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/<redacted>y-kapoor/UI_Navigator.git
cd UI_Navigator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium --with-deps
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY to your Gemini key
```

### 3. Run

```bash
# Load env and start server
export $(grep -v '^#' .env | xargs) && python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

API docs: `http://localhost:8080/docs`

---

## API Endpoints

### Browser Mode
| Method | Path | Description |
|---|---|---|
| `POST` | `/navigate` | Start a browser task, returns `task_id` |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{task_id}` | Poll task status/result |
| `DELETE` | `/tasks/{task_id}` | Cancel a running task |
| `WS` | `/ws/{task_id}` | Stream step events in real time |
| `POST` | `/screenshot` | One-shot screenshot + Gemini analysis |
| `POST` | `/clarify` | Get clarifying questions for ambiguous tasks |

### WebPilot Extension
| Method | Path | Description |
|---|---|---|
| `POST` | `/webpilot/sessions` | Create WebPilot session |
| `DELETE` | `/webpilot/sessions/{id}` | End WebPilot session |
| `WS` | `/webpilot/ws/{session_id}` | Real-time action loop |
| `POST` | `/webpilot/tts` | Gemini TTS narration (base64 WAV) |

### Desktop Mode
| Method | Path | Description |
|---|---|---|
| `POST` | `/desktop/sessions` | Create Desktop session |
| `DELETE` | `/desktop/sessions/{id}` | End Desktop session |
| `GET` | `/desktop/sessions/{id}` | Poll Desktop session status |
| `POST` | `/desktop/start` | Start autonomous desktop task |
| `WS` | `/desktop/ws/{session_id}` | Interactive Desktop action loop |

### Other
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/sessions` | Create ADK session |
| `POST` | `/sessions/{id}/step` | Send screenshot, get ActionPlan |

---

## Web Dashboards

| URL | Purpose |
|---|---|
| `/ui/desktop.html` | DesktopPilot — start autonomous desktop tasks, live polling |
| `/ui/visual-tests.html` | 10 WS scenarios with live message log (stub mode) |
| `/ui/e2e-tests.html` | 13 E2E test scenarios with live screenshots |

---

## WebPilot Chrome Extension

Controls real browser tabs via a sidebar UI.

```bash
# Build the sidebar
cd webpilot-extension/sidebar && npm install && npm run build && cd ../..

# Load in Chrome:
# 1. chrome://extensions → Developer mode → Load unpacked
# 2. Select the webpilot-extension/ folder
# 3. Click the sidebar icon to open
```

---

## Desktop Mode

Full-screen OS control via mss (screenshots) + pyautogui (input). Requires `DESKTOP_MODE_ENABLED=true`.

Two execution paths:
- **Interactive** (WS): Client captures screenshots, sends to server for AI thinking, executes actions locally
- **Autonomous** (`POST /desktop/start`): Server captures + thinks + executes everything

```bash
# Start with Desktop Mode enabled
DESKTOP_MODE_ENABLED=true uvicorn src.api.server:app --host 0.0.0.0 --port 8080

# Open the dashboard
# http://localhost:8080/ui/desktop.html
```

---

## Docker

```bash
cp .env.example .env
docker compose up --build
```

---

## Cloud Run Deployment

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_API_KEY=your-key
chmod +x deploy.sh && ./deploy.sh
```

---

## Testing

```bash
# All non-browser tests (155 tests, no Chromium needed)
DESKTOP_MOCK=true DESKTOP_MODE_ENABLED=true python -m pytest tests/ -v \
  --ignore=tests/test_agent.py --ignore=tests/test_live_integration.py

# Browser tests (requires Chromium)
python -m pytest tests/test_agent.py -v

# Single file
python -m pytest tests/test_webpilot_e2e.py -v
```

### Test Files

| File | Tests | What |
|---|---|---|
| `test_api.py` | 18 | REST API endpoints |
| `test_webpilot_api.py` | 22 | WebPilot unit tests |
| `test_webpilot_e2e.py` | 24 | WebPilot e2e (real server + WS) |
| `test_sessions.py` | 13 | ADK sessions |
| `test_clarifier.py` | 7 | Clarify endpoint |
| `test_gap_coverage.py` | 17 | Architecture review gaps |
| `test_user_failures.py` | 8 | User-testing regressions |
| `test_desktop_executor.py` | 25 | DesktopExecutor unit tests |
| `test_desktop_api.py` | 21 | Desktop Mode API integration |
| `test_dashboard.py` | 83 | Dashboard system tests |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `DESKTOP_MODE_ENABLED` | No | Set `true` to enable Desktop Mode |
| `DESKTOP_MOCK` | No | Set `true` for CI (stubs all desktop actions) |
| `BROWSER_HEADLESS` | No | Headless Chromium (default: `true`) |
| `MAX_CONCURRENT_TASKS` | No | Concurrent task limit (default: `5`) |
| `BROWSER_WIDTH` / `HEIGHT` | No | Viewport size (default: `1280x800`) |
| `GEMINI_MODEL` | No | Gemini model (default: `gemini-2.5-flash`) |
| `ACTION_LOOP_TIMEOUT` | No | Hard timeout per action loop (default: `120s`) |
| `MAX_LOOP_STEPS` | No | Max steps per loop (default: `30`) |
| `TASK_STORE` | No | `memory` (default) or `firestore` |

---

## Project Structure

```
UI_Navigator/
├── src/
│   ├── agent/
│   │   ├── core.py                  # UINavigatorAgent — main browser loop
│   │   ├── vision.py                # GeminiVisionClient — Gemini API
│   │   ├── planner.py               # ActionPlanner — JSON parsing
│   │   ├── webpilot_handler.py      # WebPilot/Desktop Gemini handler
│   │   └── desktop_system_prompt.py # Desktop Mode system prompt
│   ├── executor/
│   │   ├── base.py                  # AbstractExecutor interface
│   │   ├── browser.py               # PlaywrightBrowserExecutor
│   │   ├── desktop.py               # DesktopExecutor (mss + pyautogui)
│   │   └── actions.py               # Action/ActionResult models
│   └── api/
│       ├── server.py                # FastAPI app + lifespan
│       ├── webpilot_routes.py       # WebPilot WS + REST
│       ├── desktop_routes.py        # Desktop Mode WS + REST
│       ├── models.py                # Shared Pydantic models
│       ├── webpilot_models.py       # WebPilot models
│       ├── desktop_models.py        # Desktop models
│       ├── store.py                 # TaskStore abstraction
│       └── static/                  # Web dashboards
├── webpilot-extension/              # Chrome Extension
├── tests/                           # 238 tests
├── docs/                            # PRDs
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
└── requirements.txt
```
