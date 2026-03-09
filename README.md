# UI Navigator

An AI agent that controls web browsers by analysing screenshots with **Gemini 2.0 Flash** and executing actions through **Playwright**.  Exposed as a **FastAPI** REST + WebSocket service and deployable to **Google Cloud Run**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client / User                         │
│  POST /navigate   GET /tasks/{id}   WS /ws/{id}             │
└──────────────┬───────────────────────────────────────────────┘
               │ HTTP / WebSocket
┌──────────────▼───────────────────────────────────────────────┐
│              FastAPI Server  (src/api/server.py)             │
│  • Task queue (in-memory dict)                               │
│  • Background asyncio tasks                                  │
│  • WebSocket broadcast for real-time events                  │
└──────────────┬───────────────────────────────────────────────┘
               │ async
┌──────────────▼───────────────────────────────────────────────┐
│         UINavigatorAgent  (src/agent/core.py)                │
│                                                              │
│  ┌──────────────────┐      ┌─────────────────────────────┐  │
│  │  Screenshot loop │      │  Conversation history       │  │
│  │  (max_steps)     │      │  (last 10 turns)            │  │
│  └────────┬─────────┘      └─────────────────────────────┘  │
│           │                                                  │
│  ┌────────▼─────────┐      ┌─────────────────────────────┐  │
│  │  ActionPlanner   │─────▶│  GeminiVisionClient         │  │
│  │  (planner.py)    │      │  (vision.py)                │  │
│  │  • JSON parsing  │      │  • gemini-2.0-flash         │  │
│  │  • retry logic   │      │  • multimodal (image+text)  │  │
│  └────────┬─────────┘      │  • JSON response mode       │  │
│           │                └─────────────────────────────┘  │
│  ┌────────▼──────────────────────────────────────────────┐  │
│  │  PlaywrightBrowserExecutor  (executor/browser.py)     │  │
│  │  • click / type / key / scroll / navigate / wait      │  │
│  │  • headless Chromium  1280×800                        │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| Playwright / Chromium | installed via `playwright install chromium` |
| GCP account | for Cloud Run deployment |
| Gemini API key | [Google AI Studio](https://aistudio.google.com/) |

---

## Local Development Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd UI_Navigator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium --with-deps
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=<your key>
```

### 3. Run the server

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8080
```

The API is now available at `http://localhost:8080`.
Interactive docs: `http://localhost:8080/docs`.

---

## Docker Usage

### Build and run

```bash
# Copy and fill in your API key
cp .env.example .env

docker compose up --build
```

### Build manually

```bash
docker build -t ui-navigator .
docker run -p 8080:8080 --env-file .env ui-navigator
```

---

## GCP Deployment

### Prerequisites

- `gcloud` CLI installed and authenticated
- Project with billing enabled
- Gemini API key stored in Secret Manager (the script creates it if missing)

### One-command deploy

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_REGION=us-central1   # optional, default

chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Enable required GCP APIs
2. Create an Artifact Registry repository
3. Build and push the Docker image
4. Deploy to Cloud Run (2 GB RAM, 2 CPU, 0–5 instances)
5. Print the live service URL

### Automated CI/CD with Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions _REGION=us-central1,_SERVICE_NAME=ui-navigator .
```

Or connect a GitHub trigger in the Cloud Build console to auto-deploy on push.

---

## API Documentation

### `POST /navigate`

Start a navigation task.  Returns immediately with a `task_id`.

```bash
curl -X POST http://localhost:8080/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Go to https://news.ycombinator.com and list the top 3 story titles",
    "max_steps": 10
  }'
```

**Response:**
```json
{"task_id": "c4f2a1b3-...", "status": "started"}
```

**Body parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | string | required | Natural language instruction |
| `start_url` | string | null | Optional URL to open before starting |
| `max_steps` | int | 20 | Max Gemini→action cycles (1–50) |

---

### `GET /tasks/{task_id}`

Poll the status and result of a task.

```bash
curl http://localhost:8080/tasks/c4f2a1b3-...
```

**Response:**
```json
{
  "task_id": "c4f2a1b3-...",
  "status": "done",
  "task": "Go to HN and list top 3 stories",
  "start_url": null,
  "max_steps": 10,
  "result": {
    "success": true,
    "result": "Top 3 stories: 1. ...",
    "steps_taken": 4,
    "screenshots": ["<base64>", "..."],
    "error": null
  }
}
```

`status` values: `pending` | `running` | `done` | `error`

---

### `WebSocket /ws/{task_id}`

Stream real-time progress events for a running task.

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/c4f2a1b3-...");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

**Event types:**

```jsonc
// Step progress
{"type": "step", "step": 1, "observation": "...", "action": "navigate: Open example.com", "screenshot": "<base64>"}

// Task complete
{"type": "done", "success": true, "result": "Task completed successfully.", "steps_taken": 3, "error": null}

// Error
{"type": "error", "message": "Browser crashed: ..."}

// Keep-alive
{"type": "ping"}
```

---

### `POST /screenshot`

Analyse a URL or uploaded screenshot with Gemini.

**Option A — Upload an image:**
```bash
curl -X POST http://localhost:8080/screenshot \
  -F "file=@screen.png" \
  -F "task=List all clickable buttons"
```

**Option B — Capture from URL:**
```bash
curl -X POST http://localhost:8080/screenshot \
  -F "url=https://example.com" \
  -F "task=Describe the page layout"
```

**Response:**
```json
{
  "screenshot": "<base64 PNG>",
  "analysis": {
    "observation": "...",
    "reasoning": "...",
    "actions": [...],
    "done": false,
    "result": null
  }
}
```

---

### `GET /health`

```bash
curl http://localhost:8080/health
# {"status": "ok", "active_tasks": 0, "total_tasks": 5}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key |
| `GOOGLE_CLOUD_PROJECT` | — | GCP project ID (for deployment) |
| `GOOGLE_CLOUD_REGION` | `us-central1` | GCP region |
| `PORT` | `8080` | Server port |
| `LOG_LEVEL` | `INFO` | Python log level |
| `MAX_CONCURRENT_TASKS` | `5` | Max simultaneous browser sessions |
| `BROWSER_HEADLESS` | `true` | Run Chromium headlessly |
| `BROWSER_WIDTH` | `1280` | Viewport width in pixels |
| `BROWSER_HEIGHT` | `800` | Viewport height in pixels |

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests include:
- **Unit tests** — ActionPlanner JSON parsing (valid, malformed, markdown-wrapped)
- **Integration tests** — real headless Chromium (navigate, scroll, screenshot)
- **Full agent loop** — mocked Gemini API, complete navigate-and-report cycle

---

## How It Works

1. **User submits a task** via `POST /navigate`.
2. **Agent opens a browser** (headless Chromium, 1280×800).
3. **Loop begins** (up to `max_steps` iterations):
   a. **Screenshot** — capture the current viewport as a PNG.
   b. **Gemini analysis** — send the screenshot + task to `gemini-2.0-flash` with a carefully crafted system prompt.  The model returns a structured JSON `ActionPlan` describing what it sees, why it is taking certain actions, and the list of actions to execute.
   c. **Action execution** — Playwright executes each action (click, type, navigate, scroll, etc.).
   d. **History update** — the plan is appended to conversation history so the next call has context.
   e. If `done: true` is returned, the loop exits and the result is returned.
4. **Progress events** are broadcast via WebSocket in real time.
5. **Result** is stored and accessible via `GET /tasks/{task_id}`.

### Supported actions

| Action | Description |
|---|---|
| `navigate` | Open a URL |
| `click` | Click at pixel coordinates |
| `type` | Type text into the focused element |
| `key` | Press a named key (Enter, Tab, Escape, …) |
| `scroll` | Scroll in a direction by N units |
| `wait` | Pause for N milliseconds |
| `screenshot` | Re-capture the viewport (observe-only) |
| `done` | Signal task completion |

---

## Project Structure

```
UI_Navigator/
├── src/
│   ├── agent/
│   │   ├── core.py       # UINavigatorAgent — main loop
│   │   ├── vision.py     # GeminiVisionClient — multimodal API calls
│   │   └── planner.py    # ActionPlanner — JSON parsing & validation
│   ├── executor/
│   │   ├── browser.py    # PlaywrightBrowserExecutor
│   │   └── actions.py    # Action / ActionResult Pydantic models
│   └── api/
│       └── server.py     # FastAPI app — REST + WebSocket
├── tests/
│   └── test_agent.py
├── Dockerfile
├── docker-compose.yml
├── cloudbuild.yaml
├── deploy.sh
├── requirements.txt
└── .env.example
```
