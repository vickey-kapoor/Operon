# WebPilot — AI Browser Navigator
## Product Requirements Document (PRD)
### Version 2.0 | Google Cloud x Gemini Hackathon Submission

---

## 1. OVERVIEW

### 1.1 Product Summary
WebPilot is a Chrome browser extension that lets users control any website using natural voice or text commands. It uses Gemini 2.5 Flash multimodal vision to observe the browser screen as a human would — without relying on APIs or DOM scraping — and executes actions (clicks, typing, scrolling, keyboard shortcuts, navigation) to complete tasks on behalf of the user.

### 1.2 Hackathon Track
**UI Navigator ☸️** — Visual UI Understanding & Interaction

### 1.3 Mandatory Tech Requirements
- Gemini multimodal API (gemini-2.5-flash) to interpret screenshots and output executable actions
- Agent hosted on Google Cloud Run
- Gemini Live API (gemini-live-2.5-flash-preview) for real-time per-session voice interaction and interruption handling
- Gemini TTS (gemini-2.5-flash-preview-tts) for voice narration

### 1.4 One-Line Pitch
> "Tell WebPilot what you want. It sees your screen and does it — on any website, without any integrations."

### 1.5 Core Differentiator
Every existing automation tool (Zapier, Selenium, browser copilots) requires APIs, DOM access, or pre-built connectors. WebPilot works purely from visual understanding of screenshots — making it universal across any website, legacy tool, or internal system.

---

## 2. USER EXPERIENCE

### 2.1 User Mental Model
The user should feel like they have a smart co-pilot sitting next to them who can see their screen and execute tasks when asked. No setup. No training. No configuring automations. Just talk or type.

### 2.2 Entry Points
The extension is always available via:
- A persistent floating sidebar on the right edge of any browser tab
- Keyboard shortcut: `Cmd+Shift+A` (Mac) / `Ctrl+Shift+A` (Windows)

### 2.3 Interaction Model
**Dual input — voice + text hybrid:**
- **Voice:** User holds mic button and speaks naturally (Web Speech API with 3 restart attempts)
- **Text:** User types intent into sidebar input field
- No rigid syntax or command structure required
- Natural language intent only

### 2.4 Sidebar UI Layout

```
┌─────────────────────────┐
│  🌐 WebPilot        [X] │
├─────────────────────────┤
│  🎙️ [Hold to Speak]    │
│  ✏️ [Type a task...]   │
├─────────────────────────┤
│  CURRENT TASK           │
│  "Find flight Austin    │
│   → Tokyo under $400"   │
├─────────────────────────┤
│  LIVE ACTIONS           │
│  ✅ Opened Google Flights│
│  ✅ Entered destination  │
│  ✅ Set date: Fri Nov 8  │
│  🔄 Reading results...  │
├─────────────────────────┤
│  ⚠️ CONFIRM BEFORE      │
│  PROCEEDING             │
│  [✅ Yes] [⛔ Stop]     │
└─────────────────────────┘
```

### 2.5 Voice Narration — Full Narration Mode
Agent speaks at every meaningful step using Gemini TTS (Aoede voice) with Web Speech Synthesis fallback. Tone: short, confident, human.

Narration fires ONLY when a new action log entry is added — not on status updates (prevents double-speak).

| Moment | Agent Says |
|---|---|
| Task received | "Got it, I'll find flights from Austin to Tokyo under $400" |
| Navigating | "Opening Google Flights now" |
| Filling form | "Setting destination to Tokyo, dates to next Friday" |
| Waiting for page | "Loading results, just a moment" |
| Silent retry | *(retries quietly, no interruption)* |
| Stuck after 3 retries | "Having trouble with this step, trying a different approach" |
| Confirm before irreversible action | "I found a flight for $387 on ANA. Want me to go ahead and book it?" |
| Task complete | "Done! Three non-stop options found. Cheapest is $387 on ANA departing Friday 11pm. Want me to check hotels too?" |
| User interrupts | "Got it, updating the search now" |

### 2.6 Interruption Handling
Three types, classified automatically by keyword matching (abort checked before redirect):

**Type 1 — Refinement** (default — e.g. "make it non-stop only")
- Agent finishes current micro-action
- Applies new constraint, merges with original intent
- Continues from current screen state

**Type 2 — Redirect** (keywords: "instead", "new goal", "start over", "different", "actually")
- Agent stops immediately
- Clears conversation history
- Confirms new goal verbally
- Starts fresh from current screen

**Type 3 — Abort** (keywords: "stop", "abort", "quit", "never mind", "nevermind", "forget it", "forget about it")
- Agent stops mid-action instantly
- Says: "Stopped. What would you like to do?"
- Returns to idle state, awaiting new instruction
- Idle-interrupt guard: rejects interrupt messages when session is not running/thinking/done

### 2.7 Confirmation Gate — Irreversible Actions
Before executing any irreversible action (booking, submitting a form, deleting, purchasing):
1. Agent pauses — sends `confirm_required` action
2. Sidebar shows confirmation card with action summary and narration
3. Agent speaks the confirmation prompt via `useEffect` on `status === "confirming"`
4. User must explicitly say "yes" / "proceed" or click confirm
5. If user says "wait" or "stop" → action cancelled, sidebar dispatches STOPPED

### 2.8 Error Handling — Auto-Retry with Stuck Detection
```
Action attempted
      ↓
Capture new screenshot → MD5 hash comparison
      ↓
Same as previous screenshot?
      ↓
  YES → increment stuck counter (up to MAX_RETRIES = 3)
      ↓
  After 3 identical screenshots:
      → inject "STUCK" hint suggesting keyboard alternatives
      → reset baseline hash for fresh comparison
      → Gemini replans from current screen
      ↓
  NO → reset stuck counter, continue normally
```
User never sees failure states. Agent always appears to be progressing.

### 2.9 Action Loop Safeguards
- **Hard timeout**: ACTION_LOOP_TIMEOUT = 120s (configurable)
- **Step budget**: MAX_LOOP_STEPS = 30 (configurable)
- **Extension auto-stop**: 15 max steps or 3 consecutive failures → sends stop to server
- **Post-interrupt budget**: inherits half of MAX_LOOP_STEPS

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture
```
USER BROWSER
│
├── Chrome Extension (Manifest V3 — v1.1.0)
│   ├── Background Service Worker (WS owner, screenshot, action dispatch)
│   ├── Sidebar UI (React 18 + Vite)
│   ├── Voice Input (Web Speech API — hold-to-speak)
│   ├── Voice Output (Gemini TTS + Web Speech Synthesis fallback)
│   ├── Screen Capture (chrome.tabs.captureVisibleTab)
│   └── Content Script (DOM executor: click, type, scroll, key)
│
│ [HTTPS WebSocket — persistent connection, owned by background.js]
│
├── Google Cloud Run (Backend — FastAPI v1.4.0)
│   ├── WebPilot Session Manager (in-memory per session)
│   ├── WebPilot Handler (Gemini Live API, per-session persistent)
│   ├── Legacy Handler (Gemini 2.5 Flash, shared fallback)
│   ├── Action Decision Engine (thinking_budget=1024)
│   ├── Interruption Classifier (abort/redirect/refinement)
│   ├── TTS Endpoint (Gemini TTS → base64 WAV)
│   ├── Task Store (memory or Firestore)
│   └── API Key Auth + Rate Limiting Middleware
│
│ [Gemini API — google-genai >= 0.8]
│
└── Gemini 2.5 Flash (Vision + Reasoning)
    ├── Receives screenshot + intent + history + current URL
    ├── Uses thinking_budget=1024 for spatial reasoning
    ├── Outputs next action as structured JSON
    └── Handles interruptions via context injection
```

### 3.2 Component Responsibilities

| Component | Location | Responsibility |
|---|---|---|
| Background SW | Extension (background.js) | WS owner, screenshot capture, action dispatch, session retry, keep-alive alarm |
| Sidebar UI | Extension (React 18) | User input, action log, confirmation UI, status indicator |
| Content Script | Extension (content.js) | DOM execution: click (shadow DOM), type (React-compatible), scroll, key |
| Voice input | Extension (useVoiceInput.js) | Web Speech API, 3 restart attempts, final results only |
| Voice output | Extension (useVoiceOutput.js) | Gemini TTS with Web Speech Synthesis fallback (Aoede / Google UK English Female) |
| WebPilot Handler | Cloud Run (webpilot_handler.py) | Per-session Gemini Live API connection, action planning, interruption classification |
| Legacy Handler | Cloud Run (webpilot_handler.py) | Shared Gemini 2.5 Flash handler (fallback when Live API unavailable) |
| Session Manager | Cloud Run (webpilot_routes.py) | Per-session state, handler lifecycle, cleanup |
| TTS Endpoint | Cloud Run (webpilot_routes.py) | POST /webpilot/tts — Gemini TTS narration → base64 WAV |
| API Auth | Cloud Run (server.py) | APIKeyMiddleware + RateLimitMiddleware |
| Task Store | Cloud Run (store.py) | Abstract store + memory/Firestore backends |

### 3.3 Data Flow — Action Loop

**Step 1: User Input**
```
User speaks or types intent
→ Web Speech API transcribes audio to text (or text input)
→ Sidebar sends message to background.js via chrome.runtime.sendMessage
→ Background.js sends to Cloud Run via WebSocket:
{
  "type": "task",
  "intent": "Find flight Austin to Tokyo under $400",
  "screenshot": "<base64 PNG>",
  "current_url": "https://..."
}
```

**Step 2: Cloud Run → Gemini**
```
WebPilot Handler constructs prompt with:
- System prompt (one action at a time, structured JSON only)
- Goal / intent
- Current URL context
- Previous action summaries
- Screenshot as image (types.Part.from_bytes)
- thinking_budget=1024 for spatial reasoning

Gemini responds with structured JSON action.
```

**Step 3: Gemini Response → Action Schema**
```json
{
  "observation": "I see Google Flights search page",
  "plan": "Click the destination field and type Tokyo",
  "steps_completed": 2,
  "action": "click",
  "x": 412,
  "y": 280,
  "narration": "Opening Google Flights now",
  "action_label": "Opened Google Flights",
  "is_irreversible": false
}
```

**Step 4: Cloud Run → Extension**
```
If is_irreversible = true:
  → Send confirm_required action to extension
  → Block on WebSocket waiting for user confirm/deny
  → On confirm: execute action; on deny: stop

If is_irreversible = false:
  → Send action directly to extension via WebSocket
```

**Step 5: Extension Executes Action**
```
Background.js receives action
→ Re-injects content script (handles post-navigation loss)
→ Dispatches to content.js for DOM execution:
  - click: elementFromPoint with shadow DOM traversal
  - type: React-compatible input events (input + change + dispatchEvent)
  - scroll: window.scrollBy
  - key: keyboard event simulation
  - navigate: chrome.tabs.update (not window.location — content scripts blocked on new pages)
→ After navigate: waitForTabLoad (tab status "complete" + 1.5s settle, 15s timeout)
```

**Step 6: Loop**
```
Action executed
→ Capture new screenshot
→ MD5 hash comparison for stuck detection
→ Send screenshot to Cloud Run via WebSocket
→ Gemini plans next action
→ Repeat until action = "done" or budget exhausted
```

**Step 7: Interruption**
```
User speaks/types mid-task
→ Extension sends interrupt message via WebSocket:
{
  "type": "interrupt",
  "message": "make it non-stop only"
}

→ Cloud Run classifies: ABORT / REDIRECT / REFINEMENT
→ ABORT: short-circuit, no Gemini call
→ REDIRECT: clear history, start fresh with new goal
→ REFINEMENT: merge new constraint with original intent, continue
```

---

## 4. FILE STRUCTURE

```
UI_Navigator/
├── src/
│   ├── agent/
│   │   ├── core.py               # UINavigatorAgent — main screenshot→plan→execute loop
│   │   ├── vision.py             # GeminiVisionClient — gemini-2.5-flash, history alternation
│   │   ├── planner.py            # ActionPlanner — parses Gemini JSON → ActionPlan
│   │   ├── webpilot_handler.py   # WebPilotHandler (Live API) + LegacyWebPilotHandler
│   │   └── adk_agent.py          # ADK Agent + Runner + InMemorySessionService
│   ├── executor/
│   │   ├── actions.py            # ActionType enum, Action/ActionResult models
│   │   └── browser.py            # PlaywrightBrowserExecutor — headless Chromium
│   ├── api/
│   │   ├── server.py             # FastAPI REST + WebSocket server (v1.4.0)
│   │   ├── models.py             # Shared Pydantic models
│   │   ├── store.py              # Abstract TaskStore + create_store() factory
│   │   ├── store_memory.py       # MemoryTaskStore (default)
│   │   ├── store_firestore.py    # FirestoreTaskStore (TASK_STORE=firestore)
│   │   ├── session_routes.py     # ADK session endpoints
│   │   ├── webpilot_routes.py    # WebPilot WS endpoint + session REST + TTS
│   │   └── webpilot_models.py    # WebPilotAction, WebPilotSession, InterruptionType
│   ├── metrics.py                # Cloud Monitoring fire-and-forget
│   ├── tracing.py                # OTel + Cloud Trace
│   ├── storage.py                # GCS screenshot upload → signed URLs
│   └── logging_config.py         # JSON structured logging
│
├── webpilot-extension/
│   ├── manifest.json             # MV3 v1.1.0
│   ├── background.js             # SW: WS owner, screenshot, action dispatch, session retry
│   ├── content.js                # DOM executor: click, type, scroll, key (shadow DOM + React)
│   ├── icons/                    # 16/48/128px
│   └── sidebar/
│       ├── package.json / vite.config.js
│       ├── index.html / main.jsx / App.jsx
│       ├── components/
│       │   ├── TaskInput.jsx         # Textarea + mic button (hold-to-speak)
│       │   ├── ActionLog.jsx         # Step log with ✓/✗ per action
│       │   ├── ConfirmCard.jsx       # Proceed/Cancel for irreversible actions
│       │   └── StatusIndicator.jsx   # Colored dot: green/amber/red/grey
│       └── hooks/
│           ├── useVoiceInput.js      # SpeechRecognition, 3 restart attempts
│           ├── useVoiceOutput.js     # Gemini TTS + Web Speech fallback
│           └── useWebSocket.js       # chrome.runtime bridge to background.js
│
├── tests/
│   ├── test_agent.py             # 16 integration tests (require Chromium)
│   ├── test_api.py               # 18 API tests
│   ├── test_api_extended.py      # 6 extended API tests
│   ├── test_webpilot_api.py      # 22 WebPilot unit tests
│   ├── test_webpilot_e2e.py      # 24 WebPilot e2e tests
│   ├── test_sessions.py          # 13 session tests
│   ├── test_clarifier.py         # 7 clarifier tests
│   ├── test_observability.py     # 11 observability tests
│   ├── test_store_firestore.py   # 8 Firestore store tests
│   ├── dashboard/index.html      # React 18 + Tailwind test dashboard
│   ├── dashboard_server.py       # Dashboard HTTP server (port 3333)
│   ├── agent_runner.py           # Pytest orchestrator for dashboard
│   ├── judge_runner.py           # Claude Sonnet judge evaluator
│   ├── judge_prompt.md           # Judge evaluation prompt
│   └── load/locustfile.py        # Locust load test scenarios
│
├── .github/workflows/            # CI/CD pipeline
├── Dockerfile                    # Python 3.12-slim + Playwright Chromium
├── docker-compose.yml
├── deploy.sh                     # Cloud Run deploy script
├── requirements.txt              # 21 Python packages
├── requirements-dev.txt
├── pyproject.toml
├── cloudbuild.yaml
└── CLAUDE.md                     # Claude Code project guide
```

---

## 5. ACTION SCHEMA

### 5.1 WebPilotAction (Pydantic Model)

**Action Types**: `click`, `type`, `scroll`, `wait`, `navigate`, `key`, `done`, `confirm_required`, `captcha_detected`, `login_required`

**Required Fields**:
| Field | Type | Description |
|---|---|---|
| observation | string | What the agent sees on screen |
| plan | string | What the agent intends to do next |
| steps_completed | int | Count of completed steps |
| action | string | Action type to execute |
| narration | string | What agent says out loud |
| action_label | string | Short label for sidebar log |
| is_irreversible | boolean | True if action cannot be undone |

**Optional Fields** (depend on action type):
| Field | Type | Used By |
|---|---|---|
| x, y | int | click |
| text | string | type, key |
| url | string | navigate |
| direction | "up"/"down" | scroll |
| duration | int (ms) | wait |

### 5.2 Interruption Types
| Type | Keywords | Behavior |
|---|---|---|
| ABORT | stop, abort, quit, never mind, nevermind, forget it, forget about it | Immediate stop, no Gemini call |
| REDIRECT | instead, new goal, start over, different, actually | Clear history, replan from scratch |
| REFINEMENT | (default) | Merge constraint with original intent |

---

## 6. API ENDPOINTS

| Method | Path | Description |
|---|---|---|
| POST | /navigate | Start a task, returns task_id |
| GET | /tasks | List all tasks |
| GET | /tasks/{task_id} | Poll status/result |
| DELETE | /tasks/{task_id} | Cancel a running task |
| WS | /ws/{task_id} | Stream step events in real time |
| POST | /screenshot | One-shot screenshot + Gemini analysis |
| POST | /clarify | Get clarifying questions for ambiguous tasks |
| GET | /health | Health check |
| POST | /webpilot/sessions | Create WebPilot session |
| DELETE | /webpilot/sessions/{id} | End WebPilot session |
| WS | /webpilot/ws/{session_id} | WebPilot real-time action loop |
| POST | /webpilot/tts | Gemini TTS narration → base64 WAV audio |
| POST | /sessions | Create ADK session |
| POST | /sessions/{id}/step | Send screenshot → get ActionPlan (ADK) |
| POST | /sessions/{id}/events | Log client-side telemetry |
| DELETE | /sessions/{id} | End ADK session |

---

## 7. GEMINI MODELS & CONFIGURATION

| Purpose | Model | Thinking Budget | Override Env Var |
|---|---|---|---|
| Core vision (agent loop) | gemini-2.5-flash | 1024 | GEMINI_MODEL |
| WebPilot Legacy handler | gemini-2.5-flash | 1024 | GEMINI_MODEL |
| WebPilot Live API | gemini-live-2.5-flash-preview | 1024 | GEMINI_LIVE_MODEL |
| TTS narration | gemini-2.5-flash-preview-tts | — | (hardcoded) |
| Task clarifier | gemini-2.5-flash | — | (hardcoded) |
| Action verification | gemini-2.5-flash | 512 | — |

**SDK**: `google-genai >= 0.8` — uses `types.Part.from_bytes(data=bytes, mime_type="image/png")` for images.

**History alternation**: Gemini requires strict `user → model → user → model` turn alternation. The vision client stores `_last_user_turn` and prepends it before each model turn.

---

## 8. LATENCY MANAGEMENT

### 8.1 Latency Budget Per Action
```
Screenshot capture:     ~50ms
Network to Cloud Run:   ~50ms
Gemini API response:    ~800ms - 1500ms (with thinking_budget=1024)
Action execution:       ~50ms
Page settle wait:       ~1500ms
─────────────────────────────
Total per action:       ~2.5s - 3.2s
```

### 8.2 Making Latency Feel Intentional
- Voice narration begins **immediately** when action is dispatched
- Sidebar shows action label **before** page responds
- "Thinking..." animation plays only during Gemini call
- Agent always sounds like it's working, never frozen

---

## 9. TECHNICAL CONSTRAINTS & SOLUTIONS

| Constraint | Solution |
|---|---|
| CAPTCHAs blocking agent | `captcha_detected` action type → pause → ask user to solve → resume |
| Login-required pages | `login_required` action type → ask user to log in → resume |
| Dynamic pages (infinite scroll) | Screenshot after scroll + scroll detection in Gemini prompt |
| Slow page loads | `waitForTabLoad()`: tab status "complete" + 1.5s settle (15s timeout) |
| Shadow DOM / Canvas | Pure visual approach + shadow DOM traversal in content.js |
| Popup modals / cookie banners | Gemini detects and dismisses before proceeding |
| Content script loss after navigate | Re-inject via `chrome.scripting.executeScript` before each action |
| React input fields | React-compatible: dispatchEvent with input + change events |
| Cloud Run WS frame limits | Known issue with screenshots >150KB; investigation ongoing |
| Service worker dormancy | Keep-alive alarm every 25s in background.js |

---

## 10. NON-FUNCTIONAL REQUIREMENTS

### 10.1 Performance
- WebSocket connection established on extension install/startup with retry backoff
- First action taken within 3 seconds of task submission
- Voice narration latency < 200ms from action trigger

### 10.2 Security
- Gemini API keys stored only on Cloud Run, never in extension
- APIKeyMiddleware + RateLimitMiddleware on all endpoints
- CORS default: `chrome-extension://*` (not wildcard)
- No user data stored beyond session duration (MAX_SESSION_DURATION = 1800s)
- No screenshot data persisted after session ends

### 10.3 Reliability
- Auto-retry failed Gemini calls up to 3 times
- Graceful degradation if voice input unavailable (text fallback)
- WebSocket session retry with backoff if backend is down
- Dedicated ThreadPoolExecutor prevents thread exhaustion under retry
- Message queue `.catch()` prevents crashed handler from blocking future messages

---

## 11. ENVIRONMENT VARIABLES

| Variable | Required | Default | Description |
|---|---|---|---|
| GOOGLE_API_KEY | Yes | — | Gemini API key |
| GOOGLE_CLOUD_PROJECT | Deploy only | — | GCP project ID |
| GOOGLE_CLOUD_REGION | Deploy only | us-central1 | Cloud Run region |
| GEMINI_MODEL | No | gemini-2.5-flash | Legacy handler model |
| GEMINI_LIVE_MODEL | No | gemini-live-2.5-flash-preview | Live API model |
| BROWSER_HEADLESS | No | true | Run Chromium headless |
| MAX_CONCURRENT_TASKS | No | 5 | Semaphore limit + thread pool size |
| BROWSER_WIDTH/HEIGHT | No | 1280x800 | Viewport size |
| TASK_STORE | No | memory | `memory` or `firestore` |
| GCS_BUCKET | No | — | GCS bucket for screenshots |
| MAX_SESSION_DURATION | No | 1800 | WebPilot session timeout (seconds) |
| MAX_RETRIES | No | 3 | Identical screenshots before stuck hint |
| ACTION_LOOP_TIMEOUT | No | 120 | Hard timeout for action loop (seconds) |
| MAX_LOOP_STEPS | No | 30 | Max steps per action loop |
| API_KEYS | No | — | Comma-separated API keys (auth disabled if unset) |
| CORS_ORIGINS | No | chrome-extension://* | Allowed CORS origins |
| RATE_LIMIT_RPM | No | 60 | Requests per minute per key |

---

## 12. DEPLOYMENT

### Cloud Run
```bash
# Deploy
export GOOGLE_CLOUD_PROJECT=<project-id>
export GOOGLE_API_KEY=<key>
chmod +x deploy.sh && ./deploy.sh
```

**Config**: 2 vCPU, 2 GiB RAM, min 0/max 5 instances, 300s timeout, session affinity enabled.

**Uvicorn WS**: `--ws-ping-interval 0 --ws-ping-timeout 0 --ws-per-message-deflate false` (Cloud Run proxy limitations).

### Extension
```bash
cd webpilot-extension/sidebar && npm install && npm run build
# Chrome → chrome://extensions → Load unpacked → select webpilot-extension/
```

**Chrome Setup**: Site access → "On all sites"; Mic access → allow extension origin.

**Backend URL**: configurable via `chrome.storage.sync.set({backendUrl: "http://localhost:8080"})`.

### Docker (local)
```bash
docker-compose up --build
```

---

## 13. DEMO SCRIPT (3 Minutes)

### Demo Scene 1 — It Just Works (60 seconds)
```
User: "Find me a non-stop flight from Austin to Tokyo
       next Friday under $400"

Agent: "Got it, searching for flights from Austin to Tokyo"
→ Opens google.com/flights
→ Clicks origin field, types "Austin"
→ Clicks destination field, types "Tokyo"
→ Sets date to next Friday
→ Applies non-stop filter

Agent: "Found some great options. Cheapest non-stop
        is ANA at $387, departing Friday 11pm,
        arriving Sunday 5am. Want me to book it?"
```

### Demo Scene 2 — Live Interruption (30 seconds)
```
[Agent is mid-search, filling in dates]

User: "Actually, make it a round trip, return the following Sunday"

Agent: "Got it, updating to round trip"
→ Classified as REDIRECT (keyword: "actually")
→ Agent replans from current screen
→ Clicks "Round trip" toggle
→ Sets return date
→ Continues search
```

### Demo Scene 3 — Completely Different Website (60 seconds)
```
User: "Now find me the top-rated noise cancelling
       headphones under $200 on Amazon"

Agent: "Sure, searching Amazon for headphones"
→ Navigates to amazon.com
→ Types search query
→ Applies price filter < $200
→ Sorts by customer rating
→ Reads top 3 results

Agent: "Top pick is Sony WH-1000XM4 at $199,
        4.4 stars with 89,000 reviews.
        Want me to add it to your cart?"

[Universal — works on any website without pre-built connectors.]
```

### Demo Scene 4 — Form Filling with Confirmation (30 seconds)
```
User: "Fill out the contact form on this page
       with my name John Smith and email john@example.com"

Agent: "Filling in the contact form now"
→ Clicks name field, types "John Smith"
→ Clicks email field, types "john@example.com"
→ Detects submit button → is_irreversible = true

Agent: "All filled in. Should I go ahead and submit?"
→ Sidebar shows ConfirmCard
→ User clicks ✅ or says "yes" → submits
→ User clicks ⛔ or says "stop" → cancels
```

---

## 14. SUCCESS CRITERIA

The hackathon demo is considered successful if:

1. ✅ Agent completes a multi-step flight search end-to-end from voice command
2. ✅ Agent correctly handles a mid-task voice interruption without restarting
3. ✅ Agent successfully navigates a completely different website (proving universality)
4. ✅ Agent pauses and asks for confirmation before any irreversible action
5. ✅ Full voice narration throughout with no silent gaps > 3 seconds
6. ✅ All processing hosted on Google Cloud Run
7. ✅ Gemini multimodal vision used for all UI interpretation (no DOM access)

---

## 15. DEPENDENCIES

### Extension (sidebar/package.json)
```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "@vitejs/plugin-react": "^4.3.1",
  "vite": "^5.4.2"
}
```

### Server (requirements.txt — 21 packages)
```
google-genai>=0.8.0
google-adk>=0.3.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
playwright>=1.48.0
pydantic>=2.9.0
# + OTel packages for Cloud Trace
```

---

## 16. TEST SUITE

125 total tests (109 non-browser pass without Chromium):

| File | Count | Category |
|---|---|---|
| test_api.py | 18 | REST API |
| test_api_extended.py | 6 | Extended API |
| test_webpilot_api.py | 22 | WebPilot unit |
| test_webpilot_e2e.py | 24 | WebPilot e2e |
| test_sessions.py | 13 | ADK sessions |
| test_clarifier.py | 7 | Task clarifier |
| test_observability.py | 11 | Metrics/tracing |
| test_store_firestore.py | 8 | Firestore store |
| test_agent.py | 16 | Integration (requires Chromium) |

**Test dashboard**: `python tests/dashboard_server.py` → `http://localhost:3333`

---

*End of PRD — WebPilot v2.0*
*Built for Google Cloud x Gemini Hackathon — UI Navigator Track*
