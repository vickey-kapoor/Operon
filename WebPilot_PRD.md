# WebPilot — AI Browser Navigator
## Product Requirements Document (PRD)
### Version 1.0 | Google Cloud x Gemini Hackathon Submission

---

## 1. OVERVIEW

### 1.1 Product Summary
WebPilot is a Chrome browser extension that lets users control any website using natural voice or text commands. It uses Gemini's multimodal vision to observe the browser screen as a human would — without relying on APIs or DOM scraping — and executes actions (clicks, typing, scrolling) to complete tasks on behalf of the user.

### 1.2 Hackathon Track
**UI Navigator ☸️** — Visual UI Understanding & Interaction

### 1.3 Mandatory Tech Requirements
- Gemini multimodal API to interpret screenshots and output executable actions
- Agent hosted on Google Cloud Run
- Gemini Live API for real-time voice interaction and interruption handling

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
- **Voice:** User holds mic button and speaks naturally
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
Agent speaks at every meaningful step. Tone: short, confident, human.

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
Three types, each handled differently:

**Type 1 — Refinement** (e.g. "make it non-stop only")
- Agent finishes current micro-action
- Applies new constraint
- Continues from current screen state

**Type 2 — Redirect** (e.g. "forget flights, find a train")
- Agent stops immediately
- Confirms new goal verbally
- Starts fresh from current screen

**Type 3 — Abort** (user says "stop" or clicks ⛔ button)
- Agent stops mid-action instantly
- Says: "Stopped. What would you like to do?"
- Returns to idle state, awaiting new instruction

### 2.7 Confirmation Gate — Irreversible Actions
Before executing any irreversible action (booking, submitting a form, deleting, purchasing):
1. Agent pauses
2. Sidebar shows confirmation card with action summary
3. Agent speaks the confirmation prompt
4. User must explicitly say "yes" / "proceed" or click confirm
5. If user says "wait" or "stop" → action cancelled, agent replans

### 2.8 Error Handling — Auto-Retry (Silent)
```
Action attempted
      ↓
Did page respond as expected? (screenshot comparison)
      ↓
  NO → wait 1.5s → take new screenshot → retry same action
      ↓
  Still wrong? → try alternative action
  (e.g. if click failed → try keyboard shortcut equivalent)
      ↓
  Still wrong after 3 attempts?
      → narrate: "Having trouble with this, trying a different approach"
      → replan from current screen state using Gemini
```
User never sees failure states. Agent always appears to be progressing.

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture
```
USER BROWSER
│
├── Chrome Extension (Manifest V3)
│   ├── Sidebar UI (React + Tailwind)
│   ├── Voice Input (Web Speech API)
│   ├── Voice Output (Web Speech Synthesis / Gemini TTS)
│   ├── Screen Capture (chrome.tabs.captureVisibleTab)
│   └── Action Executor (content script)
│
│ [HTTPS WebSocket — persistent connection]
│
├── Google Cloud Run (Backend)
│   ├── Session Manager
│   ├── Gemini Live API Handler
│   ├── Action Decision Engine
│   └── Task History Store (in-memory per session)
│
│ [Gemini API]
│
└── Gemini 2.0 Flash / Pro (Vision + Reasoning)
    ├── Receives screenshot + intent + history
    ├── Outputs next action as structured JSON
    └── Handles interruptions via context injection
```

### 3.2 Component Responsibilities

| Component | Location | Responsibility |
|---|---|---|
| Sidebar UI | Extension frontend | User input, action log, confirmation UI |
| Voice input | Extension frontend | Capture and transcribe speech |
| Voice output | Extension frontend | Speak agent narration |
| Screen capture | Extension background | Capture visible tab as base64 PNG |
| Action executor | Extension content script | Simulate clicks, typing, scrolling |
| Session manager | Cloud Run | Maintain task state and history per user |
| Gemini handler | Cloud Run | Send prompts + screenshots, parse responses |
| API keys | Cloud Run only | Never exposed to frontend |

### 3.3 Data Flow — Step by Step

**Step 1: User Input**
```
User speaks or types intent
→ Web Speech API transcribes audio to text
→ Extension captures current tab screenshot (base64 PNG)
→ Sends to Cloud Run via WebSocket:
{
  "session_id": "abc123",
  "intent": "Find flight Austin to Tokyo under $400",
  "screenshot": "<base64>",
  "history": []
}
```

**Step 2: Cloud Run → Gemini**
```
Cloud Run constructs prompt:

SYSTEM:
You are WebPilot, a browser agent that controls websites visually.
You receive screenshots and must output the single next action to take.
Never output multiple actions. One action at a time.
Always output valid JSON only.

USER:
Goal: {intent}
Previous actions: {history}
Current screen: {screenshot}

What is the single next action?
Respond ONLY in this JSON format:
{
  "action": "click" | "type" | "scroll" | "wait" | "navigate" | "done" | "confirm_required",
  "x": number,              // pixel x coordinate (if click)
  "y": number,              // pixel y coordinate (if click)
  "text": "string",         // text to type (if type)
  "url": "string",          // url to navigate to (if navigate)
  "direction": "up"|"down", // scroll direction (if scroll)
  "duration": number,       // wait duration in ms (if wait)
  "narration": "string",    // what agent says out loud
  "action_label": "string", // short label for sidebar action log
  "is_irreversible": boolean // true if action cannot be undone
}
```

**Step 3: Gemini Response**
```json
{
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
  → Send confirmation_required event to extension
  → Wait for user confirm before sending action

If is_irreversible = false:
  → Send action directly to extension
```

**Step 5: Extension Executes Action**
```javascript
// content.js
chrome.runtime.onMessage.addListener((message) => {
  const { action, x, y, text, direction } = message;
  
  if (action === 'click') {
    const el = document.elementFromPoint(x, y);
    el?.click();
  }
  
  if (action === 'type') {
    const el = document.activeElement;
    el.focus();
    el.value = text;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  
  if (action === 'scroll') {
    window.scrollBy(0, direction === 'down' ? 400 : -400);
  }
  
  if (action === 'navigate') {
    window.location.href = message.url;
  }
});
```

**Step 6: Loop**
```
Action executed
→ Wait 1.5s for page to settle
→ Capture new screenshot
→ Send to Cloud Run with updated history
→ Get next action from Gemini
→ Execute
→ Repeat until action = "done"
```

**Step 7: Interruption**
```
User speaks mid-task
→ Web Speech API detects new speech
→ Extension sends interruption event to Cloud Run:
{
  "session_id": "abc123",
  "type": "interruption",
  "new_instruction": "make it non-stop only",
  "screenshot": "<current screen base64>"
}

→ Cloud Run injects into Gemini context:
"INTERRUPTION: User says: 'make it non-stop only'
 Replan from current screen. Previous goal still applies."

→ Gemini replans from current screen state
→ New action sent to extension
```

---

## 4. FILE STRUCTURE

```
webpilot/
├── extension/
│   ├── manifest.json
│   ├── background.js          # Screen capture, WebSocket, tab management
│   ├── content.js             # Action executor (injected into pages)
│   ├── sidebar/
│   │   ├── index.html
│   │   ├── App.jsx            # Main sidebar React component
│   │   ├── components/
│   │   │   ├── TaskInput.jsx      # Voice + text input
│   │   │   ├── ActionLog.jsx      # Live action feed
│   │   │   ├── ConfirmCard.jsx    # Confirmation gate UI
│   │   │   └── StatusIndicator.jsx
│   │   └── hooks/
│   │       ├── useVoiceInput.js   # Web Speech API hook
│   │       ├── useVoiceOutput.js  # TTS narration hook
│   │       └── useWebSocket.js    # Cloud Run connection
│   └── icons/
│       └── icon128.png
│
├── server/
│   ├── main.py                # FastAPI Cloud Run server
│   ├── session_manager.py     # Per-session state management
│   ├── gemini_handler.py      # Gemini API prompting + parsing
│   ├── action_validator.py    # Irreversible action detection
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

## 5. DETAILED FILE SPECIFICATIONS

### 5.1 manifest.json
```json
{
  "manifest_version": 3,
  "name": "WebPilot",
  "version": "1.0",
  "description": "AI agent that navigates any website by voice or text",
  "permissions": [
    "activeTab",
    "tabs",
    "scripting",
    "storage",
    "tabCapture"
  ],
  "host_permissions": ["<all_urls>"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_end"
    }
  ],
  "action": {
    "default_popup": "sidebar/index.html"
  },
  "side_panel": {
    "default_path": "sidebar/index.html"
  },
  "commands": {
    "toggle-sidebar": {
      "suggested_key": {
        "default": "Ctrl+Shift+A",
        "mac": "Command+Shift+A"
      },
      "description": "Toggle WebPilot sidebar"
    }
  }
}
```

### 5.2 background.js — Key Functions
```javascript
// Core responsibilities:
// 1. Capture visible tab as base64 PNG
// 2. Maintain WebSocket connection to Cloud Run
// 3. Route messages between sidebar, content script, and server
// 4. Handle tab state changes

async function captureScreenshot() {
  const tab = await getCurrentActiveTab();
  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: 'png',
    quality: 80
  });
  return dataUrl.split(',')[1]; // return base64 only
}

// WebSocket connection to Cloud Run
const ws = new WebSocket('wss://your-cloudrun-url/ws');

// Message routing
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'EXECUTE_TASK') {
    handleTask(message.intent);
  }
  if (message.type === 'INTERRUPTION') {
    handleInterruption(message.instruction);
  }
  if (message.type === 'STOP') {
    handleStop();
  }
});
```

### 5.3 server/main.py — FastAPI Server
```python
# FastAPI server on Cloud Run
# Key endpoints:
# POST /task - Start new task session
# WebSocket /ws/{session_id} - Persistent connection per user
# POST /interrupt/{session_id} - Handle mid-task interruption
# POST /confirm/{session_id} - User confirmed irreversible action

# Key behaviors:
# - Maintain session history (list of actions taken)
# - Call Gemini with screenshot + intent + history
# - Parse Gemini JSON response
# - Validate action safety (irreversible detection)
# - Stream actions back to extension via WebSocket
```

### 5.4 server/gemini_handler.py
```python
# Key responsibilities:
# - Construct Gemini prompt with screenshot + context
# - Call Gemini 2.0 Flash multimodal API
# - Parse and validate JSON response
# - Handle Gemini errors gracefully
# - Maintain conversation context for interruptions

SYSTEM_PROMPT = """
You are WebPilot, an AI browser agent that controls websites visually.
You receive a screenshot of a browser tab and a user goal.
You must output the single next action to take to progress toward the goal.

Rules:
- Output ONE action at a time, never multiple
- Use pixel coordinates relative to the screenshot dimensions
- If the goal is complete, output action: "done"
- If an action would be irreversible (booking, purchase, form submit, delete), set is_irreversible: true
- Keep narration short, confident, and human-sounding
- If you cannot determine the next action, output action: "wait" with a short duration

Always respond with valid JSON only. No markdown, no explanation.
"""
```

---

## 6. GEMINI PROMPT ENGINEERING

### 6.1 Action Decision Prompt
```
SYSTEM: {SYSTEM_PROMPT}

USER:
=== GOAL ===
{user_intent}

=== PREVIOUS ACTIONS ===
{json.dumps(session.history, indent=2)}

=== CURRENT SCREEN ===
[Screenshot attached as image]

=== SCREEN DIMENSIONS ===
Width: {width}px, Height: {height}px

What is the single next action?
```

### 6.2 Interruption Replan Prompt
```
=== INTERRUPTION ===
The user has given a new instruction mid-task: "{new_instruction}"

Original goal: {original_intent}
Actions taken so far: {history}
Current screen: [Screenshot attached]

Replan from the current screen state incorporating the new instruction.
What is the single next action?
```

### 6.3 Task Completion Detection
Gemini returns `"action": "done"` when it assesses the goal is met.
Additionally, Cloud Run checks after every `done` response:
- Does the current screenshot match the expected outcome?
- If not, continue loop with "goal not yet achieved" injected into context.

---

## 7. LATENCY MANAGEMENT

### 7.1 Latency Budget Per Action
```
Screenshot capture:     ~50ms
Network to Cloud Run:   ~50ms  
Gemini API response:    ~800ms - 1500ms
Action execution:       ~50ms
Page settle wait:       ~1500ms
─────────────────────────────
Total per action:       ~2.5s - 3.2s
```

### 7.2 Making Latency Feel Intentional
- Voice narration begins **immediately** when action is dispatched
- Sidebar shows action label **before** page responds
- "Thinking..." animation plays only during Gemini call
- Agent always sounds like it's working, never frozen

---

## 8. DEMO SCRIPT (3 Minutes)

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
→ Agent replans from current screen
→ Clicks "Round trip" toggle
→ Sets return date
→ Continues search

[Judges see the agent adapt in real time without starting over]
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

[Judges realize: this is NOT a flight bot. It's truly universal.]
```

### Demo Scene 4 — Form Filling (30 seconds)
```
User: "Fill out the contact form on this page 
       with my name John Smith and email john@example.com"

Agent: "Filling in the contact form now"
→ Clicks name field, types "John Smith"  
→ Clicks email field, types "john@example.com"
→ Pauses before submit

Agent: "All filled in. Should I go ahead and submit?"
```

---

## 9. TECHNICAL CONSTRAINTS & SOLUTIONS

| Constraint | Solution |
|---|---|
| CAPTCHAs blocking agent | Detect CAPTCHA in screenshot → pause → ask user to solve → resume |
| Dynamic pages (infinite scroll) | Screenshot after scroll + scroll detection in Gemini prompt |
| Login-required pages | Detect login page → ask user to log in → resume after |
| Very long pages | Scroll incrementally, capture partial screenshots |
| Slow page loads | Implement page-stable detection before taking action screenshot |
| Shadow DOM / Canvas elements | Pure visual approach means these are invisible — handled naturally |
| Popup modals / cookie banners | Gemini detects and dismisses before proceeding to real content |

---

## 10. NON-FUNCTIONAL REQUIREMENTS

### 10.1 Performance
- WebSocket connection established within 500ms of extension open
- First action taken within 3 seconds of task submission
- Voice narration latency < 200ms from action trigger

### 10.2 Security
- Gemini API keys stored only on Cloud Run, never in extension
- No user data stored beyond session duration
- Session auto-clears after 30 minutes of inactivity
- No screenshot data persisted after session ends

### 10.3 Reliability
- Auto-retry failed Gemini calls up to 3 times
- Graceful degradation if voice input unavailable (text fallback)
- WebSocket auto-reconnect on disconnect

---

## 11. OUT OF SCOPE (Hackathon Version)

- Multi-tab simultaneous task execution
- User accounts / saved task history
- Scheduled / recurring task automation
- Mobile browser support
- Firefox / Safari support
- Fine-tuned Gemini model (uses base Gemini 2.0 Flash)

---

## 12. SUCCESS CRITERIA

The hackathon demo is considered successful if:

1. ✅ Agent completes a multi-step flight search end-to-end from voice command
2. ✅ Agent correctly handles a mid-task voice interruption without restarting
3. ✅ Agent successfully navigates a completely different website (proving universality)
4. ✅ Agent pauses and asks for confirmation before any irreversible action
5. ✅ Full voice narration throughout with no silent gaps > 3 seconds
6. ✅ All processing hosted on Google Cloud Run
7. ✅ Gemini multimodal vision used for all UI interpretation (no DOM access)

---

## 13. ENVIRONMENT SETUP

### Cloud Run
```bash
# Deploy server
gcloud run deploy webpilot-server \
  --source ./server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

### Extension
```bash
# Install dependencies
cd extension/sidebar
npm install

# Build sidebar
npm run build

# Load unpacked extension in Chrome:
# chrome://extensions → Developer mode → Load unpacked → select /extension
```

### Environment Variables (Cloud Run)
```
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash-exp
MAX_SESSION_DURATION=1800
SCREENSHOT_QUALITY=80
MAX_RETRIES=3
```

---

## 14. DEPENDENCIES

### Extension
```json
{
  "react": "^18.0.0",
  "tailwindcss": "^3.0.0",
  "lucide-react": "^0.383.0"
}
```

### Server
```
fastapi==0.104.0
uvicorn==0.24.0
websockets==12.0
google-generativeai==0.3.0
pillow==10.1.0
python-dotenv==1.0.0
```

---

*End of PRD — WebPilot v1.0*
*Built for Google Cloud x Gemini Hackathon — UI Navigator Track*
