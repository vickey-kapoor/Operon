# Desktop Mode — Feature PRD, HLD, and LLD

**Document version:** 1.0  
**Date:** 2026-03-15  
**Status:** Ready for implementation  
**Target:** Hackathon — Gemini / Google GenAI / Google Cloud

---

## PRD — Product Requirements Document

### Problem Statement

UI Navigator today controls a single Chromium browser tab managed internally by Playwright. The vision pipeline (screenshot → Gemini → action plan → execute) is completely source-agnostic — it does not care where the screenshot came from or what executes the resulting click. Yet users who want to automate desktop applications (Figma, VS Code, Slack, Excel, Photoshop, native system dialogs, installers) are entirely blocked because the only executor speaks the Playwright API.

Desktop Mode extends the executor layer to target the **entire screen** — capturing via `mss` and dispatching input via `pyautogui` — while reusing 100 % of the existing Gemini vision pipeline, session model, WebSocket protocol, and API surface. The same `WebPilotAction` JSON that the browser extension understands maps directly to desktop actions, requiring no protocol changes on the frontend.

### User Stories

| ID | As a… | I want to… | So that… |
|---|---|---|---|
| US-1 | Power user | Give a voice command and have the AI complete a task in any open desktop app | I don't have to leave my current application context |
| US-2 | Developer | Automate a multi-app workflow (e.g. copy data from Excel → paste into a web form → send Slack message) | I can script cross-application sequences without coding |
| US-3 | QA engineer | Run the same AI-driven test scenarios against a locally installed native app | CI coverage extends beyond the browser |
| US-4 | Demo presenter | Show the hackathon judges a live AI agent controlling VS Code, opening files, running commands | The demo is visually impressive and differentiated |
| US-5 | API consumer | POST a desktop task through the same `/navigate` API with `mode=desktop` | My existing integration requires no code changes |

### Functional Requirements

**FR-1 — Desktop screenshot capture**  
The system MUST capture full-screen screenshots using `mss` as base64-encoded PNG. Capture resolution equals the native display resolution. No crop or resize by default.

**FR-2 — Desktop action execution**  
The system MUST execute the following action types against the host OS using `pyautogui`:
- `click` — left-click at absolute pixel coordinates
- `right_click` — right-click at absolute pixel coordinates  
- `double_click` — double-click at absolute pixel coordinates
- `type` — keyboard type with configurable inter-key delay
- `key` — single key or chord (e.g. "ctrl+c", "alt+F4", "win")
- `scroll` — scroll wheel at a coordinate in a direction
- `move` — move the mouse cursor without clicking (hover)
- `wait` — sleep for a specified duration in milliseconds
- `done` — signal task completion

**FR-3 — Coordinate passthrough**  
Gemini returns pixel coordinates relative to the screenshot dimensions. The executor MUST scale coordinates from screenshot dimensions to actual screen dimensions if they differ (for high-DPI / Retina displays where mss captures at logical resolution but pyautogui works in physical pixels or vice versa).

**FR-4 — Existing action types preserved**  
The `navigate`, `screenshot`, `confirm_required`, `captcha_detected`, and `login_required` action types remain in the `WebPilotAction` schema. In Desktop Mode, `navigate` is rejected with a descriptive error (it is browser-specific). `screenshot` triggers an immediate re-capture.

**FR-5 — Mode selection**  
A new `mode` field on `POST /navigate` and `POST /desktop/start` accepts `"browser"` (default, existing behavior) or `"desktop"`. Desktop Mode MUST NOT launch Playwright or Chromium.

**FR-6 — WebPilot Desktop Session**  
A new WebSocket endpoint `/desktop/ws/{session_id}` mirrors the WebPilot WS protocol exactly. The client sends `{"type": "task", "intent": str, "screenshot": base64}` and receives `{"type": "action", ...}` or `{"type": "done", ...}`. The extension (or a desktop client app) executes the action and sends back `{"type": "screenshot", "screenshot": base64}`.

**FR-7 — System context injection**  
Gemini MUST be told it is controlling a desktop environment. The system prompt MUST include: active OS name, that coordinates are screen-relative (not browser-relative), and that `navigate` is not available. The word "browser" MUST NOT appear in the desktop system prompt.

**FR-8 — Stuck detection**  
The existing MD5-hash stuck detection mechanism in `webpilot_routes.py::_run_action_loop_inner` MUST apply to Desktop Mode unchanged. After 3 identical screenshots, the `stuck=True` hint is injected into Gemini.

**FR-9 — Safety — confirmation for destructive actions**  
All existing `confirm_required` and `is_irreversible` rules in `WEBPILOT_SYSTEM_PROMPT` apply to Desktop Mode. File deletions, form submits, and send-message actions are irreversible.

**FR-10 — Safety — failsafe**  
`pyautogui` `FAILSAFE` MUST be `True` (the default). Moving the mouse to the top-left screen corner aborts. `PAUSE` MUST be set to 0.05 seconds between actions to prevent runaway input floods.

**FR-11 — Platform support**  
Desktop Mode MUST work on Windows 10/11 and macOS 13+. Linux (X11) is supported on a best-effort basis. ARM Mac (Apple Silicon) is supported via Rosetta-compatible `pyautogui`.

**FR-12 — Concurrent session isolation**  
Each desktop session uses the same semaphore as the browser executor (`_semaphore`). Desktop sessions are gated identically to prevent parallelism issues (multiple agents writing to the same screen simultaneously).

**FR-13 — REST trigger endpoint**  
`POST /desktop/start` accepts `{intent, max_steps}` and returns `{session_id, status}`. This is a fire-and-trigger endpoint: it creates a session and begins the loop server-side by taking an initial screenshot itself (no screenshot sent by client). This is the API-consumer path for non-interactive use.

**FR-14 — Health check**  
`GET /health` response MUST include `desktop_sessions_active` count.

### Non-Functional Requirements

**NFR-1 — Latency**  
Round-trip per desktop step (screenshot → Gemini → action dispatch) MUST be ≤ 6 s on a modern desktop with a 1080p display. The screenshot capture + encode step MUST complete in ≤ 200 ms.

**NFR-2 — Reliability**  
`pyautogui` exceptions (e.g., coordinate out of bounds) MUST be caught and returned as `ActionResult(success=False, error=...)` rather than crashing the session.

**NFR-3 — Security — local execution only**  
Desktop Mode may ONLY run when the server is running on `localhost` or when `DESKTOP_MODE_ENABLED=true` is explicitly set. When deployed to Cloud Run, Desktop Mode returns `503 Service Unavailable` because Cloud Run containers have no physical display.

**NFR-4 — Resource usage**  
`mss` screenshot instances MUST be reused across captures within a session (do not re-initialize `mss.mss()` on every call). The PIL Image MUST be discarded after base64 encoding — do not accumulate in memory.

**NFR-5 — Testability**  
All new classes MUST be testable with a mock pyautogui backend. A `DesktopExecutorStub` (parallel to the existing `WebPilotStubHandler`) MUST be provided for CI use where no display is available.

**NFR-6 — Observability**  
Desktop Mode MUST emit the same Cloud Monitoring metrics as Browser Mode: `tasks_started`, `tasks_completed`, `tasks_failed`, `step_latency_ms`, `gemini_latency_ms`.

### Success Metrics

| Metric | Target |
|---|---|
| Task completion rate (demo tasks) | ≥ 70 % on predefined scenarios |
| Screenshot capture latency | ≤ 200 ms p95 |
| Gemini call success rate | ≥ 95 % (uses existing retry logic) |
| Action execution success rate | ≥ 90 % on well-defined coordinates |
| Hackathon demo: agent opens VS Code, types a file, runs it | Pass |
| Hackathon demo: agent fills a native OS dialog | Pass |

### Out of Scope

- Multi-monitor support (capture defaults to primary display; secondary monitor support is a follow-on)
- OCR-only mode (no Gemini) — all vision goes through Gemini
- Recording / replay of sessions (future)
- Accessibility API integration (e.g., AT-SPI, Windows UIA) — screenshot-only for now
- Audio output control (mute/volume) — not actionable via screenshot
- Drag-and-drop actions — follow-on feature
- GPU/game window capture — out of scope
- Cloud Run deployment of Desktop Mode — explicitly unsupported (no display)

### Hackathon Constraints

- Vision: **Gemini 2.5 Flash** via `google-genai` SDK (same as existing)
- Thinking budget: `thinking_budget=1024` on all calls (same as WebPilot)
- TTS narration: **gemini-2.5-flash-preview-tts** with Aoede voice (same endpoint as `/webpilot/tts`)
- GenAI SDK: `google-genai` (not `google-generativeai`) — same client reused
- Cloud: Cloud Monitoring metrics, Cloud Trace, optional Cloud Run guard
- No new cloud services introduced — stays within existing project budget

---

## High-Level Design (HLD)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  Chrome Extension (WebPilot)    Desktop Client App / Python Script  │
│  [sidebar → background.js]      [mss capture → REST / WS client]   │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │ WS /webpilot/ws/{id}          │ WS /desktop/ws/{id}
               │ (existing)                    │ (NEW)
               ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FASTAPI SERVER  (server.py)                   │
│                                                                     │
│  /webpilot/*  (webpilot_routes.py)   /desktop/*  (desktop_routes.py NEW) │
│                                                                     │
│  ┌────────────────────────────┐   ┌────────────────────────────┐   │
│  │   WebPilot Session Model   │   │   Desktop Session Model    │   │
│  │   WebPilotSession          │   │   DesktopSession (NEW)     │   │
│  └─────────────┬──────────────┘   └────────────┬───────────────┘   │
│                │                               │                   │
│                ▼                               ▼                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   SHARED HANDLER LAYER                         │ │
│  │  WebPilotHandler.get_next_action()  (webpilot_handler.py)      │ │
│  │  ← identical call, different system prompt for desktop         │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
│                             │                                       │
│                             ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   GEMINI VISION PIPELINE                       │ │
│  │  GeminiVisionClient  →  generate_content()  →  WebPilotAction  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                             │                                       │
│        ┌────────────────────┼────────────────────────┐             │
│        ▼                    ▼                         ▼             │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  Browser     │  │  Desktop         │  │  (future: mobile,     │ │
│  │  Executor    │  │  Executor (NEW)   │  │   cloud VM, etc.)     │ │
│  │  (Playwright)│  │  (pyautogui+mss) │  │                       │ │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Location | Role | Changed? |
|---|---|---|---|
| `GeminiVisionClient` | `src/agent/vision.py` | Calls Gemini with screenshot + text | No change |
| `ActionPlanner` | `src/agent/planner.py` | Parses Gemini JSON → ActionPlan | No change |
| `WebPilotHandler` | `src/agent/webpilot_handler.py` | Builds Gemini prompts, parses WebPilotAction | Minor: accept `mode` param for system prompt switch |
| `DesktopExecutor` | `src/executor/desktop.py` (NEW) | Captures screenshots, dispatches pyautogui actions | New |
| `BaseExecutor` | `src/executor/base.py` (NEW) | Abstract protocol shared by Browser + Desktop | New |
| `DesktopSession` | `src/api/desktop_models.py` (NEW) | Session state for desktop WS loop | New |
| `desktop_routes.py` | `src/api/desktop_routes.py` (NEW) | FastAPI router for /desktop/* endpoints | New |
| `desktop_system_prompt` | Inside `webpilot_handler.py` or separate constant | Desktop-specific Gemini system prompt | New constant |
| `server.py` | `src/api/server.py` | Lifespan: init desktop handler | Small addition |

### How Desktop Mode Fits Alongside Existing Modes

```
MODE TAXONOMY
═════════════
Browser Mode     — UINavigatorAgent (core.py) + PlaywrightBrowserExecutor
                   triggered via POST /navigate
                   runs headless Chromium, full multi-step loop internally

WebPilot Mode    — WebPilotHandler + Chrome Extension executor
                   triggered via WS /webpilot/ws/{session_id}
                   extension sends screenshots, server sends single actions

Desktop Mode     — WebPilotHandler (same) + DesktopExecutor
                   triggered via WS /desktop/ws/{session_id} OR POST /desktop/start
                   two sub-paths:
                   a) Interactive: client (desktop app / script) sends screenshots
                      like WebPilot — pure WS protocol
                   b) Server-side: server runs DesktopExecutor directly,
                      captures its own screenshots in a background loop
```

The critical insight: **WebPilot Mode and Desktop Mode share the identical WS protocol and Gemini call chain.** The only differences are:
1. The Gemini system prompt (swap "browser" framing for "desktop OS" framing)
2. The action executor (pyautogui instead of chrome extension content.js)
3. The `navigate` action type is rejected for desktop

### Data Flow — Interactive Desktop Mode (client sends screenshots)

```
Desktop Client                  Server                         Gemini
──────────────                  ──────                         ──────
WS connect /desktop/ws/{id}
    ──────────────────────────►
                                session lookup
◄───────────── WS accepted ─────

{"type":"task",                 create DesktopSession
 "intent":"...",                session.intent = intent
 "screenshot":"<b64>"}  ──────►
                                thinking → WS
                                ◄──── generate_content ────────►
                                parse WebPilotAction
◄──── {"type":"action",         
       "action":"click",        
       "x":640,"y":360,...} ────

execute pyautogui.click(640,360)
capture mss screenshot
{"type":"screenshot",
 "screenshot":"<b64>"}  ──────►
                                ◄──── generate_content ────────►
◄──── {"type":"action",...} ────
...
◄──── {"type":"done",...} ──────
WS close
```

### Data Flow — Server-Side Desktop Mode (POST /desktop/start)

```
API Client                      Server                         Desktop OS
──────────                      ──────                         ──────────
POST /desktop/start
{"intent":"...",
 "max_steps":15}  ────────────►
                                create DesktopSession
◄── {"session_id":"...",         mss.capture() → b64
     "status":"started"} ──────►
                                ──── Gemini call ────────────
                                parse action
                                pyautogui.click(x,y)       ──► OS executes
                                mss.capture()              ◄── new screenshot
                                loop...
GET /desktop/sessions/{id}
     ──────────────────────────►
◄── {"status":"done",
     "result":"..."}
```

### Key Design Decisions and Rationale

**Decision 1: Reuse WebPilotHandler verbatim, switch via system prompt**  
The handler is already parameterized — it accepts `viewport_width`, `viewport_height`, and `current_url`. Adding a `context_hint` parameter to `_build_user_content` (one line) makes it inject desktop-specific framing without duplicating 300 lines of handler logic. Alternative (a separate `DesktopHandler`) would duplicate the entire prompt-building, history management, and retry logic for no benefit.

**Decision 2: Two delivery paths — interactive WS and server-side REST**  
Interactive WS mirrors WebPilot exactly, enabling the Chrome extension sidebar to control the desktop with zero protocol changes (future: the extension could offer a "Desktop Mode" toggle). The server-side REST path enables non-interactive API use and is the demo-friendly path where the server autonomously executes tasks on the host machine.

**Decision 3: `mss` for capture, `pyautogui` for input**  
`mss` is the fastest Python screenshot library (no subprocess, direct OS framebuffer), produces PIL-compatible images, and is cross-platform. `pyautogui` is the de facto standard for Python desktop automation with a well-understood coordinate model. `pynput` was considered as an alternative for input but offers no advantage here and is less battle-tested for hackathon speed.

**Decision 4: Coordinate scaling as a first-class concern**  
On Windows with display scaling (e.g. 150 %), `mss` reports coordinates in logical pixels while `pyautogui` works in physical pixels. The `DesktopExecutor` MUST resolve and cache the scale factor at session start. This is non-obvious and a frequent failure mode in desktop automation.

**Decision 5: Server-side Desktop Mode gated behind env var**  
`DESKTOP_MODE_ENABLED=true` must be set. This prevents accidental activation on Cloud Run where the server has no display and `mss` would fail or capture a blank framebuffer. Security-conscious: an unguarded `/desktop/start` endpoint on a remote server could be used to exfiltrate screenshots of the host.

---

## Low-Level Design (LLD)

### New Files to Create

```
src/
  executor/
    base.py                    # AbstractExecutor protocol
    desktop.py                 # DesktopExecutor implementation
  api/
    desktop_models.py          # Pydantic models + DesktopSession dataclass
    desktop_routes.py          # FastAPI router for /desktop/*
  agent/
    desktop_system_prompt.py   # Separate module for the desktop system prompt constant
tests/
  test_desktop_executor.py     # Unit tests for DesktopExecutor
  test_desktop_api.py          # Integration tests for desktop REST + WS endpoints
```

### Existing Files to Modify

| File | Change |
|---|---|
| `src/executor/__init__.py` | Export `DesktopExecutor`, `AbstractExecutor` |
| `src/api/webpilot_models.py` | Add `right_click`, `double_click`, `move` to `WebPilotAction.action` Literal |
| `src/agent/webpilot_handler.py` | Accept `system_prompt_override: Optional[str]` in `__init__`; pass to all `generate_content` calls; add `get_next_action_desktop()` convenience wrapper |
| `src/api/server.py` | Lifespan: conditionally init desktop executor; include `desktop_routes.router`; add `desktop_sessions_active` to health check |
| `requirements.txt` | Add `mss>=9.0`, `pyautogui>=0.9.54` |
| `CLAUDE.md` | Update Architecture section, API Endpoints, Environment Variables table |

---

### Pydantic Models — Full Schema

**File: `src/api/desktop_models.py`**

```python
"""Pydantic models and session dataclass for Desktop Mode."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action model — extends WebPilotAction with desktop-specific actions
# ---------------------------------------------------------------------------

class DesktopAction(BaseModel):
    """
    Single-action response from Gemini for desktop control.

    Extends the WebPilot action schema with right_click, double_click, and move.
    The 'navigate' action type is intentionally excluded — it is browser-only.
    """
    model_config = ConfigDict(extra="ignore")

    observation: Optional[str] = None
    plan: Optional[List[str]] = None
    steps_completed: Optional[int] = None

    action: Literal[
        "click",
        "right_click",
        "double_click",
        "move",
        "type",
        "key",
        "scroll",
        "wait",
        "done",
        "confirm_required",
    ]

    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = Field(None, max_length=10000)
    direction: Optional[Literal["up", "down", "left", "right"]] = None
    duration: Optional[int] = Field(None, ge=0, le=10000)  # ms, wider range for desktop

    narration: str
    action_label: str
    is_irreversible: bool = False


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

@dataclass
class DesktopSession:
    """State for a single Desktop Mode agent session."""
    session_id: str
    intent: Optional[str] = None
    history: List = field(default_factory=list)
    status: str = "idle"          # idle | running | thinking | paused | done
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_active: float = field(default_factory=time.time)
    # Pixel coordinates of the screen at session creation time.
    screen_width: int = 1920
    screen_height: int = 1080
    # Scale factors: logical-to-physical pixel ratios (for HiDPI screens).
    scale_x: float = 1.0
    scale_y: float = 1.0


# ---------------------------------------------------------------------------
# REST request / response models
# ---------------------------------------------------------------------------

class DesktopStartRequest(BaseModel):
    """Request body for POST /desktop/start (server-side autonomous execution)."""
    intent: str = Field(..., min_length=1, max_length=2000,
                        description="High-level task for the AI agent")
    max_steps: int = Field(20, ge=1, le=50, description="Max agent loop steps")


class DesktopStartResponse(BaseModel):
    """Response from POST /desktop/start."""
    session_id: str
    status: str          # always "started"
    screen_width: int
    screen_height: int


class DesktopSessionStatusResponse(BaseModel):
    """Response from GET /desktop/sessions/{session_id}."""
    session_id: str
    status: str
    intent: Optional[str]
    steps_taken: int
    result: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# WS incoming message schemas (mirrors WebPilot protocol exactly)
# ---------------------------------------------------------------------------

class DesktopTaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["task"]
    intent: str
    screenshot: str          # base64 PNG


class DesktopScreenshotMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["screenshot"]
    screenshot: str          # base64 PNG
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None


class DesktopInterruptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["interrupt"]
    instruction: str
    screenshot: str          # base64 PNG


class DesktopConfirmMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["confirm"]
    confirmed: bool


class DesktopStopMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["stop"]
```

---

### API Endpoint Specifications

**File: `src/api/desktop_routes.py`**

#### POST /desktop/sessions

Create a Desktop session (WS path — client sends screenshots).

**Request:** `{}` (empty body or no body)

**Response 200:**
```json
{"session_id": "uuid"}
```

**Response 503:** `{"detail": "Desktop mode not enabled. Set DESKTOP_MODE_ENABLED=true."}` when env var absent.

---

#### DELETE /desktop/sessions/{session_id}

Delete a session.

**Response 200:** `{"status": "deleted"}`
**Response 404:** `{"detail": "Session not found"}`

---

#### POST /desktop/start

Start a server-side autonomous desktop task. The server captures its own screenshots.

**Request body:**
```json
{
  "intent": "Open VS Code and create a new file called hello.py",
  "max_steps": 20
}
```

**Response 202:**
```json
{
  "session_id": "uuid",
  "status": "started",
  "screen_width": 1920,
  "screen_height": 1080
}
```

**Response 503:** Desktop mode disabled or `mss` unavailable.

---

#### GET /desktop/sessions/{session_id}

Poll session status for the server-side execution path.

**Response 200:**
```json
{
  "session_id": "uuid",
  "status": "done",
  "intent": "Open VS Code...",
  "steps_taken": 7,
  "result": "Opened VS Code and created hello.py successfully.",
  "error": null
}
```

---

#### WS /desktop/ws/{session_id}

Interactive desktop WS loop. Mirrors `/webpilot/ws/{session_id}` protocol exactly.

**Incoming message types:**

| Type | Fields | Notes |
|---|---|---|
| `task` | `intent: str`, `screenshot: base64` | Start / restart task |
| `screenshot` | `screenshot: base64`, `screen_width?: int`, `screen_height?: int` | Next frame after action |
| `interrupt` | `instruction: str`, `screenshot: base64` | Mid-task user instruction |
| `confirm` | `confirmed: bool` | Response to `confirmation_required` |
| `stop` | — | Abort current task |

**Outgoing message types:**

| Type | Fields | Notes |
|---|---|---|
| `thinking` | `detail?: str` | Gemini call in progress |
| `action` | all `DesktopAction` fields | Execute this action |
| `confirmation_required` | `action: dict`, `narration: str` | Pause for user approval |
| `done` | all `DesktopAction` fields | Task complete |
| `stopped` | `narration?: str` | Task stopped |
| `paused` | `reason: str`, `narration: str`, `action_label: str` | Manual intervention needed |
| `error` | `message: str` | Non-fatal error |

---

### Desktop Executor — Full Implementation Design

**File: `src/executor/desktop.py`**

```python
"""
DesktopExecutor — captures OS screenshots with mss and dispatches
input events with pyautogui.

Thread safety: pyautogui is NOT thread-safe. All execute() calls MUST
be serialized — the FastAPI semaphore ensures at most one desktop session
runs at a time, making this safe without additional locking.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class DesktopExecutorError(Exception):
    """Raised when the desktop executor cannot be initialized."""


class DesktopExecutor:
    """
    Captures full-screen screenshots via mss and dispatches actions
    via pyautogui. Coordinates are expected in logical screen pixels.

    Lifecycle:
        executor = DesktopExecutor()
        await executor.start()   # detects screen size, warms mss
        ...
        await executor.stop()    # releases mss resources

    All public methods are async for interface compatibility with
    PlaywrightBrowserExecutor, even though the underlying calls are
    synchronous. Blocking calls run in a thread executor.
    """

    # pyautogui inter-action pause — 50ms prevents input floods
    _PYAUTOGUI_PAUSE = 0.05
    # Typing delay per character in seconds (human-like speed)
    _TYPE_INTERVAL = 0.02

    def __init__(self) -> None:
        self._mss = None            # mss.mss() instance
        self._screen_width: int = 0
        self._screen_height: int = 0
        # HiDPI scale factors: pyautogui coords / mss coords
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize mss, detect screen dimensions and DPI scale."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        await self._loop.run_in_executor(None, self._init_sync)
        self._started = True
        logger.info(
            "DesktopExecutor started: screen=%dx%d scale=%.2fx%.2f",
            self._screen_width, self._screen_height,
            self._scale_x, self._scale_y,
        )

    def _init_sync(self) -> None:
        """Synchronous initialization — runs in thread executor."""
        try:
            import mss
            import pyautogui
        except ImportError as exc:
            raise DesktopExecutorError(
                f"Desktop mode requires 'mss' and 'pyautogui': {exc}"
            ) from exc

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = self._PYAUTOGUI_PAUSE

        self._mss = mss.mss()
        monitor = self._mss.monitors[1]  # index 1 = primary monitor

        # mss returns logical resolution (what the OS reports)
        mss_w = monitor["width"]
        mss_h = monitor["height"]

        # pyautogui.size() returns screen resolution in pyautogui's
        # coordinate space — may differ on HiDPI displays.
        pyag_w, pyag_h = pyautogui.size()

        self._screen_width = mss_w
        self._screen_height = mss_h

        # Scale factor: convert screenshot pixel → pyautogui pixel
        self._scale_x = pyag_w / mss_w if mss_w > 0 else 1.0
        self._scale_y = pyag_h / mss_h if mss_h > 0 else 1.0

        logger.debug(
            "Screen: mss=%dx%d pyautogui=%dx%d scale=%.3fx%.3f",
            mss_w, mss_h, pyag_w, pyag_h, self._scale_x, self._scale_y,
        )

    async def stop(self) -> None:
        """Release mss resources."""
        if not self._started:
            return
        if self._mss:
            try:
                self._mss.close()
            except Exception as exc:
                logger.debug("mss close error (non-fatal): %s", exc)
            self._mss = None
        self._started = False
        logger.info("DesktopExecutor stopped")

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def screenshot(self) -> Image.Image:
        """Capture the primary monitor and return a PIL Image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._screenshot_sync)

    def _screenshot_sync(self) -> Image.Image:
        monitor = self._mss.monitors[1]
        sct = self._mss.grab(monitor)
        # mss returns BGRA; convert to RGB for PIL / Gemini
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return img

    async def screenshot_base64(self) -> str:
        """Capture and return base64-encoded PNG."""
        img = await self.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) in mss/screenshot coordinates."""
        return (self._screen_width, self._screen_height)

    # ------------------------------------------------------------------
    # Coordinate scaling
    # ------------------------------------------------------------------

    def _scale_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Scale coordinates from screenshot space (mss) to input space (pyautogui).

        On standard displays this is 1:1.
        On HiDPI/Retina displays this converts logical → physical pixels.
        Clamps to screen bounds after scaling.
        """
        sx = int(x * self._scale_x)
        sy = int(y * self._scale_y)
        pyag_w, pyag_h = int(self._screen_width * self._scale_x), int(self._screen_height * self._scale_y)
        sx = max(0, min(sx, pyag_w - 1))
        sy = max(0, min(sy, pyag_h - 1))
        return sx, sy

    # ------------------------------------------------------------------
    # Action helpers (all synchronous — run via thread executor)
    # ------------------------------------------------------------------

    def _click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        logger.debug("click (%d,%d) → scaled (%d,%d)", x, y, sx, sy)
        pyautogui.click(sx, sy)

    def _right_click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.rightClick(sx, sy)

    def _double_click_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.doubleClick(sx, sy)

    def _move_sync(self, x: int, y: int) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.moveTo(sx, sy, duration=0.1)

    def _type_sync(self, text: str) -> None:
        import pyautogui
        pyautogui.typewrite(text, interval=self._TYPE_INTERVAL)

    def _key_sync(self, key_combo: str) -> None:
        """
        Press a key or chord.

        Single key: "enter", "escape", "tab", "f5"
        Chord: "ctrl+c", "alt+f4", "ctrl+shift+t"
        Win key: "win" or "super"

        pyautogui.hotkey() accepts comma-separated key names.
        We split on '+' to handle chords.
        """
        import pyautogui
        # Normalize: lowercase, strip spaces
        normalized = key_combo.strip().lower()
        # Map common aliases
        aliases = {
            "win": "winleft",
            "super": "winleft",
            "cmd": "command",
            "return": "enter",
            "del": "delete",
            "esc": "escape",
        }
        parts = [aliases.get(k.strip(), k.strip()) for k in normalized.split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)

    def _scroll_sync(self, x: int, y: int, direction: str, amount: int = 3) -> None:
        import pyautogui
        sx, sy = self._scale_coords(x, y)
        pyautogui.moveTo(sx, sy)
        clicks = amount if direction in ("down", "right") else -amount
        if direction in ("up", "down"):
            pyautogui.scroll(clicks)
        else:
            pyautogui.hscroll(clicks)

    # ------------------------------------------------------------------
    # Main execute method
    # ------------------------------------------------------------------

    async def execute(self, action: "DesktopAction") -> "ActionResult":
        """
        Execute a single DesktopAction and return an ActionResult.

        Mirrors PlaywrightBrowserExecutor.execute() interface.
        All blocking pyautogui calls run in the thread executor.
        """
        from src.executor.actions import ActionResult
        loop = asyncio.get_running_loop()

        if not self._started:
            return ActionResult(
                success=False,
                error="DesktopExecutor not started",
                action_type=action.action,
            )

        act = action.action

        try:
            if act in ("click", "right_click", "double_click", "move"):
                if action.x is None or action.y is None:
                    return ActionResult(
                        success=False,
                        error=f"{act} requires x and y coordinates",
                        action_type=act,
                    )
                fn_map = {
                    "click": self._click_sync,
                    "right_click": self._right_click_sync,
                    "double_click": self._double_click_sync,
                    "move": self._move_sync,
                }
                await loop.run_in_executor(None, fn_map[act], action.x, action.y)
                # Take a screenshot after pointer actions so caller has fresh state
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=act,
                )

            elif act == "type":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="type action requires text",
                        action_type=act,
                    )
                await loop.run_in_executor(None, self._type_sync, action.text)
                return ActionResult(success=True, action_type=act)

            elif act == "key":
                if not action.text:
                    return ActionResult(
                        success=False,
                        error="key action requires text (key name / chord)",
                        action_type=act,
                    )
                await loop.run_in_executor(None, self._key_sync, action.text)
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=act,
                )

            elif act == "scroll":
                x = action.x or (self._screen_width // 2)
                y = action.y or (self._screen_height // 2)
                direction = action.direction or "down"
                amount = 3  # fixed; DesktopAction doesn't have scroll_amount
                await loop.run_in_executor(None, self._scroll_sync, x, y, direction, amount)
                return ActionResult(success=True, action_type=act)

            elif act == "wait":
                duration_s = (action.duration or 1000) / 1000.0
                await asyncio.sleep(duration_s)
                return ActionResult(success=True, action_type=act)

            elif act in ("done", "confirm_required"):
                # No physical action — signal handled by the caller
                return ActionResult(success=True, action_type=act)

            else:
                return ActionResult(
                    success=False,
                    error=f"Unknown desktop action type: {act!r}",
                    action_type=act,
                )

        except Exception as exc:
            logger.exception("DesktopExecutor error executing %s: %s", act, exc)
            return ActionResult(success=False, error=str(exc), action_type=act)
```

---

### Abstract Base Executor

**File: `src/executor/base.py`**

```python
"""AbstractExecutor protocol — shared interface for Browser and Desktop executors."""
from __future__ import annotations

from typing import Protocol, Tuple
from PIL import Image
from src.executor.actions import ActionResult


class AbstractExecutor(Protocol):
    """
    Structural protocol that both PlaywrightBrowserExecutor and
    DesktopExecutor satisfy. Allows type-checking code that accepts either.
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def screenshot(self) -> Image.Image: ...
    async def screenshot_base64(self) -> str: ...
    async def execute(self, action) -> ActionResult: ...
```

---

### Desktop System Prompt

**File: `src/agent/desktop_system_prompt.py`**

The desktop system prompt is structurally identical to `WEBPILOT_SYSTEM_PROMPT` with these specific changes:

1. All references to "browser", "tab", "URL bar", "navigate" are removed or replaced
2. The VALID ACTION TYPES section replaces `navigate` with `right_click`, `double_click`, `move`
3. The COORDINATE RULES section states coordinates are screen-relative from top-left of the primary display
4. The context is "desktop OS application" not "web browser"

```python
DESKTOP_SYSTEM_PROMPT = """\
You are DesktopPilot, an AI agent that controls desktop applications by analyzing screenshots.

Your task: Given a screenshot of the current desktop state and a user's intent, determine
the SINGLE NEXT ACTION to take. Return ONE action at a time as a JSON object.

RESPONSE FORMAT (always return exactly this JSON structure, no markdown fences, no prose):
{
  "observation": "<one sentence: describe exactly what you currently see on screen>",
  "plan": ["<step 1>", "<step 2>", ...],
  "steps_completed": <integer: how many plan steps are done so far>,
  "action": "<action_type>",
  "x": <integer pixel x coordinate, or null if not applicable>,
  "y": <integer pixel y coordinate, or null if not applicable>,
  "text": "<text to type or key name, or null if not applicable>",
  "direction": "<up, down, left, or right, or null if not applicable>",
  "duration": <milliseconds to wait, or null if not applicable>,
  "narration": "<short human-readable description of what you are doing and why>",
  "action_label": "<very short action label, e.g. 'Click Save', 'Type filename'>",
  "is_irreversible": <true if action cannot be undone, false otherwise>
}

LOOK BEFORE YOU ACT:
- You MUST fill in "observation" first, describing exactly what is on screen RIGHT NOW.
- Only after observing should you decide x/y coordinates.
- Never guess coordinates — only click elements you can clearly see in the screenshot.

VALID ACTION TYPES:
- "click": Left-click at coordinates (x, y). Required: x, y.
- "right_click": Right-click at coordinates (x, y). Required: x, y.
- "double_click": Double-click at coordinates (x, y). Required: x, y.
- "move": Move the mouse cursor to (x, y) without clicking. Required: x, y.
- "type": Type text using the keyboard. Required: text.
- "scroll": Scroll at coordinates. Required: direction ("up", "down", "left", "right").
  Optional: x, y (defaults to screen center).
- "wait": Wait for a specified duration. Required: duration (milliseconds).
- "key": Press a key or keyboard shortcut. Required: text (key name or chord).
  Examples: "enter", "escape", "tab", "ctrl+c", "alt+f4", "win", "ctrl+shift+t".
- "done": The task is fully complete.
- "confirm_required": The next logical action is irreversible (e.g., deleting files,
  sending messages, submitting forms). Pause and ask the user to confirm.

COORDINATE RULES:
- Coordinates are pixel positions from the top-left corner of the primary screen.
- Exact screen dimensions are provided in each user message — stay within those bounds.
- Be precise — click on the center of the target UI element.
- Only use coordinates of elements you can clearly see in the screenshot.

IRREVERSIBILITY RULES:
- Set is_irreversible=true for: deleting files, sending messages, submitting forms,
  emptying trash, uninstalling software, closing unsaved work.
- When is_irreversible=true, also set action="confirm_required".

PLANNING (Phase 1 — first call only):
- Before taking any action, generate a "plan" field: a list of concrete steps.
- The plan defines your completion criteria. You are done ONLY when all steps are complete.

PROGRESS CHECK (Phase 2 — every call):
- Set "steps_completed" to the number of plan steps fully finished so far.
- Carry forward the same "plan" array from your first response on every subsequent response.

CRITICAL: Only return action="done" when steps_completed equals the TOTAL number of steps.

IMPORTANT:
- Respond ONLY with the JSON object — no markdown fences, no extra prose.
- You are controlling a DESKTOP APPLICATION, not a web browser. There is no URL bar.
- If a dialog box appears, address it before continuing the main task.
- If the screen is locked, use action="done" with narration explaining you cannot proceed.
"""
```

---

### Modified: `src/agent/webpilot_handler.py`

Add a constructor parameter and a `get_next_action_desktop()` method:

```python
# In __init__, add:
def __init__(
    self,
    vision_client: GeminiVisionClient,
    planner,
    system_prompt_override: Optional[str] = None,   # NEW
) -> None:
    self._client = vision_client._client
    self._planner = planner
    self._system_prompt = system_prompt_override or WEBPILOT_SYSTEM_PROMPT  # NEW

# In get_next_action(), replace hardcoded WEBPILOT_SYSTEM_PROMPT with self._system_prompt
# (one line change in the generate_content config block)

# Add new public method:
async def get_next_action_desktop(
    self,
    image_b64: str,
    intent: str,
    history: list,
    stuck: bool = False,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> "DesktopAction":
    """
    Wrapper around get_next_action() for Desktop Mode.

    Returns a DesktopAction (superset of WebPilotAction) parsed from
    the Gemini response. Uses the desktop system prompt.
    """
    from src.api.desktop_models import DesktopAction
    # get_next_action calls self._system_prompt which is already set to
    # DESKTOP_SYSTEM_PROMPT when this handler was constructed for desktop.
    wp_action = await self.get_next_action(
        image_b64=image_b64,
        intent=intent,
        history=history,
        stuck=stuck,
        viewport_width=screen_width,
        viewport_height=screen_height,
        current_url="",
    )
    # Reparse raw action into DesktopAction (which has right_click, double_click, move)
    data = wp_action.model_dump()
    try:
        return DesktopAction(**data)
    except Exception:
        # Fallback: if action type not in DesktopAction, treat as done
        return DesktopAction(
            action="done",
            narration="Action type not supported on desktop",
            action_label="Done",
        )
```

**Important:** Because `get_next_action()` uses `self._system_prompt`, the desktop handler instance simply needs to be constructed with `system_prompt_override=DESKTOP_SYSTEM_PROMPT`. No duplication of the Gemini call logic.

---

### Desktop Routes — Full Implementation Design

**File: `src/api/desktop_routes.py`**

```python
"""Desktop Mode API router — WS-driven and REST-triggered desktop control."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.api.desktop_models import (
    DesktopAction,
    DesktopConfirmMessage,
    DesktopInterruptMessage,
    DesktopScreenshotMessage,
    DesktopSession,
    DesktopStartRequest,
    DesktopStartResponse,
    DesktopSessionStatusResponse,
    DesktopTaskMessage,
)

_DESKTOP_MODE_ENABLED = os.environ.get("DESKTOP_MODE_ENABLED", "").lower() in ("1", "true", "yes")
_MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
_ACTION_LOOP_TIMEOUT = int(os.environ.get("ACTION_LOOP_TIMEOUT", "120"))
_MAX_LOOP_STEPS = int(os.environ.get("MAX_LOOP_STEPS", "30"))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/desktop", tags=["desktop"])

# Session store: session_id → DesktopSession
_sessions: Dict[str, DesktopSession] = {}

# Shared handler — same WebPilotHandler instance configured with desktop system prompt
_desktop_handler = None
# Desktop executor (server-side mode only)
_desktop_executor = None


def init_desktop_handler(handler, executor=None) -> None:
    """Called from server.py lifespan."""
    global _desktop_handler, _desktop_executor
    _desktop_handler = handler
    _desktop_executor = executor


async def cleanup_desktop_sessions() -> None:
    """Remove sessions inactive for >30 minutes."""
    _MAX_SESSION_DURATION = int(os.environ.get("MAX_SESSION_DURATION", "1800"))
    while True:
        await asyncio.sleep(300)
        cutoff = time.time() - _MAX_SESSION_DURATION
        stale = [sid for sid, s in list(_sessions.items()) if s.last_active < cutoff]
        for sid in stale:
            _sessions.pop(sid)
            logger.info("Cleaned up stale desktop session %s", sid)


def _check_desktop_enabled():
    """Raise 503 if DESKTOP_MODE_ENABLED is not set."""
    if not _DESKTOP_MODE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Desktop mode is not enabled. "
                "Set DESKTOP_MODE_ENABLED=true on the host running the server. "
                "Desktop mode cannot run on Cloud Run or headless containers."
            ),
        )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions")
async def create_desktop_session() -> dict:
    """Create a desktop session for the interactive WS path."""
    _check_desktop_enabled()
    if len(_sessions) >= 1000:
        raise HTTPException(status_code=503, detail="Maximum session limit reached")
    session_id = str(uuid.uuid4())
    _sessions[session_id] = DesktopSession(session_id=session_id)
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_desktop_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "deleted"}


@router.get("/sessions/{session_id}", response_model=DesktopSessionStatusResponse)
async def get_desktop_session(session_id: str) -> DesktopSessionStatusResponse:
    """Poll status of a server-side desktop task."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    return DesktopSessionStatusResponse(
        session_id=session_id,
        status=session.status,
        intent=session.intent,
        steps_taken=getattr(session, "_steps_taken", 0),
        result=getattr(session, "_result", None),
        error=getattr(session, "_error", None),
    )


@router.post("/start", response_model=DesktopStartResponse, status_code=202)
async def start_desktop_task(body: DesktopStartRequest) -> DesktopStartResponse:
    """
    Start a server-side autonomous desktop task.

    The server captures its own screenshots and executes actions directly
    against the host OS. The client polls GET /desktop/sessions/{id} for status.
    """
    _check_desktop_enabled()
    if _desktop_handler is None or _desktop_executor is None:
        raise HTTPException(status_code=503, detail="Desktop handler not initialized")

    session_id = str(uuid.uuid4())
    w, h = _desktop_executor.screen_size
    session = DesktopSession(
        session_id=session_id,
        intent=body.intent,
        screen_width=w,
        screen_height=h,
        scale_x=_desktop_executor._scale_x,
        scale_y=_desktop_executor._scale_y,
    )
    _sessions[session_id] = session

    # Run the autonomous loop in the background
    asyncio.create_task(
        _run_autonomous_loop(session, body.max_steps)
    )

    return DesktopStartResponse(
        session_id=session_id,
        status="started",
        screen_width=w,
        screen_height=h,
    )


# ---------------------------------------------------------------------------
# Server-side autonomous loop
# ---------------------------------------------------------------------------

async def _run_autonomous_loop(session: DesktopSession, max_steps: int) -> None:
    """
    Server drives the entire loop: capture → Gemini → execute → repeat.
    No WebSocket involved — results stored on the session object.
    """
    session.status = "running"
    session._steps_taken = 0
    session._result = None
    session._error = None

    try:
        await asyncio.wait_for(
            _autonomous_loop_inner(session, max_steps),
            timeout=_ACTION_LOOP_TIMEOUT,
        )
    except asyncio.TimeoutError:
        session.status = "error"
        session._error = f"Task timed out after {_ACTION_LOOP_TIMEOUT}s"
        logger.warning("Autonomous desktop loop timed out for session %s", session.session_id)
    except Exception as exc:
        session.status = "error"
        session._error = str(exc)
        logger.exception("Autonomous desktop loop error for session %s", session.session_id)


async def _autonomous_loop_inner(session: DesktopSession, max_steps: int) -> None:
    history = []
    prev_hash = b""
    retry_count = 0

    for step in range(1, max_steps + 1):
        session._steps_taken = step
        if session.abort_event.is_set():
            session.status = "idle"
            return

        screenshot_b64 = await _desktop_executor.screenshot_base64()

        # Stuck detection
        new_hash = hashlib.md5(base64.b64decode(screenshot_b64)).digest()
        if new_hash == prev_hash:
            retry_count += 1
        else:
            retry_count = 0
        prev_hash = new_hash
        stuck = retry_count >= _MAX_RETRIES
        if stuck:
            retry_count = 0
            prev_hash = b""

        try:
            action = await _desktop_handler.get_next_action_desktop(
                image_b64=screenshot_b64,
                intent=session.intent,
                history=history,
                stuck=stuck,
                screen_width=session.screen_width,
                screen_height=session.screen_height,
            )
        except Exception as exc:
            session._error = str(exc)
            session.status = "error"
            return

        if action.action == "done":
            session._result = action.narration
            session.status = "done"
            return

        if action.action == "confirm_required" or action.is_irreversible:
            # Autonomous mode: auto-deny irreversible actions — safety first
            logger.warning(
                "Autonomous mode blocked irreversible action at step %d: %s",
                step, action.action_label,
            )
            session._result = f"Stopped: action requires confirmation ({action.action_label})"
            session.status = "done"
            return

        result = await _desktop_executor.execute(action)
        if not result.success:
            logger.warning(
                "Desktop action failed at step %d: %s — %s",
                step, action.action, result.error,
            )
        # Brief yield between steps
        await asyncio.sleep(0.1)

    session._result = f"Reached max steps ({max_steps}) without completing task."
    session.status = "done"


# ---------------------------------------------------------------------------
# Interactive WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def desktop_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Interactive Desktop Mode WS endpoint.

    Protocol is identical to /webpilot/ws/{session_id}.
    The client (desktop app or script) executes actions and sends back screenshots.
    """
    if session_id not in _sessions:
        await websocket.accept()
        await websocket.close(code=4404, reason="Session not found")
        return

    if _desktop_handler is None:
        await websocket.accept()
        await websocket.close(code=4503, reason="Desktop handler not initialized")
        return

    session = _sessions[session_id]
    await websocket.accept()
    logger.info("Desktop WS connected: session=%s", session_id)

    had_error = False
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 15 * 1024 * 1024:
                await websocket.send_json({"type": "error", "message": "Message too large"})
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")
            session.last_active = time.time()

            try:
                if msg_type == "task":
                    msg = DesktopTaskMessage(**data)
                    session.intent = msg.intent
                    session.history = []
                    session.status = "running"
                    session.abort_event.clear()
                    await _run_ws_action_loop(websocket, session, msg.screenshot)

                elif msg_type == "interrupt":
                    if session.status not in ("running", "thinking", "done"):
                        await websocket.send_json({"type": "stopped"})
                        continue
                    msg = DesktopInterruptMessage(**data)
                    session.status = "running"
                    await _handle_desktop_interrupt(websocket, session, msg.screenshot, msg.instruction)

                elif msg_type == "stop":
                    session.abort_event.set()
                    session.status = "idle"
                    await websocket.send_json({"type": "stopped"})

            except (ValidationError, KeyError) as exc:
                await websocket.send_json({"type": "error", "message": f"Invalid message: {exc}"})

    except WebSocketDisconnect:
        logger.info("Desktop WS disconnected: session=%s", session_id)
    except Exception:
        had_error = True
        logger.exception("Desktop WS error: session=%s", session_id)
    finally:
        if had_error:
            try:
                await websocket.send_json({"type": "error", "message": "Internal server error"})
            except Exception:
                pass


async def _run_ws_action_loop(
    websocket: WebSocket,
    session: DesktopSession,
    first_screenshot: str,
    steps_remaining: Optional[int] = None,
) -> None:
    """Wrap inner loop with hard timeout."""
    try:
        await asyncio.wait_for(
            _run_ws_action_loop_inner(websocket, session, first_screenshot, steps_remaining),
            timeout=_ACTION_LOOP_TIMEOUT,
        )
    except asyncio.TimeoutError:
        session.status = "idle"
        await websocket.send_json(
            {"type": "stopped", "narration": f"Task timed out after {_ACTION_LOOP_TIMEOUT}s."}
        )


async def _run_ws_action_loop_inner(
    websocket: WebSocket,
    session: DesktopSession,
    first_screenshot: str,
    steps_remaining: Optional[int] = None,
) -> None:
    """
    Core interactive WS action loop for Desktop Mode.

    Structurally identical to webpilot_routes._run_action_loop_inner().
    """
    if steps_remaining is None:
        steps_remaining = _MAX_LOOP_STEPS

    screenshot = first_screenshot
    _prev_hash = hashlib.md5(base64.b64decode(first_screenshot)).digest()
    retry_count = 0
    step_count = 0

    while not session.abort_event.is_set():
        if step_count >= steps_remaining:
            session.status = "idle"
            await websocket.send_json(
                {"type": "stopped", "narration": "Reached maximum number of steps."}
            )
            return
        step_count += 1

        await websocket.send_json({"type": "thinking"})

        stuck = retry_count >= _MAX_RETRIES
        if stuck:
            retry_count = 0
            _prev_hash = b""

        try:
            action = await _desktop_handler.get_next_action_desktop(
                image_b64=screenshot,
                intent=session.intent,
                history=session.history,
                stuck=stuck,
                screen_width=session.screen_width,
                screen_height=session.screen_height,
            )
        except Exception as exc:
            logger.exception("Desktop handler error: session=%s", session.session_id)
            await websocket.send_json({"type": "error", "message": str(exc)})
            session.status = "idle"
            return

        action_dict = action.model_dump()

        if action.action == "done":
            await websocket.send_json({"type": "done", **action_dict})
            session.status = "done"
            return

        if action.is_irreversible or action.action == "confirm_required":
            session.status = "awaiting_confirm"
            await websocket.send_json(
                {"type": "confirmation_required", "action": action_dict, "narration": action.narration}
            )
            raw_confirm = await websocket.receive_text()
            try:
                confirm_data = json.loads(raw_confirm)
                confirmed = DesktopConfirmMessage(**confirm_data).confirmed
            except (json.JSONDecodeError, ValidationError):
                confirmed = False
            if not confirmed:
                await websocket.send_json({"type": "stopped", "narration": "Action cancelled."})
                session.status = "idle"
                return
            session.status = "running"

        await websocket.send_json({"type": "action", **action_dict})

        # Trim history
        if len(session.history) >= 20:
            session.history = session.history[-18:]

        # Wait for next screenshot from client
        raw = await websocket.receive_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            session.status = "idle"
            return

        msg_type = data.get("type")
        if msg_type == "stop":
            session.abort_event.set()
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
            return
        elif msg_type == "interrupt":
            msg = DesktopInterruptMessage(**data)
            await _handle_desktop_interrupt(websocket, session, msg.screenshot, msg.instruction)
            return
        elif msg_type == "screenshot":
            screenshot = data["screenshot"]
            # Update screen dimensions if client reports them
            if data.get("screen_width"):
                session.screen_width = data["screen_width"]
            if data.get("screen_height"):
                session.screen_height = data["screen_height"]
            new_hash = hashlib.md5(base64.b64decode(screenshot)).digest()
            if new_hash == _prev_hash:
                retry_count += 1
            else:
                retry_count = 0
            _prev_hash = new_hash
        else:
            await websocket.send_json(
                {"type": "error", "message": f"Unexpected message type: {msg_type}"}
            )
            session.status = "idle"
            return


async def _handle_desktop_interrupt(
    websocket: WebSocket,
    session: DesktopSession,
    screenshot: str,
    instruction: str,
) -> None:
    """Handle interruption — reuses WebPilotHandler.classify_interruption_type()."""
    from src.agent.webpilot_handler import WebPilotHandler
    from src.api.webpilot_models import InterruptionType

    interrupt_type = WebPilotHandler.classify_interruption_type(instruction)

    if interrupt_type == InterruptionType.ABORT:
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json(
            {"type": "stopped", "narration": "Stopped. What would you like to do?"}
        )
        return

    original_intent = session.intent
    if interrupt_type == InterruptionType.REDIRECT:
        session.history = []
        session.intent = instruction
    else:
        session.intent = f"{session.intent} ({instruction})"

    await websocket.send_json({"type": "thinking"})

    try:
        action = await _desktop_handler.get_interruption_replan(
            screenshot, original_intent, instruction, session.history, interrupt_type,
            viewport_width=session.screen_width,
            viewport_height=session.screen_height,
        )
    except Exception as exc:
        logger.exception("Desktop interrupt replan error: session=%s", session.session_id)
        await websocket.send_json({"type": "error", "message": str(exc)})
        session.status = "idle"
        return

    action_dict = action.model_dump()
    if action.action == "done":
        await websocket.send_json({"type": "done", **action_dict})
        session.status = "done"
        return

    await websocket.send_json({"type": "action", **action_dict})

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Timeout waiting for screenshot after interrupt"})
        session.status = "idle"
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "Invalid JSON after interrupt"})
        session.status = "idle"
        return

    if data.get("type") == "screenshot":
        await _run_ws_action_loop(
            websocket, session, data["screenshot"],
            steps_remaining=max(1, _MAX_LOOP_STEPS // 2),
        )
        if session.status not in ("done", "idle"):
            session.status = "idle"
            await websocket.send_json({"type": "stopped"})
    elif data.get("type") == "stop":
        session.abort_event.set()
        session.status = "idle"
        await websocket.send_json({"type": "stopped"})
```

---

### server.py Modifications

In `lifespan()`, add after the WebPilot handler block:

```python
# Desktop Mode initialization
if os.environ.get("DESKTOP_MODE_ENABLED", "").lower() in ("1", "true", "yes"):
    try:
        from src.executor.desktop import DesktopExecutor
        from src.agent.desktop_system_prompt import DESKTOP_SYSTEM_PROMPT
        from src.api.desktop_routes import init_desktop_handler

        _desktop_executor = DesktopExecutor()
        await _desktop_executor.start()

        # Construct a second WebPilotHandler configured with the desktop prompt.
        # Reuses the same GeminiVisionClient and ActionPlanner instances.
        _desktop_wp_handler = WebPilotHandler(
            vision_client=_wp_vision,
            planner=_wp_planner,
            system_prompt_override=DESKTOP_SYSTEM_PROMPT,
        )
        init_desktop_handler(_desktop_wp_handler, _desktop_executor)
        logger.info("Desktop Mode enabled — DesktopExecutor started")
    except Exception as exc:
        logger.error("Failed to initialize Desktop Mode: %s", exc)
        # Non-fatal: server continues without desktop mode

desktop_cleanup_task = asyncio.create_task(cleanup_desktop_sessions())
```

In lifespan shutdown:
```python
# Cleanup desktop executor
if _desktop_executor is not None:
    await _desktop_executor.stop()
desktop_cleanup_task.cancel()
```

Add router include:
```python
from src.api import desktop_routes
app.include_router(desktop_routes.router)
```

In `health_check()` response:
```python
from src.api.desktop_routes import _sessions as _desktop_sessions
return {
    ...existing fields...,
    "desktop_sessions_active": sum(
        1 for s in _desktop_sessions.values() if s.status == "running"
    ),
}
```

---

### Modified: `src/api/webpilot_models.py`

Add new action types to `WebPilotAction.action` Literal:

```python
# Old:
action: Literal["click", "type", "scroll", "wait", "navigate", "key", "done",
                "confirm_required", "captcha_detected", "login_required"]

# New — add right_click, double_click, move for desktop compatibility:
action: Literal["click", "right_click", "double_click", "move", "type", "scroll",
                "wait", "navigate", "key", "done",
                "confirm_required", "captcha_detected", "login_required"]
```

This change is backward-compatible — existing WebPilot paths never receive the new types from Gemini (the WebPilot system prompt doesn't list them), but the model can accept them without crashing if they ever appear.

---

### Modified: `requirements.txt`

Add:
```
mss>=9.0
pyautogui>=0.9.54
```

Note: On Linux, `pyautogui` requires `python3-tk` and `python3-dev` and the `scrot` or `gnome-screenshot` package for screenshot fallback. Add a comment to requirements.txt noting this.

On macOS 10.15+, pyautogui requires screen recording permission granted to the Python process via System Preferences → Security & Privacy → Screen Recording.

---

### Coordinate System Handling

This is the most failure-prone aspect. Here is the complete mapping:

```
SCENARIO 1: Standard 1080p monitor, no DPI scaling
  mss monitor dimensions:  1920 x 1080
  pyautogui.size():         1920 x 1080
  scale_x = 1.0, scale_y = 1.0
  Screenshot pixel (640, 360) → pyautogui.click(640, 360)  ✓

SCENARIO 2: Windows 4K monitor at 150% scaling
  mss monitor dimensions:  3840 x 2160  (physical pixels)
  pyautogui.size():         2560 x 1440  (logical pixels at 150%)
  scale_x = 2560/3840 = 0.667
  scale_y = 1440/2160 = 0.667
  Screenshot pixel (1920, 1080) → pyautogui.click(1280, 720)  ✓

SCENARIO 3: macOS Retina (2x)
  mss monitor dimensions:  2560 x 1600  (logical * 2)
  pyautogui.size():         1280 x 800   (logical)
  scale_x = 1280/2560 = 0.5
  scale_y = 800/1600  = 0.5
  Screenshot pixel (1280, 800) → pyautogui.click(640, 400)  ✓

SCENARIO 4: Windows monitor at 100% scaling
  mss monitor dimensions:  1920 x 1080
  pyautogui.size():         1920 x 1080
  scale_x = 1.0
  Straightforward.
```

The `_scale_coords()` method in `DesktopExecutor` handles all four cases with the same formula: `scaled = int(coord * scale_factor)`.

**Gemini coordinate accuracy note:** Gemini returns coordinates as pixel positions within the screenshot. Because we send the screenshot at its native mss resolution, the coordinate mapping is direct. We do NOT resize screenshots before sending to Gemini (unlike some automation frameworks that downscale to 1280x800). Downscaling would introduce coordinate errors. This is an explicit design choice — send full-resolution screenshots to Gemini, accept slightly higher token cost per image.

---

### Error Handling Strategy

| Failure | Handler | Recovery |
|---|---|---|
| `mss` import fails on Cloud Run | `_check_desktop_enabled()` returns 503 before executor starts | Graceful 503 |
| `pyautogui.FailSafeException` (mouse corner) | Caught in `execute()` → `ActionResult(success=False)` | WS error message sent |
| Coordinate out of bounds | `_scale_coords()` clamps to `[0, screen_size-1]` | Silent clamp |
| `pyautogui.typewrite()` non-ASCII character | `try/except` in `_type_sync` → fallback to `pyperclip` paste | Log warning, continue |
| Gemini returns `navigate` action in desktop mode | `DesktopAction` Literal validation fails → `_parse_action` raises → handler falls back to `done` | Session continues |
| Screen goes to sleep / locks | Next screenshot is dark/lock screen → Gemini observes it → returns narration about screen lock | Session ends gracefully |
| mss capture error mid-session | `screenshot_base64()` raises → caught in loop → `ActionResult(success=False)` | Error event on WS |

---

### Platform Differences

| Platform | Known Issue | Mitigation |
|---|---|---|
| Windows 10/11 | `pyautogui.typewrite()` doesn't handle Unicode (emoji, CJK) | Use `pyperclip.copy(text); pyautogui.hotkey('ctrl', 'v')` for non-ASCII |
| macOS 13+ | Screen Recording permission required for mss | `_init_sync()` logs actionable warning if `mss.grab()` raises `AttributeError` |
| macOS (Apple Silicon) | pyautogui works via Rosetta — minor coordinate offset on some apps | No mitigation needed for hackathon |
| Linux (X11) | Requires `DISPLAY` env var; `mss` fails on Wayland | Set `DISPLAY=:0`; Wayland unsupported |
| All | `pyautogui.typewrite()` uses 20ms delay — too slow for long text | Increase interval or use clipboard paste |
| Windows | Some UAC dialogs run at SYSTEM level — pyautogui can't click them | Document as known limitation |

---

## Interface Contracts

### AbstractExecutor Protocol

```python
# src/executor/base.py
from typing import Protocol, Tuple
from PIL import Image
from src.executor.actions import ActionResult

class AbstractExecutor(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def screenshot(self) -> Image.Image: ...
    async def screenshot_base64(self) -> str: ...
    async def execute(self, action: object) -> ActionResult: ...
```

### DesktopExecutor Public Interface

```python
class DesktopExecutor:
    # Lifecycle
    async def start(self) -> None
    async def stop(self) -> None

    # Capture
    async def screenshot(self) -> Image.Image
    async def screenshot_base64(self) -> str
    @property
    def screen_size(self) -> Tuple[int, int]

    # Execute
    async def execute(self, action: DesktopAction) -> ActionResult

    # Internal (not called externally but used in tests)
    def _scale_coords(self, x: int, y: int) -> Tuple[int, int]
    def _init_sync(self) -> None          # called in thread executor by start()
    def _screenshot_sync(self) -> Image.Image
    def _click_sync(self, x: int, y: int) -> None
    def _right_click_sync(self, x: int, y: int) -> None
    def _double_click_sync(self, x: int, y: int) -> None
    def _move_sync(self, x: int, y: int) -> None
    def _type_sync(self, text: str) -> None
    def _key_sync(self, key_combo: str) -> None
    def _scroll_sync(self, x: int, y: int, direction: str, amount: int) -> None
```

### WebPilotHandler Updated Interface

```python
class WebPilotHandler:
    def __init__(
        self,
        vision_client: GeminiVisionClient,
        planner: ActionPlanner,
        system_prompt_override: Optional[str] = None,   # NEW
    ) -> None

    # Existing methods unchanged in signature
    async def get_next_action(
        self,
        image_b64: str,
        intent: str,
        history: list,
        stuck: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        current_url: str = "",
    ) -> WebPilotAction

    # NEW — desktop wrapper
    async def get_next_action_desktop(
        self,
        image_b64: str,
        intent: str,
        history: list,
        stuck: bool = False,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> DesktopAction

    # Unchanged
    async def get_interruption_replan(...) -> WebPilotAction
    async def verify_completion(...) -> bool
    async def get_narration_audio(text: str) -> bytes
    @staticmethod
    def classify_interruption_type(instruction: str) -> InterruptionType
```

### DesktopSession dataclass

```python
@dataclass
class DesktopSession:
    session_id: str
    intent: Optional[str] = None
    history: List = field(default_factory=list)
    status: str = "idle"
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_active: float = field(default_factory=time.time)
    screen_width: int = 1920
    screen_height: int = 1080
    scale_x: float = 1.0
    scale_y: float = 1.0
    # Runtime state set by loops (not in constructor):
    # _steps_taken: int
    # _result: Optional[str]
    # _error: Optional[str]
```

---

## Implementation Plan

### Phase 1: Foundation (no Gemini dependency, all unit-testable)

**Task 1.1 — Create `src/executor/base.py`**  
Create the `AbstractExecutor` protocol. No dependencies. 15 min.

**Task 1.2 — Create `src/executor/desktop.py` (stub)**  
Implement `DesktopExecutor` with a `MockBackend` path: if `DESKTOP_MOCK=true`, all action methods no-op and `screenshot_base64()` returns a 1920x1080 blank PNG. This enables testing without a real display. 60 min.

**Task 1.3 — Create `src/agent/desktop_system_prompt.py`**  
Write the desktop system prompt constant. Copy-edit from `WEBPILOT_SYSTEM_PROMPT`, replace browser-specific language. 30 min.

**Task 1.4 — Create `src/api/desktop_models.py`**  
Implement all Pydantic models and `DesktopSession` dataclass per the LLD spec. 30 min.

**Validation checkpoint:** Run `python -c "from src.executor.desktop import DesktopExecutor; from src.api.desktop_models import DesktopAction"` — no import errors.

---

### Phase 2: Handler extension

**Task 2.1 — Modify `src/agent/webpilot_handler.py`**  
Add `system_prompt_override` parameter to `__init__`. Replace the hardcoded `WEBPILOT_SYSTEM_PROMPT` references in `generate_content` config with `self._system_prompt`. Add `get_next_action_desktop()` method. One test in `tests/test_webpilot_api.py` to verify the override works. 45 min.

**Task 2.2 — Update `src/api/webpilot_models.py`**  
Add `right_click`, `double_click`, `move` to `WebPilotAction.action` Literal. Verify existing tests still pass. 10 min.

---

### Phase 3: Routes and server wiring

**Task 3.1 — Create `src/api/desktop_routes.py`**  
Implement the full router per the LLD. 90 min.

**Task 3.2 — Modify `src/api/server.py`**  
Add lifespan block for desktop executor init, router include, and health check field per the LLD spec. 30 min.

**Task 3.3 — Update `requirements.txt`**  
Add `mss>=9.0` and `pyautogui>=0.9.54`. Add platform notes as comments. 5 min.

---

### Phase 4: Tests

**Task 4.1 — `tests/test_desktop_executor.py`**  
Test all of the following with `DESKTOP_MOCK=true`:
- `start()` / `stop()` lifecycle
- `_scale_coords()` with various scale factors (1.0, 0.5, 0.667, 2.0)
- `execute()` for each action type returns `ActionResult(success=True)`
- `execute()` with missing coordinates returns `ActionResult(success=False)`
- `screenshot_base64()` returns valid base64 PNG
- Platform key alias mapping ("win" → "winleft", "return" → "enter")
Estimated: 15 tests, 60 min.

**Task 4.2 — `tests/test_desktop_api.py`**  
Test with a real FastAPI TestClient and `DESKTOP_MOCK=true`:
- `POST /desktop/sessions` → 503 without `DESKTOP_MODE_ENABLED=true`
- `POST /desktop/sessions` → 200 with env var set
- `DELETE /desktop/sessions/{id}` — 200 and 404 cases
- `GET /desktop/sessions/{id}` — status polling
- `POST /desktop/start` → 202, session created
- `WS /desktop/ws/{id}` — full task → screenshot → action → done flow
- `WS /desktop/ws/{id}` — stop message handling
- `WS /desktop/ws/{id}` — interrupt message handling (ABORT, REDIRECT, REFINEMENT)
- `WS /desktop/ws/{id}` — confirm_required flow (approve and deny)
- `WS /desktop/ws/{id}` — step budget exhaustion
- `WS /desktop/ws/{id}` — invalid session_id returns 4404
Estimated: 20 tests, 90 min.

**Task 4.3 — CLAUDE.md update**  
Update Architecture, API Endpoints, Environment Variables, and Testing Notes sections. 20 min.

---

### Phase 5: Integration and demo

**Task 5.1 — Manual smoke test: server-side autonomous mode**  
Set `DESKTOP_MODE_ENABLED=true`, start server, POST to `/desktop/start` with `intent="Open Notepad and type Hello World"`. Poll status. Verify.

**Task 5.2 — Manual smoke test: interactive WS mode**  
Write a 50-line Python test client that connects to `/desktop/ws/{id}`, captures its own screenshot via mss, sends it as the first message, executes the returned actions via pyautogui, and loops until `done`.

**Task 5.3 — Hackathon demo script**  
Prepare a specific canned demo: `intent="Open VS Code, create a new file called demo.py, type a Hello World program, and save it"`.

---

### Effort Summary

| Phase | Tasks | Estimated Time |
|---|---|---|
| 1 — Foundation | 1.1–1.4 | 2.5 hours |
| 2 — Handler extension | 2.1–2.2 | 1 hour |
| 3 — Routes + wiring | 3.1–3.3 | 2.5 hours |
| 4 — Tests | 4.1–4.3 | 3 hours |
| 5 — Integration + demo | 5.1–5.3 | 2 hours |
| **Total** | | **~11 hours** |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **R1 — Gemini coordinate accuracy on high-res screenshots** | High | High | Explicitly inject screen dimensions in every user turn (already done in WebPilotHandler). If accuracy is poor, add a pre-processing step to downscale the screenshot to 1920x1080 max before sending to Gemini, and scale returned coordinates back up. |
| **R2 — pyautogui typewrite Unicode failure** | High on Windows | Medium | Implement clipboard-paste fallback: `pyperclip.copy(text)` + `pyautogui.hotkey('ctrl','v')`. Add `pyperclip` to requirements.txt. The `_type_sync` method should try typewrite first and fall back to clipboard. |
| **R3 — HiDPI scaling formula wrong on specific OS/display combos** | Medium | High | Write explicit unit tests for all 4 scale scenarios. Accept a `DESKTOP_SCALE_OVERRIDE` env var (float) that bypasses auto-detection for the demo. |
| **R4 — mss hangs on certain GPU/display driver configs** | Low–Medium | High | Set a 5-second timeout on `mss.grab()` by running it in a thread with `asyncio.wait_for`. If it times out, return the last known screenshot. |
| **R5 — pyautogui mouse drifts to corner (FailSafe)** | Low | High | The FAILSAFE is intentional. Document it. Add a `DESKTOP_DISABLE_FAILSAFE=true` env var for demo contexts where mouse may accidentally reach the corner. Never disable by default. |
| **R6 — Gemini returns "navigate" action for desktop task** | Medium | Low | `DesktopAction` Literal validation excludes "navigate" — it raises `ValidationError` → `get_next_action_desktop()` falls back to `done` with a descriptive narration. The DESKTOP_SYSTEM_PROMPT explicitly tells Gemini that navigate is not available. |
| **R7 — Server-side loop writes to shared screen while user is working** | High (in practice) | Medium | Document that server-side autonomous mode (`/desktop/start`) takes full control of the screen. Recommend the interactive WS mode for production use. Provide a `DESKTOP_AUTONOMOUS_REQUIRE_CONFIRM=true` env var that gates all actions through a simple CLI prompt. |
| **R8 — Cloud Run deploy accidentally exposes desktop endpoint** | Low | High | `_check_desktop_enabled()` gates all `/desktop/*` endpoints behind `DESKTOP_MODE_ENABLED=true`. Cloud Run containers do not have a display, so even if the env var were set, `mss.grab()` would fail at `start()` with an informative error, preventing the executor from ever being available. |
| **R9 — Screen recording permission not granted (macOS)** | High on fresh Mac | Medium | `_init_sync()` runs `self._mss.grab(self._mss.monitors[0])` as a warm-up capture. If it raises, a clear `DesktopExecutorError` is raised with actionable instructions: "Grant screen recording permission to Python in System Preferences." |
| **R10 — Action history grows unbounded across long sessions** | Low | Medium | The existing history trim `if len(session.history) >= 20: session.history = session.history[-18:]` is copied from webpilot_routes and applied identically in the desktop loop. |
| **R11 — Test suite runs on CI (no display)** | Certain | High | `DESKTOP_MOCK=true` env var activates `DesktopExecutorStub` (all action methods no-op). All tests in `test_desktop_api.py` set this env var in their fixtures. Tests that require a real display are marked `@pytest.mark.skip(reason="requires display")`. |

---

## Appendix: File Change Summary

```
NEW FILES:
  src/executor/base.py
  src/executor/desktop.py
  src/agent/desktop_system_prompt.py
  src/api/desktop_models.py
  src/api/desktop_routes.py
  tests/test_desktop_executor.py
  tests/test_desktop_api.py

MODIFIED FILES:
  src/executor/__init__.py          — export DesktopExecutor, AbstractExecutor
  src/api/webpilot_models.py        — add right_click/double_click/move to Literal
  src/agent/webpilot_handler.py     — system_prompt_override param + get_next_action_desktop()
  src/api/server.py                 — lifespan desktop init, router include, health field
  requirements.txt                  — add mss, pyautogui
  CLAUDE.md                         — sync architecture, endpoints, env vars, test counts

UNCHANGED:
  src/agent/core.py                 — no change needed
  src/agent/vision.py               — no change needed
  src/agent/planner.py              — no change needed
  src/executor/browser.py           — no change needed
  src/executor/actions.py           — no change needed (ActionResult reused by desktop)
  src/api/webpilot_routes.py        — no change needed
  src/api/models.py                 — no change needed
  src/api/store*.py                 — no change needed
  webpilot-extension/*              — no change needed (future: add Desktop toggle)
```

---

## Appendix: Demo Scenarios for Hackathon Judges

The following are ready-to-run demo intents that should work on a standard Windows desktop without special setup:

```python
# Demo 1 — Native app automation
intent = "Open Notepad, type 'Hello from AI Desktop Mode!', and save the file as ai_test.txt on the Desktop"

# Demo 2 — Cross-app workflow
intent = "Open Calculator, compute 1337 * 42, take note of the result"

# Demo 3 — File system operation
intent = "Open File Explorer, navigate to Documents folder, and create a new folder named 'AI_Demo'"

# Demo 4 — VS Code (if installed)
intent = "Open VS Code, create a new file, write a Python hello world program, and save it as hello.py"

# Demo 5 — System settings
intent = "Open Settings, navigate to Display settings, and tell me the current screen resolution"
```

Each demo is concise, observable in real time, and reversible (except Demo 3 which creates a folder — acceptable for a demo). Demo 5 is a "read and report" task that tests Gemini's observation capability without requiring destructive writes.
