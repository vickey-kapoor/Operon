/**
 * WebPilot background service worker.
 *
 * Owns the single WebSocket connection to the backend and coordinates
 * between the sidebar (via chrome.runtime messaging) and the content
 * script (via chrome.tabs.sendMessage).
 */

const DEFAULT_BACKEND = "https://ui-navigator-749384771196.us-central1.run.app";
let BACKEND_URL = DEFAULT_BACKEND;
let WS_BASE_URL = DEFAULT_BACKEND.replace("http", "ws");

// Load persisted backend URL from sync storage on startup.
// Override via DevTools console: chrome.storage.sync.set({backendUrl: "http://your-host:8080"})
chrome.storage.sync.get("backendUrl", (res) => {
  if (res.backendUrl) {
    BACKEND_URL = res.backendUrl;
    WS_BASE_URL = res.backendUrl.replace(/^http/, "ws");
    log("log", "Using custom backend URL:", BACKEND_URL);
  }
});

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_DELAY_MS = 30000;
let _reconnectDelay = RECONNECT_DELAY_MS;
const ACTION_SETTLE_DELAY_MS = 1200;
const NAVIGATE_SETTLE_DELAY_MS = 2500; // kept as fallback only

let _ws = null;
let _sessionId = null;
let _reconnecting = false;
let _stepCounter = 0;
let _consecutiveFailures = 0;
let _messageQueue = Promise.resolve(); // serialises handleServerMessage calls
let _pendingOutbound = [];

const MAX_STEPS = 15;
const MAX_CONSECUTIVE_FAILURES = 3;
const INTERRUPT_WATCHDOG_MS = 10000; // 10s watchdog after interrupt
let _interruptWatchdog = null;

// ---------------------------------------------------------------------------
// Logging — all logs tagged [WebPilot] with a step counter where relevant
// ---------------------------------------------------------------------------

function log(level, ...args) {
  const tag = "[WebPilot]";
  if (level === "error") console.error(tag, ...args);
  else if (level === "warn")  console.warn(tag, ...args);
  else                        console.log(tag, ...args);
}

function logStep(label, data = {}) {
  _stepCounter++;
  console.group(`[WebPilot] Step ${_stepCounter}: ${label}`);
  if (Object.keys(data).length) console.table(data);
  console.groupEnd();
  return _stepCounter;
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

async function getOrCreateSession() {
  const stored = await chrome.storage.session.get("sessionId");
  if (stored.sessionId) {
    _sessionId = stored.sessionId;
    log("log", "Reusing session:", _sessionId);
    return _sessionId;
  }

  const DELAYS = [1000, 2000, 4000, 8000, 15000];
  for (let attempt = 0; attempt <= DELAYS.length; attempt++) {
    try {
      const resp = await fetch(`${BACKEND_URL}/webpilot/sessions`, { method: "POST" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      _sessionId = data.session_id;
      await chrome.storage.session.set({ sessionId: _sessionId });
      log("log", "Session created:", _sessionId);
      return _sessionId;
    } catch (err) {
      const delay = DELAYS[attempt];
      if (delay === undefined) {
        log("error", "Backend unreachable after all retries:", err.message);
        broadcastToSidebar({ type: "WS_STATUS", connected: false, error: "Backend unreachable" });
        return null;
      }
      log("warn", `Session create failed (attempt ${attempt + 1}), retrying in ${delay}ms:`, err.message);
      await sleep(delay);
    }
  }
}

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

async function connectWebSocket() {
  if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const sid = await getOrCreateSession();
  if (!sid) return;

  const url = `${WS_BASE_URL}/webpilot/ws/${sid}`;
  log("log", "Connecting WebSocket →", url);
  _ws = new WebSocket(url);

  _ws.onopen = () => {
    log("log", "WebSocket connected ✓");
    _reconnecting = false;
    _reconnectDelay = RECONNECT_DELAY_MS;  // reset backoff on success
    broadcastToSidebar({ type: "WS_STATUS", connected: true });
    flushPendingOutbound();
  };

  _ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      log("warn", "Unparseable WS message:", event.data);
      return;
    }
    log("log", "← Server:", msg.type, msg);
    // Serialise: wait for the previous message to finish before handling the next.
    // This prevents concurrent executeAction calls from racing each other.
    // The .catch ensures a crashed handler doesn't poison the entire queue.
    _messageQueue = _messageQueue
      .then(() => handleServerMessage(msg))
      .catch(err => log("error", "Message handler crashed:", err));
  };

  _ws.onclose = (event) => {
    log("warn", "WebSocket closed", { code: event.code, reason: event.reason });
    _ws = null;
    broadcastToSidebar({ type: "WS_STATUS", connected: false });

    if (event.code === 4404) {
      log("warn", "Stale session ID — clearing and reconnecting fresh");
      _sessionId = null;
      chrome.storage.session.remove("sessionId");
    }

    if (!_reconnecting) {
      _reconnecting = true;
      const delay = _reconnectDelay;
      _reconnectDelay = Math.min(_reconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
      setTimeout(connectWebSocket, delay);
    }
  };

  _ws.onerror = (err) => {
    log("error", "WebSocket error:", err);
  };
}

function sendWS(payload) {
  if (_ws && _ws.readyState === WebSocket.OPEN) {
    log("log", "→ Server:", payload.type, payload);
    _ws.send(JSON.stringify(payload));
  } else {
    log("warn", "WS not open — queueing message:", payload.type);
    _pendingOutbound.push(payload);
    if (_pendingOutbound.length > 20) {
      _pendingOutbound = _pendingOutbound.slice(-20);
    }
    connectWebSocket();
  }
}

function flushPendingOutbound() {
  if (!_ws || _ws.readyState !== WebSocket.OPEN || _pendingOutbound.length === 0) {
    return;
  }

  const queued = _pendingOutbound;
  _pendingOutbound = [];
  for (const payload of queued) {
    log("log", "→ Server (flushed):", payload.type, payload);
    _ws.send(JSON.stringify(payload));
  }
}

// ---------------------------------------------------------------------------
// Server → sidebar forwarding + action execution
// ---------------------------------------------------------------------------

async function handleServerMessage(msg) {
  switch (msg.type) {
    case "thinking":
      broadcastToSidebar({ type: "WP_MSG", payload: msg });
      break;

    case "action": {
      const step = logStep(`action: ${msg.action}`, {
        action: msg.action,
        label: msg.action_label,
        narration: msg.narration,
        x: msg.x,
        y: msg.y,
        target: msg.target,
        text: msg.text,
      });
      broadcastToSidebar({ type: "WP_MSG", payload: { ...msg, _step: step } });

      // Start action watchdog — navigate gets longer timeout due to page load.
      // Cleared when screenshot is sent back (in executeAction).
      clearTimeout(_interruptWatchdog);
      const watchdogMs = (msg.action === "navigate") ? 20000 : INTERRUPT_WATCHDOG_MS;
      _interruptWatchdog = setTimeout(() => {
        log("warn", `Action watchdog fired (${watchdogMs}ms) — no screenshot sent, forcing stop`);
        sendWS({ type: "stop" });
        broadcastToSidebar({ type: "WP_MSG", payload: { type: "stopped", narration: "Action timed out." } });
        _stepCounter = 0;
        _consecutiveFailures = 0;
      }, watchdogMs);

      // Hard stop if too many steps — prevent infinite loops.
      if (_stepCounter > MAX_STEPS) {
        log("warn", `Max steps (${MAX_STEPS}) reached — stopping task`);
        sendWS({ type: "stop" });
        broadcastToSidebar({ type: "WP_MSG", payload: { type: "error", message: `Stopped after ${MAX_STEPS} steps without completing.` } });
        _stepCounter = 0;
        _consecutiveFailures = 0;
        break;
      }

      const success = await executeAction(msg);
      log(success ? "log" : "warn", `Step ${step} execute: ${success ? "✓ OK" : "✗ FAILED"}`);
      broadcastToSidebar({ type: "WP_MSG", payload: { type: "step_result", step, success } });

      if (!success) {
        _consecutiveFailures++;
        if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          log("warn", `${MAX_CONSECUTIVE_FAILURES} consecutive failures — stopping task`);
          sendWS({ type: "stop" });
          broadcastToSidebar({ type: "WP_MSG", payload: { type: "error", message: `Stopped after ${MAX_CONSECUTIVE_FAILURES} consecutive failed actions.` } });
          _stepCounter = 0;
          _consecutiveFailures = 0;
        }
      } else {
        _consecutiveFailures = 0;
      }
      break;
    }

    case "confirmation_required":
      broadcastToSidebar({ type: "WP_MSG", payload: msg });
      break;

    case "paused":
      broadcastToSidebar({ type: "WP_MSG", payload: msg });
      break;

    case "done":
    case "stopped":
      if (_interruptWatchdog) { clearTimeout(_interruptWatchdog); _interruptWatchdog = null; }
      broadcastToSidebar({ type: "WP_MSG", payload: msg });
      log("log", `Task ${msg.type} — ${_stepCounter} steps executed`);
      _stepCounter = 0;
      _consecutiveFailures = 0;
      // Keep WS open — the server's outer loop waits for the next task message.
      // Closing here causes a race: a new task submitted before reconnect finishes gets dropped.
      break;

    case "error":
      log("error", "Server error:", msg.message);
      broadcastToSidebar({ type: "WP_MSG", payload: msg });
      break;

    default:
      log("warn", "Unknown message type:", msg.type);
      break;
  }
}

// ---------------------------------------------------------------------------
// Action execution
// ---------------------------------------------------------------------------

async function executeAction(action) {
  const actionType = action.action;

  try {
    if (actionType === "navigate") {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) { log("warn", "No active tab for navigate"); return false; }
      const url = action.target || action.url;
      log("log", `Navigating tab ${tab.id} → ${url}`);
      await chrome.tabs.update(tab.id, { url });
      await waitForTabLoad(tab.id);
    } else {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) { log("warn", "No active tab for", actionType); return false; }
      log("log", `Sending ${actionType} to content script (tab ${tab.id})`);
      // Ensure content script is present — re-inject in case tab navigated.
      // Only inject into http/https pages (not chrome:// or extension pages).
      if (tab.url && (tab.url.startsWith("http://") || tab.url.startsWith("https://"))) {
        try {
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"],
          });
          await sleep(200); // brief pause for script to initialise
        } catch {
          // Already injected or access denied — ignore.
        }
      }
      try {
        const resp = await chrome.tabs.sendMessage(tab.id, { type: "EXECUTE_ACTION", action });
        log("log", `Content script response:`, resp);
        if (resp && !resp.success) {
          log("warn", "Content script reported failure:", resp.error);
        }
      } catch (err) {
        log("warn", "Content script error:", err.message);
      }
      // Wait for DOM to stabilise — fall back to fixed delay if content script unreachable
      try {
        await chrome.tabs.sendMessage(tab.id, { type: "WAIT_STABLE", timeout: 5000 });
      } catch {
        await sleep(ACTION_SETTLE_DELAY_MS);
      }
    }

    // Capture screenshot and send back (include current URL for Gemini context).
    const screenshot = await captureScreenshot();
    if (screenshot) {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      sendWS({ type: "screenshot", screenshot, current_url: tab?.url || "" });
      // Clear action watchdog — screenshot sent successfully.
      if (_interruptWatchdog) {
        clearTimeout(_interruptWatchdog);
        _interruptWatchdog = null;
      }
      return true;
    } else {
      log("error", "Screenshot capture failed — cannot continue loop");
      return false;
    }
  } catch (err) {
    log("error", `executeAction(${actionType}) threw:`, err);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Screenshot capture
// ---------------------------------------------------------------------------

async function captureScreenshot() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) { log("warn", "captureScreenshot: no active tab"); return null; }
    log("log", `Capturing tab ${tab.id} (${tab.url})`);
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    const b64 = dataUrl.split(",")[1];
    log("log", `Screenshot captured — ${Math.round(b64.length / 1024)} KB`);
    return b64;
  } catch (err) {
    log("error", "Screenshot failed:", err.message);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Sidebar ↔ background messaging
// ---------------------------------------------------------------------------

function broadcastToSidebar(msg) {
  chrome.runtime.sendMessage(msg).catch(() => {
    // Sidebar might not be open — ignore.
  });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "TASK") {
    _stepCounter = 0;
    _consecutiveFailures = 0;
    log("log", `New task: "${msg.intent}"`);
    captureScreenshot().then((screenshot) => {
      if (!screenshot) {
        log("warn", "Could not capture initial screenshot for task");
        sendResponse({ ok: false });
        return;
      }
      sendWS({ type: "task", intent: msg.intent, screenshot });
      sendResponse({ ok: true });
    });
    return true;
  }

  if (msg.type === "INTERRUPT") {
    log("log", `Interrupt: "${msg.instruction}"`);
    captureScreenshot().then((screenshot) => {
      sendWS({ type: "interrupt", instruction: msg.instruction, screenshot });
      // Reset step counter so client and server count from the same baseline.
      _stepCounter = 0;
      _consecutiveFailures = 0;
      // Watchdog is started later in handleServerMessage when the replanned
      // action arrives — timeout depends on action type (navigate=20s, other=10s).
      sendResponse({ ok: true });
    });
    return true;
  }

  if (msg.type === "CONFIRM") {
    log("log", `Confirm: ${msg.confirmed}`);
    sendWS({ type: "confirm", confirmed: msg.confirmed });
    sendResponse({ ok: true });
  }

  if (msg.type === "RESUME") {
    log("log", "Resume requested (after CAPTCHA/login)");
    captureScreenshot().then((screenshot) => {
      if (!screenshot) {
        log("warn", "Could not capture screenshot for resume");
        sendResponse({ ok: false });
        return;
      }
      sendWS({ type: "resume", screenshot });
      sendResponse({ ok: true });
    });
    return true;
  }

  if (msg.type === "STOP") {
    log("log", "Stop requested");
    sendWS({ type: "stop" });
    sendResponse({ ok: true });
  }

  if (msg.type === "GET_STATUS") {
    sendResponse({
      connected: _ws && _ws.readyState === WebSocket.OPEN,
      sessionId: _sessionId,
    });
  }
});

// ---------------------------------------------------------------------------
// Open sidebar on icon click or keyboard shortcut
// ---------------------------------------------------------------------------

chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
});

chrome.commands.onCommand.addListener((command) => {
  if (command === "open-sidebar") {
    chrome.sidePanel.open({ windowId: chrome.windows.WINDOW_ID_CURRENT }).catch(() => {});
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

// Keep the service worker alive with a periodic alarm (Chrome kills SWs after 30s).
chrome.alarms.create("keepalive", { periodInMinutes: 0.4 }); // every ~25s
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "keepalive") {
    // Re-connect if WS dropped while SW was dormant.
    if (!_ws || _ws.readyState === WebSocket.CLOSED) {
      connectWebSocket();
    }
  }
});

chrome.runtime.onInstalled.addListener(() => {
  connectWebSocket();
});

chrome.runtime.onStartup.addListener(() => {
  connectWebSocket();
});

connectWebSocket();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait until the given tab's status is "complete", with a 15s timeout.
 * Also waits an extra 1s after load for JS-heavy pages (Gmail, etc.) to render.
 */
function waitForTabLoad(tabId, timeoutMs = 15000) {
  return new Promise((resolve) => {
    const deadline = setTimeout(() => {
      log("warn", `Tab ${tabId} load timeout — proceeding anyway`);
      chrome.tabs.onUpdated.removeListener(listener);
      resolve();
    }, timeoutMs);

    function listener(updatedTabId, changeInfo) {
      if (updatedTabId !== tabId) return;
      log("log", `Tab ${tabId} status: ${changeInfo.status}`);
      if (changeInfo.status === "complete") {
        chrome.tabs.onUpdated.removeListener(listener);
        clearTimeout(deadline);
        // Extra settle time for JS-heavy apps like Gmail.
        sleep(1500).then(resolve);
      }
    }

    chrome.tabs.onUpdated.addListener(listener);
  });
}
