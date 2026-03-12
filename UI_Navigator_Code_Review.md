# UI Navigator — Code Review

**Reviewed:** All backend Python source, FastAPI server, WebPilot routes/handler, Playwright executor, and Chrome extension background/content scripts.

**Verdict:** Genuinely impressive for a phased hackathon build. The architecture is clean, the test coverage is real, and several subtle production concerns are handled well (SSRF protection, Playwright stealth, retry logic, WS event replay, keep-alive alarms). The issues below range from critical bugs to hardening opportunities.

---

## Critical Bugs

### 1. Conversation history is malformed — agent multi-turn is broken

**File:** `src/agent/core.py` — `_update_history()`

The method only appends the **model** turn to history:
```python
history.append(
    genai_types.Content(role="model", parts=[...])
)
```
The **user** turn (screenshot + prompt) built in `vision.py::_build_user_turn` is passed directly to `generate_content` but never stored in `history`. After two steps, `contents` looks like:
```
[model_turn_1, model_turn_2, user_turn_3]
```
Gemini requires strictly alternating `user → model → user → model` turns. Sending model turns before a user turn either causes API errors or causes the model to silently ignore all history. The agent effectively operates without memory beyond the current step.

**Fix:** Store both the user and model turn after each step:
```python
history.append(user_turn)          # genai_types.Content already built in vision.py
history.append(model_turn)
```
This requires `analyze_screen` to return the user `Content` object it built, or for `_update_history` to reconstruct it from the screenshot.

---

### 2. `_rate_windows` dict is an unbounded memory leak

**File:** `src/api/server.py`

`_rate_windows` is a `defaultdict(deque)` keyed by API key. Keys are never removed. An attacker (or normal churn) sending many requests with different `X-API-Key` values will fill the dict indefinitely with stale deques. Under load this will grow without bound.

**Fix:** Evict stale keys when the sliding window becomes empty:
```python
async with _rate_lock:
    window = _rate_windows[api_key]
    # ... prune old timestamps ...
    if len(window) == 0:
        del _rate_windows[api_key]  # evict empty entries
```

---

### 3. Confirmation flow dead code — `confirm_event` / `confirm_result` unused

**File:** `src/api/webpilot_routes.py` and `src/api/webpilot_models.py`

`WebPilotSession` has `confirm_event: asyncio.Event` and `confirm_result: Optional[bool]` fields. The websocket endpoint's outer loop handles `{"type": "confirm"}` by setting these:
```python
session.confirm_result = msg.confirmed
session.confirm_event.set()
```
But `_run_action_loop` **never reads them** — it blocks on `raw_confirm = await websocket.receive_text()` directly. The outer `while True` loop is suspended during this receive, so `confirm_event` and `confirm_result` can never be set while the inner loop is waiting. These two fields are functionally dead code.

This also means a `{"type": "confirm"}` message sent while the outer loop is active (i.e., outside a confirmation flow) will silently set stale state that could interfere with subsequent confirmations.

**Fix:** Remove `confirm_event` and `confirm_result` from `WebPilotSession` and document that the confirmation is resolved inline. Or refactor to an event-driven approach that actually uses them.

---

## Security Issues

### 4. API allows arbitrary model and system prompt override

**File:** `src/api/models.py` — `NavigateRequest`

Any authenticated caller can pass:
```json
{"task": "...", "model": "gemini-1.5-pro", "system_prompt": "Ignore all previous instructions..."}
```
The `model` field has no validation — a caller could point to any Gemini model, including ones with very high per-token cost. The `system_prompt` override gives callers full control over agent behavior, which could be used to bypass intended restrictions.

**Fix:** Validate `model` against an allowlist of approved models. Consider removing `system_prompt` from the public API or restricting it to admin keys.

---

### 5. CORS defaults to wildcard `"*"` in production

**File:** `src/api/server.py`

```python
_cors_origins: List[str] = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw
    else ["*"]    # ← wildcard if CORS_ORIGINS not set
)
```
If `CORS_ORIGINS` is not configured in the Cloud Run environment (easy to forget), the server accepts cross-origin requests from any website. Combined with `allow_credentials=True`, this is a serious misconfiguration.

**Fix:** Default to a restrictive list (`["chrome-extension://*"]`) and require explicit configuration for web origins in production. Add a startup warning if `CORS_ORIGINS` is unset.

---

### 6. Auth entirely disabled when `API_KEYS` is not set

**File:** `src/api/server.py` — `APIKeyMiddleware`

```python
api_keys = _get_api_keys()
if not api_keys:
    return await call_next(request)  # skip auth silently
```
This is intentionally a dev convenience, but there is no warning at startup when auth is disabled. In a misconfigured Cloud Run deployment (if `API_KEYS` secret is missing), the server exposes all endpoints unauthenticated.

**Fix:** Log a prominent `WARNING` in the lifespan startup when `API_KEYS` is not set.

---

## Logic Errors

### 7. `metrics.emit("tasks_started")` fires twice per task

**Files:** `src/api/server.py` (`_run_agent_task`) and `src/agent/core.py` (`run()`)

Both call `metrics.emit("tasks_started")` for every task. The metric will be double-counted in Cloud Monitoring, making the dashboard misleading.

**Fix:** Remove the call from `core.py::run()` — the server layer is the canonical place to emit task lifecycle metrics.

---

### 8. `_prev_hash` not reset when stuck counter resets

**File:** `src/api/webpilot_routes.py` — `_run_action_loop`

When `retry_count >= 3` the stuck flag is sent and `retry_count` resets to 0, but `_prev_hash` is not reset. If the model tries something new that happens to screenshot-hash identically to an old state (e.g., navigating back to an earlier page), the stuck condition will re-trigger prematurely on the very next identical frame.

**Fix:** Reset `_prev_hash` together with `retry_count`:
```python
if stuck:
    retry_count = 0
    _prev_hash = b""   # force fresh baseline
```

---

### 9. `"forget it"` classified as REDIRECT, not ABORT

**File:** `src/agent/webpilot_handler.py` — `classify_interruption_type`

`redirect_keywords = {"instead", "forget", ...}` — the word `"forget"` is in the redirect set. Because redirect is checked before abort, the phrase `"forget it"` matches `"forget"` and returns `REDIRECT` (start over with new goal), not `ABORT` (stop entirely). A user saying "forget it" almost certainly means stop, not redirect.

**Fix:** Replace the broad `"forget"` keyword with more specific phrases (`"forget about it"`, `"forget the task"`) and add `"forget it"` explicitly to the abort set.

---

## Performance / Robustness

### 10. Blocking `time.sleep()` retry in a thread pool without bounded concurrency

**File:** `src/agent/vision.py` — `_call_with_retry`

The method runs in `run_in_executor(None, ...)` — the default `ThreadPoolExecutor`. With `MAX_RETRIES=3` and `RETRY_BACKOFF=2.0s` doubling each attempt, a single failing task can hold a thread for up to ~14 seconds. With `MAX_CONCURRENT_TASKS=5`, five simultaneously failing tasks exhaust the default thread pool, blocking all other executor work (including other Gemini calls).

**Fix:** Provide a dedicated `ThreadPoolExecutor` with a capacity matching `MAX_CONCURRENT_TASKS`:
```python
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_CONCURRENT)
# in vision.py:
await loop.run_in_executor(_executor, self._call_with_retry, ...)
```

---

### 11. `text` field in `Action` has no length cap

**File:** `src/executor/actions.py`

`text: Optional[str]` on `Action` is unbounded. The `_type()` method types each character with a 30ms delay:
```python
await page.keyboard.type(text, delay=30)
```
Gemini could theoretically return a very long `text` value (or a malicious user could craft one). A 10,000-character string takes 5 minutes to type.

**Fix:** Add `max_length=10000` (or a tighter limit) to the `text` field in `Action`.

---

### 12. `wait` action can stall the agent for 30 seconds

**File:** `src/executor/actions.py` and `src/executor/browser.py`

`duration_ms` is capped at 30000ms. Gemini can legitimately emit a 30-second wait, which blocks the agent step loop for that full duration with no cancellation path (except task cancellation from outside). The `asyncio.CancelledError` path in `_run_agent_task` won't fire until after the sleep completes.

**Fix:** Cap `duration_ms` at a lower practical limit (e.g., 5000ms) and use `asyncio.sleep` wrapped in `asyncio.wait_for` or periodic yield.

---

## Code Quality

### 13. `logger.exception` called with wrong signature

**File:** `src/api/server.py` — `_run_agent_task`

```python
logger.exception("Background task raised", exc, extra={"task_id": task_id})
```
`logger.exception(msg, *args)` treats positional arguments after `msg` as `%`-style format args, so `exc` would be formatted into the message string if it contained `%s`. The exception info is already captured automatically by `logger.exception`. The intended call is:
```python
logger.exception("Background task raised: %s", exc, extra={"task_id": task_id})
```

---

### 14. Extension hardcodes `localhost:8080`

**File:** `webpilot-extension/background.js`

```javascript
const BACKEND_URL = "http://localhost:8080";
const WS_BASE_URL = "ws://localhost:8080";
```
There is no way for a user to point the extension at a deployed Cloud Run instance without editing and rebuilding the extension. This is fine for development but is a usability gap.

**Fix:** Read the backend URL from `chrome.storage.sync` with a fallback to localhost, and add a simple settings page to the extension.

---

### 15. Message queue in `background.js` has no timeout guard

**File:** `webpilot-extension/background.js`

```javascript
_messageQueue = _messageQueue.then(() => handleServerMessage(msg));
```
`handleServerMessage` for an `"action"` message calls `executeAction`, which calls `waitForTabLoad` (15s timeout) + `sleep(ACTION_SETTLE_DELAY_MS)`. If `captureScreenshot()` throws unexpectedly, the Promise chain rejects and all subsequent messages are silently dropped (an unhandled rejection that poisons the queue).

**Fix:** Add a `.catch` to prevent chain poisoning:
```javascript
_messageQueue = _messageQueue
  .then(() => handleServerMessage(msg))
  .catch(err => log("error", "Message handler crashed:", err));
```

---

## Minor / Suggestions

**`bypass_csp: False` in browser context** — correct and safe; keep it.

**`take_and_analyze_screenshot` creates its own executor** — the one-shot `/screenshot` endpoint spins up a full Chromium instance. For high-traffic use, consider a shared warm browser pool.

**WebPilot session cleanup logs every 5 minutes even with zero sessions** — add an early-return if `_sessions` is empty.

**`navigate` in `browser.py` does no SSRF check on Gemini-generated URLs** — the risk is low since the initial URL is validated, but defence-in-depth would be consistent with the rest of the security model.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **Critical** | `agent/core.py` | History only stores model turns — multi-turn context broken |
| 2 | **Critical** | `api/server.py` | `_rate_windows` grows unbounded — memory leak |
| 3 | **High** | `api/webpilot_routes.py` | `confirm_event`/`confirm_result` are dead code, could corrupt state |
| 4 | **High** | `api/models.py` | `model` and `system_prompt` fields allow unchecked overrides |
| 5 | **High** | `api/server.py` | CORS defaults to `*` with credentials enabled |
| 6 | **Medium** | `api/server.py` | No startup warning when auth is silently disabled |
| 7 | **Medium** | `agent/core.py` + `server.py` | `tasks_started` metric emitted twice |
| 8 | **Medium** | `api/webpilot_routes.py` | `_prev_hash` not reset on stuck counter reset |
| 9 | **Medium** | `agent/webpilot_handler.py` | `"forget it"` misclassified as REDIRECT |
| 10 | **Medium** | `agent/vision.py` | Thread pool can be exhausted by retry backoff under load |
| 11 | **Low** | `executor/actions.py` | `text` field uncapped — slow typing on long strings |
| 12 | **Low** | `executor/browser.py` | 30s `wait` action blocks agent step with no cancellation |
| 13 | **Low** | `api/server.py` | `logger.exception` wrong call signature |
| 14 | **Low** | `background.js` | Backend URL hardcoded — no production configurability |
| 15 | **Low** | `background.js` | Message queue Promise chain can be poisoned by unhandled rejection |
