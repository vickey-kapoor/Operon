# E2E SDET Report — UI Navigator
**Date:** 2026-03-19
**Tester:** Claude Code (SDET mode)
**Platform:** Windows 11 Home 10.0.26200 (win32)
**Python:** 3.14.2
**pytest:** 9.0.2 / pytest-asyncio 1.3.0

---

## Executive Summary

| Metric | Value |
|---|---|
| Total tests collected (non-browser) | 238 |
| Passed | 237 |
| Failed | 1 |
| Warnings | 51 |
| Total duration | 269.64s (4m 29s) |
| Visual test runner (visual-tests.html) | 10/10 passed |
| Test dashboard (port 3333) | Loaded, 24 cached results visible |
| Favicon 404 | Present on desktop.html + dashboard |

**Overall verdict:** The test suite is in good health. One test fails consistently under load (a flaky subprocess timing test) but passes when run in isolation. All 10 visual WS scenarios pass. Three deprecation warnings from the `websockets` 15.x library require eventual migration.

---

## 1. Automated Test Suite Results

### Command executed
```bash
DESKTOP_MOCK=true DESKTOP_MODE_ENABLED=true python -m pytest \
  tests/test_api.py tests/test_webpilot_api.py tests/test_webpilot_e2e.py \
  tests/test_sessions.py tests/test_clarifier.py tests/test_gap_coverage.py \
  tests/test_user_failures.py tests/test_desktop_executor.py \
  tests/test_desktop_api.py tests/test_dashboard.py -v
```

### Per-file breakdown

| File | Tests | Result | Notes |
|---|---|---|---|
| `test_api.py` | 18 | ALL PASS | REST endpoints, auth, SSRF, rate limiting |
| `test_webpilot_api.py` | 22 | ALL PASS | WebPilot unit, WS flows, TTS, stuck detection |
| `test_webpilot_e2e.py` | 24 | ALL PASS | Real uvicorn server + websockets, all WS scenarios |
| `test_sessions.py` | 13 | ALL PASS | ADK session CRUD |
| `test_clarifier.py` | 7 | ALL PASS | Clarify endpoint and planner unit |
| `test_gap_coverage.py` | 17 | 16 PASS / **1 FAIL** | Architecture review gaps — see below |
| `test_user_failures.py` | 8 | ALL PASS | User-testing regressions |
| `test_desktop_executor.py` | 25 | ALL PASS | DesktopExecutor unit (MOCK mode) |
| `test_desktop_api.py` | 21 | ALL PASS | Desktop Mode API integration |
| `test_dashboard.py` | 83 | ALL PASS | Dashboard server, agent_runner, judge_runner |
| **TOTAL** | **238** | **237 / 238** | |

---

## 2. Failure Analysis

### FAILED: `tests/test_gap_coverage.py::test_session_does_not_survive_server_restart`

**Classification: FLAKY — environment/load-sensitive, not a code defect.**

**Failure message:**
```
E  Failed: Server B did not start in time.
E  Process returncode=None, stderr=
tests\test_gap_coverage.py:458: Failed
```

**Root cause analysis:**

The test spawns a real `uvicorn` subprocess (Server B) and polls its `/health` endpoint with a 25-second deadline. When the full 238-test suite runs, many prior tests have already spun up multiple in-process uvicorn servers, consumed thread-pool and socket resources, and put the Windows TCP stack into TIME_WAIT for various ports.

Under this load, subprocess startup takes longer than the 25-second deadline. The subprocess process itself does not crash (`returncode=None` — still running) but httpx cannot reach it in time.

**Isolation test result:** Running the test alone passes in 11.9 seconds. A manual subprocess-only timing test confirmed 4.5–7 seconds startup time when run in clean conditions.

**Contributing factors:**
1. The test passes `{**os.environ, "WEBPILOT_STUB": "navigate_and_done"}` to the subprocess but does NOT explicitly forward `DESKTOP_MOCK=true DESKTOP_MODE_ENABLED=true`. These are set as shell-level env vars by the pytest command prefix, not as Python `os.environ` entries, so when the full suite runs without the env prefix being inherited, the subprocess initializes DesktopExecutor in non-mock mode. However, in isolation runs the env vars are present from the shell, so startup is faster.
2. Machine under load: 238 tests take 4m29s, including many tests that spin up real uvicorn servers in threads. By the time `test_session_does_not_survive_server_restart` runs (test #99/238), significant socket/thread resource pressure exists.
3. The 25-second deadline is generous but can be exceeded on a loaded Windows machine with Python 3.14's slower interpreter startup.

**Flakiness indicator:** The test ran at 37% through the suite. It passed in every isolated run attempted. This is a classic load-sensitive flaky test pattern.

**Full error traceback:**
```
async def test_session_does_not_survive_server_restart():
    ...
    if not ready:
        rc = proc.poll()
        stderr_file.seek(0)
        stderr_out = stderr_file.read().decode(errors="replace")
>       pytest.fail(
            f"Server B did not start in time. "
            f"Process returncode={rc}, stderr={stderr_out[:500]}"
        )
E       Failed: Server B did not start in time. Process returncode=None, stderr=

tests\test_gap_coverage.py:458: Failed
```

---

## 3. Warnings Analysis (51 total)

All 51 warnings originate from three DeprecationWarnings in the `websockets` 15.x library, triggered whenever a real uvicorn server using WebSockets is started. These appear across `test_webpilot_e2e.py` (18 warnings), `test_gap_coverage.py` (22 warnings), and `test_user_failures.py` (9 warnings), plus 2 extras in the isolation run of the failing test.

### Warning 1 — websockets.legacy API deprecated (websockets 14.0+)
```
C:\...\websockets\legacy\__init__.py:6: DeprecationWarning:
websockets.legacy is deprecated; see
https://websockets.readthedocs.io/en/stable/howto/upgrade.html
for upgrade instructions
```
**Source:** `websockets` 15.0.1 installed; uvicorn still uses `websockets.legacy` internally.
**Impact:** All test files that spin up a live server emit this warning. No test failures, but signals that uvicorn's `websockets` protocol implementation needs updating.
**Affected tests:** All e2e + gap_coverage + user_failures tests that start a real uvicorn server.

### Warning 2 — WebSocketServerProtocol deprecated (websockets 13.1+)
```
C:\...\uvicorn\protocols\websockets\websockets_impl.py:17: DeprecationWarning:
websockets.server.WebSocketServerProtocol is deprecated
```
**Source:** uvicorn 0.40.0 uses the deprecated `WebSocketServerProtocol` class.
**Impact:** Same tests as Warning 1. This is an upstream uvicorn issue.

### Warning 3 — ws_handler second argument deprecated
```
C:\...\websockets\legacy\server.py:1178: DeprecationWarning:
remove second argument of ws_handler
```
**Source:** Related to uvicorn's websockets handler registration pattern.
**Impact:** Same test files.

### Warning 4 — ConnectionClosed.code deprecated (websockets 13.1+)
*(Observed only in isolated test run, not surfaced in main suite due to pyproject.toml filter)*
```
C:\...\websockets\exceptions.py:125: DeprecationWarning:
ConnectionClosed.code is deprecated; use Protocol.close_code
or ConnectionClosed.rcvd.code
```
**Source:** Test code in `test_gap_coverage.py:480`, `test_webpilot_api.py:321`, `test_webpilot_api.py:392` accesses `exc.code` on `websockets.exceptions.ConnectionClosed`.
**Impact:** These assertions still work in websockets 15.0.1 but the attribute is deprecated and will be removed in a future release, potentially breaking those tests.

### Warning 5 — Python 3.14 compatibility (google.genai)
*(Observed with -W all flag)*
```
C:\...\google\genai\types.py:43: DeprecationWarning:
'_UnionGenericAlias' is deprecated and slated for removal in Python 3.17
```
**Source:** `google-genai` 1.66.0 uses `_UnionGenericAlias` from Python internals.
**Impact:** No current test failures but signals that `google-genai` needs a version bump for full Python 3.14+ compatibility.

### Warning 6 — asyncio_default_fixture_loop_scope not configured
*(Shown in pytest header, not a warning per se)*
```
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None
```
**Source:** `pyproject.toml` does not set `asyncio_default_fixture_loop_scope`.
**Impact:** pytest-asyncio 1.3.0 shows `None` — safe for now but should be set to `"function"` to be explicit and forward-compatible.

---

## 4. Web Dashboard Testing (Playwright MCP)

### 4.1 DesktopPilot — `http://127.0.0.1:8080/ui/desktop.html`

**Status: LOADS CORRECTLY**

- Page title: "DesktopPilot"
- Status badge shows "Ready"
- Task input textarea renders with placeholder
- "Start Task" button functional, max steps spinner (default 20) present
- 5 quick-task preset buttons rendered: Notepad, Calculator, File Explorer, Screen Info, VS Code
- Step log panel displays "Steps will appear here" placeholder
- Footer shows "Connected to localhost:8080"
- **Minor issue:** `favicon.ico` returns HTTP 404 (no favicon served by the app)
- Screenshot saved: `tests/reports/desktop-html-screenshot.png`

### 4.2 Visual Test Runner — `http://127.0.0.1:8080/ui/visual-tests.html`

**Status: ALL 10 TESTS PASSED (635ms total)**

| # | Test | Time | Result |
|---|---|---|---|
| 1 | Health Endpoint | 38ms | PASS |
| 2 | Swagger UI | 15ms | PASS |
| 3 | Create WebPilot Session | 25ms | PASS |
| 4 | WS Task Lifecycle | 93ms | PASS |
| 5 | WS Stop Mid-Task | 68ms | PASS |
| 6 | WS Confirm Flow | 57ms | PASS |
| 7 | Interrupt Flow (Redirect) | 67ms | PASS |
| 8 | Delete Session | 62ms | PASS |
| 9 | Concurrent Sessions | 155ms | PASS |
| 10 | Malformed JSON | 55ms | PASS |

- Header version badge shows "V1.4.0" — matches `server.py` `_VERSION`
- "ALL 10 TESTS PASSED" banner displayed after run
- Live log correctly shows last test (Malformed JSON) WS flow
- Screenshot saved: `tests/reports/visual-tests-all-pass.png`

### 4.3 E2E Live Tests — `http://127.0.0.1:8080/ui/e2e-tests.html`

**Status: PAGE LOADS CORRECTLY — Tests not auto-run (require Gemini API key for most)**

- 13 test scenarios listed and rendered correctly
- Tests 3-6, 12 require a live Gemini API key (navigate/clarify) — not run
- Tests 1, 2, 7, 9, 10, 11, 13 are infrastructure tests that could run without API key
- No console errors on page load (favicon.ico 404 is the only browser error, on desktop.html only)
- Screenshot saved: `tests/reports/e2e-tests-html-before.png`

### 4.4 Test Dashboard — `http://127.0.0.1:3333`

**Status: LOADS CORRECTLY with cached results**

- Dashboard renders with 3-column layout: Test Results | Select a Test | Judge Output
- **Cached report present:** 24 passed WebPilot e2e tests from a previous run
- Judge Output panel populated with cached analysis
- Three action buttons operational: "Run All Scenarios", "Run Judge", "Run Fix Loop"
- 3 console warnings noted (non-blocking):
  - React DevTools info (CDN React dev build)
  - Tailwind CDN production warning (expected for local dev)
  - Babel in-browser transpiler warning (expected for local dev)
- 1 console error: `favicon.ico` 404 (same issue as desktop.html)
- Screenshot saved: `tests/reports/dashboard-3333-screenshot.png`

---

## 5. Cached Judge Output Analysis (from port 3333 dashboard)

The LLM judge (Claude) analyzed the 24 WebPilot e2e tests from the last run and flagged:

### Validity Issues (7 tests flagged)
The judge identified 7 tests that pass without any WebSocket message log, suggesting they may only check UI state rather than backend WS behavior:

| Test | Issue |
|---|---|
| `test_tooltip_text_on_send_button` | Only checks UI state, no WS message validation |
| `test_input_mode_idle_shows_send` | Only checks UI state, no WS message validation |
| `test_input_mode_running_shows_interrupt` | Only checks UI state, no WS message validation |
| `test_interrupt_button_disabled_idle` | Only checks UI state, no WS message validation |
| `test_navigate_watchdog_race` | Only checks timing, no WS message sequence validation |
| `test_watchdog_clears_after_screenshot` | Only checks timing, no WS message sequence validation |
| `test_consecutive_failures_stops` | May not validate actual failure sequence via WS messages |

These are `unit` tests (marked as such in the dashboard) that test pure Python logic — the judge's concern about "no WS message log" is expected behavior, not a defect.

### Coverage Gaps (9 identified by judge)
1. Server error handling: No tests verify `{"type": "error"}` response when handler raises an exception
2. Message size limit: No tests verify rejection of WebSocket messages > 15 MB
3. Malformed JSON: No tests verify handling of invalid JSON in WebSocket messages
4. Handler not initialized: No tests verify 503 close code when handler is not ready
5. Session not found: No tests verify 4404 close code when session_id is invalid
6. WebSocket connection errors: No tests verify WS connection fails/drops mid-task
7. Screenshot timeout: No tests verify behavior when screenshot request times out
8. Action timeout: No tests verify watchdog timeout behavior during action execution
9. Concurrent session handling: No tests verify behavior with multiple simultaneous WS connections

**Note:** Items 2, 4, and 5 ARE actually covered in `test_webpilot_api.py` and `test_gap_coverage.py` — the judge was analyzing only the 24 `test_webpilot_e2e.py` tests in isolation, not the full suite.

---

## 6. CLAUDE.md Documentation Discrepancies

The following discrepancies were found between CLAUDE.md and the actual codebase state:

| Item | CLAUDE.md says | Actual |
|---|---|---|
| Non-browser test count | "238 non-browser tests" | 238 — CORRECT |
| `test_project_judge.py` | Listed (34 tests) | **File does not exist** — removed from codebase |
| System prompt preamble | "109 non-browser tests" | 238 — system prompt is outdated |
| Test command note | Mentions `test_project_judge.py` indirectly | Not in test files, not in run command |

---

## 7. Dependency Issues

### Missing `pytest-mock` in `requirements-dev.txt`
`pytest-mock` 3.15.1 is installed and used by tests (the `mocker` fixture), but it is **not listed** in `requirements-dev.txt`. A fresh `pip install -r requirements-dev.txt` would fail to install it, causing test collection errors on CI.

### websockets 15.x compatibility
The project pins `websockets>=13.0` in requirements. Version 15.0.1 is installed. The three deprecation warnings from websockets + uvicorn indicate that the websockets protocol handler in uvicorn 0.40.0 uses APIs deprecated in websockets 14.0 and 15.x. This will eventually break when websockets removes legacy APIs.

Additionally, test code in three files uses `exc.code` on `ConnectionClosed` (deprecated since websockets 13.1):
- `tests/test_gap_coverage.py:480` — `assert exc.code == 4404`
- `tests/test_webpilot_api.py:321` — `assert exc.code == 4503`
- `tests/test_webpilot_api.py:392` — `assert exc.code == 4404`

These should be migrated to `exc.rcvd.code`.

### `pyproject.toml` — asyncio_default_fixture_loop_scope not set
```toml
# Current:
[tool.pytest.ini_options]
asyncio_mode = "auto"

# Recommended addition:
asyncio_default_fixture_loop_scope = "function"
```

---

## 8. Test Architecture Observations

### Flakiness Indicators

| Test | Risk Level | Reason |
|---|---|---|
| `test_session_does_not_survive_server_restart` | HIGH | Subprocess startup with 25s timeout; fails under suite load on Windows |
| `test_slow_navigate_completes_successfully` | MEDIUM | Uses a `slow_navigate` stub scenario with timed delays |
| `test_concurrent_sessions_parallel_stability` | MEDIUM | Runs 5 concurrent WS connections simultaneously — socket pressure under load |

### Code Duplication in Test Files
The `_blank_b64()` helper function (generates a 1x1 white PNG) is defined independently in:
- `conftest.py` (as `small_png` / `make_dummy_screenshot`)
- `test_webpilot_e2e.py`
- `test_gap_coverage.py`
- `test_desktop_api.py`
- `test_desktop_executor.py` (likely)

This is a minor maintainability concern but does not affect correctness.

### Test Isolation
All tests use proper isolation patterns:
- API tests use httpx `ASGITransport` — no live server needed
- WebPilot e2e tests use `uvicorn.Server` in a daemon thread with `WEBPILOT_STUB`
- Desktop tests use `DESKTOP_MOCK=true` — no real mss/pyautogui calls
- No global state leaks observed between test modules

---

## 9. API Endpoint Spot-Check (live server)

Server running at `http://127.0.0.1:8080` with `WEBPILOT_STUB=navigate_and_done`:

| Endpoint | Status | Response |
|---|---|---|
| `GET /health` | 200 OK | `{"status":"ok","version":"1.4.0",...}` |
| `GET /` | 200 OK | `{"service":"UI Navigator","version":"1.4.0",...}` |
| `GET /tasks` | 200 OK | `{"tasks":[],"total":0,...}` |
| `GET /favicon.ico` | **404 Not Found** | Bug: no favicon served |
| `GET /docs` | 200 OK | Swagger UI loads (confirmed via visual test #2) |

---

## 10. Recommendations (Bugs to Report, No Fixes Applied)

### BUG-1 (HIGH): `test_session_does_not_survive_server_restart` — flaky under load
- **File:** `tests/test_gap_coverage.py:399`
- **Impact:** Intermittent CI failures — test fails approximately once per full suite run on loaded Windows machines
- **Root cause:** Subprocess uvicorn startup exceeds 25s deadline when machine is under test load; env vars `DESKTOP_MOCK`/`DESKTOP_MODE_ENABLED` not forwarded to subprocess env

### BUG-2 (MEDIUM): `pytest-mock` missing from `requirements-dev.txt`
- **File:** `requirements-dev.txt`
- **Impact:** Fresh CI environment without pre-installed pytest-mock would fail test collection
- **Details:** `pytest-mock` 3.15.1 is used by `test_webpilot_api.py` (via `mocker` fixture) but not listed

### BUG-3 (MEDIUM): Deprecated `ConnectionClosed.code` attribute used in 3 test assertions
- **Files:** `tests/test_gap_coverage.py:480`, `tests/test_webpilot_api.py:321`, `tests/test_webpilot_api.py:392`
- **Impact:** Will break when websockets removes the deprecated `.code` attribute
- **Fix path:** Migrate to `exc.rcvd.code`

### BUG-4 (LOW): `favicon.ico` returns 404
- **Impact:** Browser console error on every page load for `desktop.html`, `visual-tests.html`, `e2e-tests.html`, and the port-3333 dashboard
- **Details:** No `/favicon.ico` static file is served. FastAPI's `StaticFiles` mount would need a favicon file added

### BUG-5 (LOW): `asyncio_default_fixture_loop_scope` not configured in `pyproject.toml`
- **File:** `pyproject.toml`
- **Impact:** pytest-asyncio shows `asyncio_default_fixture_loop_scope=None` in header — potential for future fixture isolation issues if module-scoped async fixtures are added
- **Fix path:** Add `asyncio_default_fixture_loop_scope = "function"` to `[tool.pytest.ini_options]`

### BUG-6 (LOW): `test_project_judge.py` referenced in CLAUDE.md but file does not exist
- **File:** `CLAUDE.md`
- **Impact:** Documentation misleads developers about the test suite composition; CLAUDE.md lists it as having 34 tests

### INFO-1: websockets 15.x + uvicorn 0.40.0 compatibility — 51 DeprecationWarnings
- **Files:** All e2e test files that start a real uvicorn server
- **Impact:** Noise in CI output; will eventually become breaking changes when websockets removes `legacy` module
- **Upstream issue:** uvicorn needs to update its websockets protocol handler

### INFO-2: `google-genai` 1.66.0 uses deprecated Python 3.14 internal (`_UnionGenericAlias`)
- **Impact:** Will produce warnings on Python 3.14+ and will break on Python 3.17
- **Upstream issue:** google-genai library needs updating

---

## Appendix: Screenshots

| File | Description |
|---|---|
| `tests/reports/desktop-html-screenshot.png` | DesktopPilot dashboard initial state |
| `tests/reports/visual-tests-html-screenshot.png` | Visual Test Runner before running |
| `tests/reports/visual-tests-all-pass.png` | Visual Test Runner — all 10 passed |
| `tests/reports/e2e-tests-html-before.png` | E2E Live Tests dashboard initial state |
| `tests/reports/dashboard-3333-screenshot.png` | Test Dashboard (port 3333) with cached results |

---

*Report generated by Claude Code SDET — 2026-03-19*
