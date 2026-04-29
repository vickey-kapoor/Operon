# Operon

**Operon is a vision-only computer use agent — no DOM, no selectors. It sees the screen like a human does and acts on it.**

Closed loop: `capture → perceive → decide → execute → verify → recover`. Desktop (pyautogui) and browser (Playwright + Gemini Computer Use) share one loop.

> 🚧 Early stage — actively building in public. Failures included.

---

## Demo

<!-- Add a GIF or screenshot here once available -->
*Demo coming soon.*

---

## Why Vision-Only?

Most automation frameworks anchor to the DOM — CSS selectors, XPath, element IDs. Operon deliberately doesn't.

- **Selectors are brittle.** A class name change, a UI redesign, a framework migration — your automation breaks. You're not building an agent, you're building a dependency on someone else's HTML decisions.
- **The DOM is a human construct. AI shouldn't need it.** Humans don't read source code to use an interface — they look at it. Agents that generalize need to see what users see.
- **Vision-only works everywhere.** Web apps, desktop apps, internal tools, legacy systems with no clean DOM. If a human can see it, Operon can use it.

---

## HLD

```text
FastAPI  →  AgentLoop  →  Capture   (mss / Playwright screenshot)
                       →  Perceive  (Gemini → ScreenPerception)
                       →  Decide    PolicyCoordinator
                                      ├── PolicyRuleEngine  (deterministic)
                                      └── GeminiPolicyService (LLM fallback)
                       →  Re-resolve DeterministicTargetSelector  (stale target)
                       →  Execute   DesktopExecutor | NativeBrowserExecutor
                       →  Verify    DeterministicVerifier + VideoVerifier
                       →  Recover   RuleBasedRecoveryManager
                       →  Reflect   PostRunReflector  (on terminal)
                       →  Persist   FileBackedRunStore + MemoryStore (+ Episodes)

Unified Contract Layer  (observer on top of the loop)
    LegacyOperonContractAdapter → PerceptionOutput / PlannerOutput / ActorOutput / CriticOutput
    UnifiedOrchestrator  →  src/core/router.py  →  AgentRuntimeState
```

- Vision only — no DOM / a11y tree.
- Desktop and browser share verifier, recovery, persistence, memory.
- Rules first, LLM fallback (`PolicyRuleEngine` → `GeminiPolicyService`).
- Hardened targets: `DeterministicTargetSelector.reresolve()` re-binds to the original `TargetIntent` if the target moves.
- Video fallback verify: on no-change, record 3s and ask Gemini for temporal evidence.
- Self-improving: `PostRunReflector` writes `MemoryHint`s and compresses successful runs into reusable `Episode`s.
- Unified contracts wrap every step in typed perception/planner/actor/critic bundles.

---

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
playwright install chromium
```

Put `GOOGLE_API_KEY` in `.env`.

## Run

```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8080
```

- `/` — Pilot UI
- `/console` — Task Console
- `POST /run-task`, `/step`, `/resume`, `/stop`, `/cleanup`
- `GET /run/{id}`, `/observer/api/runs`, `/observer/api/run/{id}`

## Env

- `OPERON_BROWSER_BACKEND=computer_use` (primary) / `json` (fallback)
- `OPERON_DESKTOP_BACKEND=json`
- `BROWSER_HEADLESS=true`
- `OPERON_TRACE=1` — cyan `[TRACE]` lines from the loop

## Tests

```powershell
python -m pytest tests -q
ruff check src tests
```

## Layout

- `src/agent/` — loop, policy, selectors, verifier, recovery, reflector
- `src/executor/` — desktop + native browser executors
- `src/api/` — FastAPI + Pilot/Console UIs
- `src/store/` — run store, memory, episodes
- `src/models/` — Pydantic schemas
- `src/core/`, `src/runtime/` — unified contract layer (observer on top of the loop)
- `src/executor/browser_adapter.py`, `src/executor/desktop_adapter.py` — thin contract-to-legacy action adapters
- `prompts/`, `docs/`, `examples/contracts/`
- Artifacts: `runs/<run_id>/`, `.browser-artifacts/`

See `CODEBASE_OVERVIEW.md` and `CLAUDE.md` for details.
