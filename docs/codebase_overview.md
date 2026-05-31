# Operon Codebase Overview
_Last refreshed: 2026-05-15_

## Core Concept

Operon is a vision-first computer-use engine for browser and desktop automation. It uses screenshots and structured visual perception to choose actions, execute them, verify progress, and recover from failures.

## Main Flow

```text
request
  -> operon.api.routes
  -> AgentLoop
  -> capture
  -> perception
  -> policy
  -> execution
  -> verification
  -> recovery
  -> persistence
```

## Key Files

| Area | Files | Purpose |
|---|---|---|
| API | `src/operon/api/server.py`, `src/operon/api/routes.py`, `src/operon/api/ws_stream.py`, `src/operon/api/observer.py` | FastAPI app, run routes, websocket stream, run inspection |
| Loop | `src/operon/agent/loop.py` | Core step orchestration |
| Backends | `src/operon/agent/browser_computer_use.py`, `src/operon/agent/browser_json.py`, `src/operon/agent/combined.py`, `src/operon/agent/fallback_backend.py` | Perception/policy backend selection |
| Policy | `src/operon/agent/policy_coordinator.py`, `src/operon/agent/policy_rules.py`, `src/operon/agent/policy.py`, `src/operon/agent/anthropic_policy.py` | Rules, LLM fallback, post-LLM guards |
| Execution | `src/operon/executor/browser_native.py`, `src/operon/executor/desktop.py` | Browser and desktop execution |
| Verification | `src/operon/agent/verifier.py`, `src/operon/models/verification.py` | Deterministic verification and status model |
| Recovery | `src/operon/agent/recovery.py` | Recovery ladder and integrity checks |
| Targeting | `src/operon/agent/selector.py`, `src/operon/agent/retry_hardening.py`, `src/operon/models/selector.py` | Target selection and re-resolution |
| Browser observation | `src/operon/browser/manager.py` | CDP attach, screencast, input injection |
| Models | `src/operon/models/` | Pydantic schemas |
| Store | `src/operon/store/` | Run store, memory store, replay, cleanup |

## Active API

The current FastAPI app exposes run, desktop, observer, health, cleanup, pause/stop, and websocket endpoints. It does not serve a bundled frontend.

## Runtime Backends

Browser:
- default: Gemini Computer Use
- optional: JSON fallback

Desktop:
- default: Gemini JSON combined perception+policy

Policy providers:
- Gemini by default
- Anthropic optional for planner/verifier paths where configured

## State And Artifacts

Run state is stored in `runs/<run_id>/state.json`. Step artifacts are stored in `runs/<run_id>/step_N/`. Logs are appended to `runs/<run_id>/run.jsonl`.

Memory support exists in `src/operon/store/memory.py` and `src/operon/models/episode.py`. The current tree does not include a post-run reflector implementation that automatically generates reflection records.

## Supporting Areas

- `src/operon/core/contracts` and `src/operon/core/router.py`: contract models and environment/action validation used by executor adapters and tests.
