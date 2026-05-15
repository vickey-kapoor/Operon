# Operon Codebase Overview
_Last refreshed: 2026-05-15_

## Core Concept

Operon is a vision-first computer-use engine for browser and desktop automation. It uses screenshots and structured visual perception to choose actions, execute them, verify progress, and recover from failures.

## Main Flow

```text
request
  -> src.api.routes
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
| API | `src/api/server.py`, `src/api/routes.py`, `src/api/ws_stream.py`, `src/api/observer.py` | FastAPI app, run routes, websocket stream, run inspection |
| Loop | `src/agent/loop.py` | Core step orchestration |
| Backends | `src/agent/browser_computer_use.py`, `src/agent/browser_json.py`, `src/agent/combined.py`, `src/agent/fallback_backend.py` | Perception/policy backend selection |
| Policy | `src/agent/policy_coordinator.py`, `src/agent/policy_rules.py`, `src/agent/policy.py`, `src/agent/anthropic_policy.py` | Rules, LLM fallback, post-LLM guards |
| Execution | `src/executor/browser_native.py`, `src/executor/desktop.py`, `src/executor/browserbase_native.py` | Browser, desktop, and Browserbase execution |
| Verification | `src/agent/verifier.py`, `src/models/verification.py` | Deterministic verification and status model |
| Recovery | `src/agent/recovery.py` | Recovery ladder and integrity checks |
| Targeting | `src/agent/selector.py`, `src/agent/retry_hardening.py`, `src/models/selector.py` | Target selection and re-resolution |
| Browser observation | `src/browser/manager.py` | CDP attach, screencast, input injection |
| Models | `src/models/` | Pydantic schemas |
| Store | `src/store/` | Run store, memory store, replay, summaries, cleanup |
| Optional UI | `ui/`, `src-tauri/` | React/Tauri command center surface |

## Active API

The current FastAPI app exposes run, desktop, observer, health, cleanup, pause/stop, and websocket endpoints. It does not serve the optional React UI and does not expose benchmark or command-center HTML routes.

## Runtime Backends

Browser:
- default: Gemini Computer Use
- optional: JSON fallback
- optional: Browserbase executor

Desktop:
- default: Gemini JSON combined perception+policy

Policy providers:
- Gemini by default
- Anthropic optional for planner/verifier paths where configured

## State And Artifacts

Run state is stored in `runs/<run_id>/state.json`. Step artifacts are stored in `runs/<run_id>/step_N/`. Logs are appended to `runs/<run_id>/run.jsonl`.

Memory support exists in `src/store/memory.py` and `src/models/episode.py`. The current tree does not include a post-run reflector implementation that automatically generates reflection records.

## Optional Or Experimental Areas

- `src/core/contracts` and `src/core/router.py`: contract models and environment/action validation used by executor adapters and tests.
- `benchmarks/`: dataset files only.
- `src/agent/benchmark.py`: local benchmark runner.
- `docs/substack_drafts/`: historical writeups.
- `docs/claude_task.md`: historical task backlog and audit notes.
