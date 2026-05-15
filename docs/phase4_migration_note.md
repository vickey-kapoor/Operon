# Phase 4 Migration Note
_Historical note, refreshed 2026-05-15_

This document previously described an in-progress migration toward a separate unified runtime/orchestrator layer. The current repository does not include a tracked `src/runtime/` package.

## Current Status

The active runtime remains:

```text
src.api.routes
  -> src.agent.loop.AgentLoop
  -> src.executor browser/desktop executor
```

The pieces that still exist from the contract work are:

- `src/core/contracts/`
- `src/core/router.py`
- `src/executor/browser_adapter.py`
- `src/executor/desktop_adapter.py`

These are used for typed contracts, route validation, executor adapters, and tests. They are not a separate orchestrator path.

## Guidance

Do not describe a unified orchestrator or runtime state package as active unless those modules are restored. The source of truth for active behavior is `src/agent/loop.py` and `src/api/routes.py`.
