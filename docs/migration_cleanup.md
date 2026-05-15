# Migration Cleanup Notes
_Historical note, refreshed 2026-05-15_

This file used to audit a migration toward a separate runtime/orchestrator package. The current repository no longer contains a tracked `src/runtime/` package, and `AgentLoop` is still the active control loop.

## Current Active Path

```text
src.api.routes
  -> src.agent.loop.AgentLoop
  -> src.agent.capture.ScreenCaptureService
  -> perception/policy backend
  -> src.agent.policy_coordinator.PolicyCoordinator
  -> src.executor browser or desktop executor
  -> src.agent.verifier.DeterministicVerifierService
  -> src.agent.recovery.RuleBasedRecoveryManager
  -> src.store.run_store.FileBackedRunStore
```

## Current Cleanup Position

The old runtime/orchestrator claims should not be used as current architecture. The remaining relevant pieces are:

- `src/core/contracts/`
- `src/core/router.py`
- `src/executor/browser_adapter.py`
- `src/executor/desktop_adapter.py`

Those pieces support adapter validation and tests. They do not own the runtime.

## Known Stale Historical Claims

Older notes may refer to:
- `PostRunReflector`
- `src/runtime/*`
- benchmark plugin registry packages
- benchmark API routes
- static HTML UI routes

Those are not active implementation claims in the current tree.
