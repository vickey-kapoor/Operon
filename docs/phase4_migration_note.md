# Phase 4 Migration Note

## Old Modules Still Used

- `src/api/routes.py`
- `src/agent/loop.py`
- `src/agent/capture.py`
- existing perception backends under `src/agent/`
- existing policy service and `src/agent/policy_coordinator.py`
- existing verifier and recovery services
- existing browser and desktop executors under `src/executor/`
- existing run store and logging

## Old Modules Now Bypassed or Guarded by the Unified Path

- legacy direct executor invocation in `src/agent/loop.py` is now routed through thin adapters when the unified path is active
- legacy per-execution retry hardening in `src/agent/loop.py` is bypassed on the unified path so Phase 3 deterministic adaptation owns retries
- legacy implicit routing decisions are now validated by `core/router.py`

## Old Modules Safe To Delete Later

- duplicated legacy retry-routing helpers in `src/agent/loop.py` once the unified path fully replaces the old hardening path
- any standalone routing helpers that duplicate `core/router.py`
- any coordinator logic that exists only to decide browser vs desktop executor selection outside the unified orchestrator

## Notes

The current migration is intentionally minimal:

- old services still do the actual perception, planning, execution, and verification work
- the unified contract/state/orchestrator path now acts as the shared control-plane layer
- executor behavior remains thin and inspectable
