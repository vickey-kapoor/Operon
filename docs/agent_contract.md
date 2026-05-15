# Unified Agent Contract Notes
_Historical design note, refreshed 2026-05-15_

This document describes the contract-layer intent behind `src/core/contracts/` and `src/core/router.py`.

Current status:
- `src/core/contracts/` exists.
- `src/core/router.py` exists.
- `src/executor/browser_adapter.py` and `src/executor/desktop_adapter.py` exist.
- There is no tracked `src/runtime/` package in the current tree.
- `AgentLoop` remains the primary control loop.

## Current Use

The contract layer is used for typed action compatibility and executor adapter tests. It is not an independent orchestrator and does not replace `AgentLoop`.

## Contracts

The current contract files are:

- `src/core/contracts/perception.py`
- `src/core/contracts/planner.py`
- `src/core/contracts/actor.py`
- `src/core/contracts/critic.py`

The router validates environment/action compatibility in `src/core/router.py`.

## Product Boundary

The product runtime truth is in:
- `src/api/routes.py`
- `src/agent/loop.py`
- `src/executor/`
- `src/models/`

Do not treat older references to `UnifiedOrchestrator`, `AgentRuntimeState`, or `LegacyOperonContractAdapter` as current implementation claims unless those modules are restored.
