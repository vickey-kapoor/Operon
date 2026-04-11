# Unified Agent Contract (Phase 1)

## Scope

Phase 1 defines the contract layer only for a shared browser + desktop agent.

This phase includes:

- strict JSON contracts
- typed schema validation
- executor routing rules
- example flows
- stub executors

This phase does not include:

- Playwright behavior
- desktop automation behavior
- polling coordination
- multi-agent logic
- business logic inside executors

## Design Rules

- One shared agent core handles both environments.
- The planner outputs subgoals and action intent, not coordinates.
- The actor chooses an executor explicitly: `browser` or `desktop`.
- The critic classifies failure types explicitly.
- Every contract is strict JSON with no extra fields.
- Executors are adapters only. They do not decide policy.

## File Layout

- `docs/agent_contract.md`
- `core/contracts/perception.py`
- `core/contracts/planner.py`
- `core/contracts/actor.py`
- `core/contracts/critic.py`
- `core/router.py`
- `runtime/state.py`
- `runtime/orchestrator.py`
- `executors/browser_executor.py`
- `executors/desktop_executor.py`
- `examples/contracts/`
- `tests/test_contracts.py`

## Shared Concepts

### Environments

- `browser`
- `desktop`

### Action Types

The shared agent core may emit only these action types:

- `click`
- `double_click`
- `type_text`
- `press_hotkey`
- `scroll`
- `wait`
- `launch_app`
- `navigate`

### Failure Types

The actor and critic may classify only these failure types:

- `target_not_found`
- `wrong_window_active`
- `dialog_not_opened`
- `text_not_entered`
- `timing_issue`
- `ambiguous_perception`
- `ui_changed`

## Routing Rules

Routing is deterministic and lives in the shared core.

### Environment routing

- If `environment` is `browser`, the routed executor must be `browser`.
- If `environment` is `desktop`, the routed executor must be `desktop`.

### Action compatibility

#### Browser executor

Allowed:

- `click`
- `double_click`
- `type_text`
- `press_hotkey`
- `scroll`
- `wait`
- `navigate`

Rejected:

- `launch_app`

#### Desktop executor

Allowed:

- `click`
- `double_click`
- `type_text`
- `press_hotkey`
- `scroll`
- `wait`
- `launch_app`

Rejected:

- `navigate`

### Routing invariants

- Planner output must include exactly one environment.
- Planner output must include exactly one action.
- Planner output must include a human-readable `subgoal`.
- Planner action payloads must not contain coordinates.
- Actor output must include exactly one executor choice.
- Actor executor must match the routed environment.
- Critic output must evaluate the same `observation_id`, `plan_id`, and `attempt_id`.

## Contract: PerceptionOutput

Perception describes the current UI state in a way that both browser and desktop flows can use.

```json
{
  "contract_version": "phase1",
  "environment": "browser",
  "observation_id": "obs_browser_001",
  "summary": "The app dashboard is visible and a Settings button appears in the left sidebar.",
  "context_label": "Dashboard",
  "visible_targets": [
    {
      "target_id": "target_settings",
      "role": "button",
      "label": "Settings",
      "text": "Settings",
      "confidence": 0.98
    }
  ],
  "focused_target_id": "target_settings",
  "notes": []
}
```

## Contract: PlannerOutput

Planner chooses a single next step using a subgoal and a symbolic action.

```json
{
  "contract_version": "phase1",
  "environment": "browser",
  "observation_id": "obs_browser_001",
  "plan_id": "plan_browser_001",
  "subgoal": "Open the settings page.",
  "rationale": "The settings button is clearly visible and matches the task.",
  "action": {
    "action_type": "click",
    "target_id": "target_settings",
    "target_label": "Settings"
  },
  "expected_outcome": "The settings screen becomes visible."
}
```

Planner rules:

- must output `subgoal`
- must not output coordinates
- must not choose an executor
- must produce only one action

## Contract: ActorOutput

Actor records the routed execution attempt. It does not decide policy.

```json
{
  "contract_version": "phase1",
  "environment": "browser",
  "observation_id": "obs_browser_001",
  "plan_id": "plan_browser_001",
  "attempt_id": "attempt_browser_001",
  "executor": "browser",
  "action": {
    "action_type": "click",
    "target_id": "target_settings",
    "target_label": "Settings"
  },
  "status": "success",
  "failure_type": null,
  "details": "Stub actor recorded a browser click request."
}
```

Actor rules:

- must choose `browser` or `desktop` explicitly
- must echo the action it attempted
- must not add policy or recovery logic
- if `status` is `failed`, `failure_type` is required

## Contract: CriticOutput

Critic evaluates the actor attempt using explicit failure classification.

```json
{
  "contract_version": "phase1",
  "environment": "browser",
  "observation_id": "obs_browser_001",
  "plan_id": "plan_browser_001",
  "attempt_id": "attempt_browser_001",
  "outcome": "success",
  "failure_type": null,
  "judgment": "The routed browser action is valid for the current subgoal."
}
```

Critic rules:

- must classify failure types explicitly for non-success outcomes
- must remain downstream of actor output
- must not execute recovery itself

## Browser Example Flow

Perception sees a pricing link, planner emits subgoal `Open the pricing page`, actor routes to `browser`, critic evaluates the attempt.

## Desktop Example Flow

Perception sees the desktop, planner emits subgoal `Open Notepad`, actor routes to `desktop`, critic evaluates the attempt.

## Cross-Environment Example Flow

One shared core can handle:

1. a browser step to download an installer
2. a desktop step to launch the installer

The contract shape stays the same. Only `environment`, executor routing, and allowed action compatibility change.
