# Recovery Ladder

`RuleBasedRecoveryManager` (`src/agent/recovery.py`) maps every failed step to one of six recovery strategies. The ladder escalates based on `state.retry_counts` for the current failure cluster.

---

## Special Cases (bypass the ladder)

| Condition | Strategy | Notes |
|---|---|---|
| `stop_condition_met=True` | `STOP` (terminal) | Goal confirmed — do not retry |
| `VerificationStatus.SUCCESS` | `ADVANCE` | Move to next step |
| `VerificationStatus.PROGRESSING_STABLE` | `ADVANCE` | Gemini confirmed micro-reaction; advance immediately |
| `FailureCategory.EXECUTION_NO_PROGRESS` | `STOP` (terminal) | Loop detected without any screen change |

---

## Escalation Ladder

Keyed on `state.retry_counts` (incremented per-failure, reset on `context_reset`):

| Attempt | Strategy | Wait | What happens |
|---|---|---|---|
| 1 | `RETRY_SAME_STEP` | 0 ms | Re-capture and re-try the exact same action |
| 2 | `RETRY_DIFFERENT_TACTIC` | 0 ms | Same goal, different approach — LLM free to choose a new action |
| 3 | `CONTEXT_RESET` | 1000 ms | `executor.context_reset()`: Escape×2, click center, Ctrl+Home. Then re-perceive. |
| 4 | `SESSION_RESET` | 1500 ms | `executor.session_reset(start_url)`: close tab, fresh session, navigate. |
| ≥ 5 | `STOP` | — | `MAX_RECOVERY_ATTEMPTS = 5`. Run terminates with `FAILURE`. |

---

## Recovery Strategies in Detail

### `RETRY_SAME_STEP`
The loop re-runs from **CAPTURE** with the same `active_subgoal`. No state changes. Used when the executor failed transiently (network glitch, timing issue).

### `RETRY_DIFFERENT_TACTIC`
`active_subgoal` is preserved but `PolicyCoordinator` is allowed to choose a completely different action. Episode replay is suppressed so the LLM gets a fresh look. Used when the chosen action was wrong, not the execution.

### `CONTEXT_RESET`
Calls `executor.context_reset()`:
- **Desktop**: Escape × 2 → click body center → Ctrl+Home (scrolls to top)
- **Browser**: clear cookies in context → reset zoom → focus window

Followed by 1000 ms wait and a fresh CAPTURE. Used when a modal, overlay, or focus state has stuck the agent.

### `SESSION_RESET`
Calls `executor.session_reset(start_url)`:
- **Desktop**: context_reset + Alt+Tab cycle to restore known window
- **Browser**: close current tab → open new tab → navigate to `state.start_url`

Followed by 1500 ms wait. Used when the page is in an unrecoverable state and a full fresh start is needed.

### `STOP`
Run terminates. `stop_reason` is set to `MAX_RECOVERY_ATTEMPTS_EXCEEDED`. `PostRunReflector` still runs — it writes failure pattern `MemoryRecord` entries so future runs can avoid the same path.

---

## Benchmark Integrity Guard

`validate_benchmark_integrity()` runs before any `ADVANCE` decision:
- Blocks unverified success claims (verification status ≠ SUCCESS but stop_condition_met=True)
- Blocks ADVANCE past a stop boundary without terminal confirmation

Violations are logged as `CoordDriftWarning` and downgraded to `RETRY_DIFFERENT_TACTIC`.

---

## Failure Categories → Recovery Strategy

The recovery manager maps `FailureCategory` from `ExecutedAction.failure_category` to a default strategy before applying the ladder:

| Failure category | Default strategy |
|---|---|
| `PERCEPTION_LOW_QUALITY` | `RETRY_SAME_STEP` |
| `EXECUTION_TARGET_NOT_FOUND` | `RETRY_DIFFERENT_TACTIC` |
| `EXECUTION_ELEMENT_NOT_INTERACTABLE` | `RETRY_DIFFERENT_TACTIC` |
| `EXECUTION_TIMEOUT` | `CONTEXT_RESET` |
| `VERIFICATION_UNCERTAIN` | `RETRY_DIFFERENT_TACTIC` |
| `SELECTOR_RERESOLUTION_FAILED` | `RETRY_DIFFERENT_TACTIC` |
| `SELECTOR_RERESOLUTION_AMBIGUOUS` | `RETRY_DIFFERENT_TACTIC` |
| `PICKER_NOT_DETECTED` | `WAIT_AND_RETRY` (orchestrator: `wait_then_retry`) |
| `FILE_NOT_REFLECTED` | `RETRY_DIFFERENT_TACTIC` (orchestrator: `reperceive_and_replan`) |
| `REPEATED_ACTION_WITHOUT_PROGRESS` | `CONTEXT_RESET` |
| `REPEATED_FAILURE_LOOP` | `SESSION_RESET` |

Full `FailureCategory` enum (40+ values) is in `src/models/common.py`.
