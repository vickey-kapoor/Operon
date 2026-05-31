# Recovery Ladder
_Last refreshed: 2026-05-15_

`RuleBasedRecoveryManager` in `src/operon/agent/policy/recovery.py` maps failed or uncertain steps to recovery decisions. The ladder is keyed by a retry cluster built from subgoal, target, and failure signal.

## Bypass Cases

| Condition | Decision |
|---|---|
| `verification.stop_condition_met` | `STOP` |
| `FailureCategory.EXECUTION_NO_PROGRESS` | terminal `STOP` |
| `VerificationStatus.SUCCESS` with expected outcome met | `ADVANCE` |
| `VerificationStatus.PROGRESSING_STABLE` | `ADVANCE` |
| TYPE failed because target was missing/not editable | `WAIT_AND_RETRY` with focus-oriented subgoal |

## Escalation

| Attempt | Strategy | Wait |
|---|---|---|
| 1 | `RETRY_SAME_STEP` | optional |
| 2 | `RETRY_DIFFERENT_TACTIC` | optional |
| 3 | `CONTEXT_RESET` | 1000 ms |
| 4 | `SESSION_RESET` | 1500 ms |
| 5+ | terminal `STOP` | none |

The stop reason for exhausted recovery is `RETRY_LIMIT_REACHED`.

## Integrity Guard

`validate_benchmark_integrity()` prevents recovery from claiming terminal success unless verification actually returned `SUCCESS`. It also prevents `ADVANCE` when the verifier set a stop boundary without success.

## Executor Reset Semantics

The actual reset behavior is implemented by each executor:

- Desktop executor: desktop-specific context/session reset behavior.
- Browser executor: browser-specific context/session reset behavior.

The recovery manager only chooses the strategy; the loop applies it.
