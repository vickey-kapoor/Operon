# Upload Action Paths
_Last refreshed: 2026-05-15_

Operon has two upload-related browser actions.

## `upload_file`

| Property | Value |
|---|---|
| Action type | `upload_file` |
| Executor | `NativeBrowserExecutor` |
| Mechanism | Playwright file chooser interception |
| Headless-safe | Yes |

This path uses Playwright's file chooser support. It bypasses the native OS picker and sets the selected file directly.

Payload requirements:
- `text`: absolute file path.
- target location: selector or visual coordinates, depending on executor path.

## `upload_file_native`

| Property | Value |
|---|---|
| Action type | `upload_file_native` |
| Executor | `NativeBrowserExecutor` with OS picker macro |
| Mechanism | click upload control, wait for native picker, type path, press Enter |
| Headless-safe | No |

This path is for custom upload controls that open the native OS picker. The current implementation lives in:

- `src/operon/executor/browser_native.py`
- `src/operon/executor/os_picker_macro.py`

The browser executor clicks the visual target, then delegates picker handling to the OS picker macro. It does not depend on a separate runtime/orchestrator state package.

## Routing

`upload_file_native` is allowed for browser actions and treated as cross-environment in `src/operon/core/router.py`. It is not a standalone desktop action; desktop input primitives are used internally by the picker macro.

## Failure Signals

Common failure categories:

- `PICKER_NOT_DETECTED`: the OS picker did not appear.
- `FILE_NOT_REFLECTED`: the chosen file did not appear attached or queued.
- `EXECUTION_ERROR`: generic execution failure, including headed-mode requirements.

## Artifacts

Upload steps write normal run artifacts under:

```text
runs/<run_id>/step_N/
```

Look for `execution_trace.json`, before/after screenshots, and the step log entry in `run.jsonl`.
