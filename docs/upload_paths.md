# Upload Action Paths

Operon provides two action types for file upload flows. They differ in how the file
selection is performed.

---

## `upload_file` — Playwright browser-native chooser

| Property | Value |
|---|---|
| Action type | `upload_file` |
| Executor | Browser (`NativeBrowserExecutor`) |
| Mechanism | `page.expect_file_chooser()` + `set_files()` |
| OS picker shown | No |

Playwright intercepts the file-chooser event before the OS dialog appears and
injects the file path directly. The OS file picker is bypassed entirely. This is
reliable, fast, and headless-safe, but it only works when the upload control is a
standard `<input type="file">` element reachable by Playwright.

**Payload**: `text` = absolute path of the file to upload. Optionally `selector` or
`x,y` to locate the control.

---

## `upload_file_native` — True OS picker path

| Property | Value |
|---|---|
| Action type | `upload_file_native` |
| Starting executor | Browser (`NativeBrowserExecutor`) |
| Continuing executor | Desktop (`DesktopExecutor`) |
| Mechanism | Browser click → OS picker appears → desktop interaction |
| OS picker shown | Yes |

This is a cross-environment action. The flow is:

1. **Browser step** — `NativeBrowserExecutor` clicks the upload control (by selector,
   coordinate, or `target_element_id`). This triggers the OS file picker to open.
   The executor returns immediately with `detail="native_picker_triggered"`.
2. **Desktop step** — `DesktopExecutor` interacts with the native OS file picker using
   `TYPE` (to enter the path) and `PRESS_KEY` (to press Enter or click Open). No new
   action types are needed; the existing primitives are sufficient.
3. **Browser verify** — After the picker closes, the agent re-perceives the browser
   page and confirms the file appears as attached or queued by the upload control.

### Routing

`UPLOAD_FILE_NATIVE` belongs to `BROWSER_ACTIONS` and `CROSS_ENVIRONMENT_ACTIONS` in
`core/router.py`. It is **not** in `DESKTOP_ACTIONS` — the desktop interaction with
the picker is handled by ordinary `TYPE` / `PRESS_KEY` actions, not a separate entry
point.

```
BROWSER_ACTIONS    ✓  upload_file_native
DESKTOP_ACTIONS    ✗  (not present — desktop sub-step uses TYPE + PRESS_KEY)
CROSS_ENVIRONMENT  ✓  upload_file_native
```

### Failure types

| FailureType | When | Adaptation strategy |
|---|---|---|
| `PICKER_NOT_DETECTED` | OS picker never appeared after the click | `wait_then_retry` |
| `FILE_NOT_REFLECTED` | Browser page never showed the file as attached | `reperceive_and_replan` |

### State tracking

`AgentRuntimeState.file_picker_active: bool` is set to `True` by the orchestrator
(`UnifiedOrchestrator.process_step`) when:

- The action was `UPLOAD_FILE_NATIVE`, and
- `detect_file_picker(perception)` finds picker signals in `context_label` or `notes`.

---

## When to use each

| Scenario | Use |
|---|---|
| Standard `<input type="file">` element, headless or not | `upload_file` |
| Custom upload button that triggers the OS picker, or non-standard control | `upload_file_native` |
| Upload control hidden or bypassed by JS (no real picker) | `upload_file` |
| Need to verify file name as typed into the OS dialog title bar | `upload_file_native` |

---

## Log / artifact trail

Both action types write `execution_trace.json` and screenshots to
`runs/<run_id>/step_N/`. `upload_file_native` additionally sets
`file_picker_active=True` in the runtime state JSON when the picker was detected.
