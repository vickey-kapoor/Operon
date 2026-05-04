# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
npm run dev        # Vite dev server on http://localhost:5173 (hot-reload)
npm run build      # tsc type-check then Vite production build → dist/
npm run preview    # Serve the dist/ output locally
```

No test runner is configured. Type correctness is the only automated gate — `tsc` runs as part of `npm run build`. Always run `npm run build` to validate changes.

## Architecture

This is the **Operon Command Center UI** — a React 19 + TypeScript frontend that acts as the observer and control surface for the Python agent backend running at `http://127.0.0.1:8080`.

### Data flow

```
Python API (port 8080) ──HTTP──▶ fetch() calls in components
Python WS stream (port 9001) ──WebSocket──▶ useAgentStream hook
                                              │
                                              ▼
                                         agentStore (Zustand 5)
                                              │
                                         React components
```

`useAgentStream` (`src/hooks/useAgentStream.ts`) owns the single WebSocket connection to `ws://127.0.0.1:9001`. It:
- Delivers binary JPEG frames directly to registered canvas callbacks (zero React state overhead) via `registerFrameCallback` — bypassing React for frame rendering
- Parses JSON text frames into typed `AgentEvent` union objects and appends them to `state.events`
- Handles `hitl_required`, `action_intent`, and `element_bounds` as special singleton state fields

`agentStore` (`src/store/agentStore.ts`) is the Zustand 5 global store. `MainLayout` bridges new WebSocket events into the store via `useAgentSync()` and is the only place this bridge runs.

### Layout

`MainLayout` (three `react-resizable-panels` v4 panes):
- **Sidebar** 15% — navigation only; `setActiveNav` drives which panel renders in the center slot
- **Task Intelligence / Settings** 35% — renders `TaskIntelligence` normally; swaps to `SettingsPane` when `activeNav` is `"moat"` or `"settings"`
- **Live Execution** 50% — always visible; contains `LiveMirror` (canvas), `ConfidenceSlider`, and HITL dim overlay

### react-resizable-panels v4 API

The installed version is **v4**, which uses renamed exports. Always use:
- `Group` (not `PanelGroup`), `Panel`, `Separator` (not `PanelResizeHandle`)
- `orientation` prop (not `direction`)

### CSS animations

All keyframes (`operon-ping`, `operon-bar-wave`, `operon-pulse`, `operon-wave-flow`) are injected once into `document.head` by `LiveMirror`'s `ensureKeyframes()` function. Do not define them elsewhere; reference them by name from any component's inline styles.

### Canvas frame delivery

`LiveMirror` renders WebSocket JPEG frames to a `<canvas>` element using `registerFrameCallback`. This bypasses React state — frames do not trigger re-renders. SVG overlays (click pulses, element bounds, action crosshairs) are drawn on a sibling `<svg>` layered above the canvas.

### Confidence-gated autonomy

`ConfidenceSlider` (in `LiveExecution`) sends `{ type: "set_confidence_threshold", threshold: 0.0–1.0 }` over WebSocket. When a step's confidence falls below the threshold, the Python loop pauses the run and emits `hitl_required`. The UI responds by showing the HITL approval card in `TaskIntelligence` and a dim overlay in `LiveExecution`.

The `SettingsPane` also manages rule toggling via `{ type: "set_disabled_rules", rules: string[] }` — the backend's `PolicyRuleEngine` skips rules in this set.

### WebSocket control messages (outbound)

All WS sends go through `sendControl(msg: object)` from `useAgentStream`. Known message types:

| type | payload | effect |
|---|---|---|
| `set_confidence_threshold` | `{ threshold: number }` | Gate LLM actions below this confidence |
| `set_disabled_rules` | `{ rules: string[] }` | Skip named engine primitives in PolicyRuleEngine |
| `override` | `{ hint: string }` | Inject user correction + resume paused run |
| `resume` | — | Resume HITL-paused run |
| `pause` | — | Request pause (best-effort) |
| `snapshot_ax` | — | Request accessibility tree snapshot |
| `inject_input` | `{ x_ratio, y_ratio, input_type, ... }` | Send click/type/scroll to live browser |

### Tauri integration

`LiveExecution` calls `invoke("attach_to_browser", { port })` via `@tauri-apps/api/core` before calling `POST /connect-cdp`. This is the only Tauri IPC call in the codebase. The app runs as a standard web app when Tauri is absent — the `invoke` call will throw, caught by the `try/catch` in `attachBrowser`.

### `RunState` type

Defined in `App.tsx` and re-exported from there. Child components import it from `"../App"`. Do not move or duplicate this type.

### Key conventions

- All inline styles — no CSS files or CSS modules exist in this codebase.
- `agentStore` state for settings (`disabledRules`, `cdpPort`, `sessionMode`) is session-scoped; it resets on page reload.
- `tsconfig.json` targets ES2022 — `Array.at()` and other ES2022 APIs are available.
- The `CommandCenter.tsx` component is present but no longer imported; `MainLayout` replaced it.
