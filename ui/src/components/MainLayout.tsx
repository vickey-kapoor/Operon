/**
 * MainLayout — Global three-pane layout using react-resizable-panels v4.
 *
 * Proportions:  Sidebar 15% │ Task Intelligence 35% │ Live Execution 50%
 *
 * Responsibilities:
 *  - Owns the single useAgentStream() WebSocket connection for the session.
 *  - Bridges incoming AGENT_UPDATE events → useAgentStore (Zustand 5).
 *  - Passes derived props to child panels; children stay decoupled from the WS.
 */

import { useCallback, useEffect, useRef } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { useAgentStore } from "../store/agentStore";
import { useAgentStream } from "../hooks/useAgentStream";
import { Sidebar } from "./Sidebar";
import { TaskIntelligence } from "./TaskIntelligence";
import { LiveExecution } from "./LiveExecution";
import { SettingsPane } from "./SettingsPane";
import type { RunState } from "../App";
import type { HitlRequiredEvent, RunEvent, StepEvent } from "../hooks/useAgentStream";

// ── AGENT_UPDATE bridge ───────────────────────────────────────────────────────

/**
 * Syncs new WebSocket events into the Zustand store.
 * Runs inside MainLayout so the bridge exists for the full session lifetime.
 */
function useAgentSync(
  events: ReturnType<typeof useAgentStream>["events"],
  latestActionIntent: ReturnType<typeof useAgentStream>["latestActionIntent"],
  pendingDecision: ReturnType<typeof useAgentStream>["pendingDecision"],
) {
  const setRunStarted = useAgentStore((s) => s.setRunStarted);
  const updateFromStep = useAgentStore((s) => s.updateFromStep);
  const setRunCompleted = useAgentStore((s) => s.setRunCompleted);
  const updateFromHitl = useAgentStore((s) => s.updateFromHitl);
  const setConfidenceScore = useAgentStore((s) => s.setConfidenceScore);
  const setPendingDecision = useAgentStore((s) => s.setPendingDecision);

  // Track how many events we have already processed to avoid reprocessing on re-render.
  const processedRef = useRef(0);

  // AGENT_UPDATE events → store: run_started, step, run_completed.
  useEffect(() => {
    const newEvents = events.slice(processedRef.current);
    processedRef.current = events.length;

    for (const event of newEvents) {
      if (event.type === "run_started") {
        const ev = event as RunEvent;
        setRunStarted(ev.run_id, ev.intent ?? "");
      } else if (event.type === "step") {
        updateFromStep(event as StepEvent);
      } else if (event.type === "run_completed") {
        setRunCompleted((event as RunEvent).status);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  // Confidence — action_intent events carry higher-frequency confidence readings.
  useEffect(() => {
    if (latestActionIntent) {
      setConfidenceScore(latestActionIntent.confidence);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestActionIntent]);

  // HITL — sync hitl_required event into the store.
  useEffect(() => {
    if (pendingDecision) {
      updateFromHitl(pendingDecision as HitlRequiredEvent);
    } else {
      setPendingDecision(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingDecision]);
}

// ── Layout ────────────────────────────────────────────────────────────────────

export function MainLayout() {
  const stream = useAgentStream();

  // Subscribe to the store slices this layout cares about.
  const runId       = useAgentStore((s) => s.runId);
  const currentTask = useAgentStore((s) => s.currentTask);
  const runStatus   = useAgentStore((s) => s.runStatus);
  const confidence  = useAgentStore((s) => s.confidenceScore);
  const hitlActive  = useAgentStore((s) => s.hitlActive);
  const activeNav   = useAgentStore((s) => s.activeNav);
  const setActiveNav = useAgentStore((s) => s.setActiveNav);
  const applyRunState = useAgentStore((s) => s.applyRunState);

  // Start the AGENT_UPDATE bridge.
  useAgentSync(stream.events, stream.latestActionIntent, stream.pendingDecision);

  // RunState shape expected by child components.
  const runState: RunState = { runId, intent: currentTask, status: runStatus };

  // Child components call this when they change run state directly (e.g. POST /resume).
  const handleRunChange = useCallback(
    (s: RunState) => applyRunState(s),
    [applyRunState],
  );

  return (
    <Group
      orientation="horizontal"
      style={{ height: "100vh", width: "100vw" }}
    >
      {/* ── Sidebar — 15% ── */}
      <Panel
        id="sidebar"
        defaultSize={15}
        minSize={10}
        maxSize={22}
        style={{ overflow: "hidden" }}
      >
        <Sidebar active={activeNav} onNav={setActiveNav} />
      </Panel>

      <Separator
        id="sep-sidebar"
        style={{ width: 1, background: "#252840", cursor: "col-resize", flexShrink: 0 }}
      />

      {/* ── Task Intelligence / Settings — 35% ── */}
      <Panel
        id="task"
        defaultSize={35}
        minSize={22}
        style={{ overflow: "hidden" }}
      >
        {activeNav === "moat" || activeNav === "settings" ? (
          <SettingsPane sendControl={stream.sendControl} />
        ) : (
          <TaskIntelligence
            events={stream.events}
            connected={stream.connected}
            runState={runState}
            onRunChange={handleRunChange}
            sendControl={stream.sendControl}
            pendingDecision={stream.pendingDecision}
          />
        )}
      </Panel>

      <Separator
        id="sep-task"
        style={{ width: 4, background: "#e5e7eb", cursor: "col-resize", flexShrink: 0 }}
      />

      {/* ── Live Execution — 50% ── */}
      <Panel
        id="execution"
        defaultSize={50}
        minSize={30}
        style={{ overflow: "hidden" }}
      >
        <LiveExecution
          registerFrameCallback={stream.registerFrameCallback}
          actionIntent={stream.latestActionIntent}
          runState={runState}
          sendControl={stream.sendControl}
          latestConfidence={confidence}
          hitlActive={hitlActive}
        />
      </Panel>
    </Group>
  );
}
