import { useRef, useState, useCallback, useEffect, type Dispatch, type SetStateAction } from "react";
import type { RunState } from "../App";
import { TaskIntelligence } from "./TaskIntelligence";
import { LiveExecution } from "./LiveExecution";
import { useAgentStream } from "../hooks/useAgentStream";
import type { StepEvent } from "../hooks/useAgentStream";

type Props = {
  runState: RunState;
  onRunChange: Dispatch<SetStateAction<RunState>>;
};

const DIVIDER_PX = 4;
const MIN_LEFT_PX = 280;
const MIN_RIGHT_PX = 400;

export function CommandCenter({ runState, onRunChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [leftWidth, setLeftWidth] = useState(380);
  const dragging = useRef(false);

  const stream = useAgentStream();

  // Sync run events into parent state
  useEffect(() => {
    const latest = stream.events.at(-1);
    if (!latest) return;
    if (latest.type === "run_started" || latest.type === "run_completed") {
      onRunChange((prev) => ({ ...prev, runId: latest.run_id, status: latest.status }));
    }
    if (latest.type === "step") {
      onRunChange((prev) => ({ ...prev, status: latest.status }));
    }
  }, [stream.events, onRunChange]);

  const steps = stream.events.filter((e) => e.type === "step") as StepEvent[];
  const latestConfidence = stream.latestActionIntent?.confidence ?? steps.at(-1)?.confidence ?? 0;
  const hitlActive = runState.status === "waiting_for_user";

  const onDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    const startX = e.clientX;
    const startWidth = leftWidth;

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const totalWidth = containerRef.current.offsetWidth - DIVIDER_PX;
      const next = Math.max(
        MIN_LEFT_PX,
        Math.min(totalWidth - MIN_RIGHT_PX, startWidth + ev.clientX - startX),
      );
      setLeftWidth(next);
    };

    const onUp = () => {
      dragging.current = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [leftWidth]);

  return (
    <div
      ref={containerRef}
      style={{ display: "flex", height: "100%", overflow: "hidden", userSelect: dragging.current ? "none" : "auto" }}
    >
      {/* Left pane — Task Intelligence */}
      <div style={{ width: leftWidth, flexShrink: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <TaskIntelligence
          events={stream.events}
          connected={stream.connected}
          runState={runState}
          onRunChange={onRunChange}
          sendControl={stream.sendControl}
          pendingDecision={stream.pendingDecision}
        />
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={onDividerMouseDown}
        style={{
          width: DIVIDER_PX,
          flexShrink: 0,
          background: "#e5e7eb",
          cursor: "col-resize",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#d1d5db")}
        onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = "#e5e7eb")}
      />

      {/* Right pane — Live Execution */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        <LiveExecution
          registerFrameCallback={stream.registerFrameCallback}
          actionIntent={stream.latestActionIntent}
          runState={runState}
          sendControl={stream.sendControl}
          latestConfidence={latestConfidence}
          hitlActive={hitlActive}
        />
      </div>
    </div>
  );
}
