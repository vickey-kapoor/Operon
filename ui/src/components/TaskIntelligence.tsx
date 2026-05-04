import { useState, useRef, useEffect } from "react";
import type { AgentEvent, StepEvent, HitlRequiredEvent } from "../hooks/useAgentStream";
import type { RunState } from "../App";
import { SubgoalTree } from "./SubgoalTree";
import { ReasoningLog } from "./ReasoningLog";

const API = "http://127.0.0.1:8080";

type Props = {
  events: AgentEvent[];
  connected: boolean;
  runState: RunState;
  onRunChange: (s: RunState) => void;
  sendControl: (msg: object) => void;
  pendingDecision: HitlRequiredEvent | null;
};

/** SVG bar-wave that increases animation frequency when the agent is cogitating. */
function ThinkingPulse({ active }: { active: boolean }) {
  const BAR_COUNT = 9;
  const W = 54, H = 20, barW = 3, gap = 3;
  const dur = active ? "0.55s" : "2.4s";
  const fill = active ? "#6366f1" : "#d1d5db";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
        {Array.from({ length: BAR_COUNT }, (_, i) => {
          const baseH = 4 + Math.abs(Math.sin((i / (BAR_COUNT - 1)) * Math.PI)) * 12;
          const y = (H - baseH) / 2;
          const cx = i * (barW + gap) + barW / 2;
          return (
            <rect
              key={i}
              x={i * (barW + gap)}
              y={y}
              width={barW}
              height={baseH}
              rx={1.5}
              fill={fill}
              style={{
                transformOrigin: `${cx}px ${H / 2}px`,
                animation: `operon-bar-wave ${dur} ease-in-out ${(i * (active ? 0.06 : 0.22)).toFixed(2)}s infinite alternate`,
                transition: "fill 0.35s",
              }}
            />
          );
        })}
      </svg>
      <span style={{ fontSize: 10, color: active ? "#6366f1" : "#9ca3af", letterSpacing: "0.05em", fontWeight: 600 }}>
        {active ? "THINKING" : "IDLE"}
      </span>
    </div>
  );
}

export function TaskIntelligence({ events, connected, runState, onRunChange, sendControl, pendingDecision }: Props) {
  const [intent, setIntent] = useState(runState.intent || "");
  const [overrideHint, setOverrideHint] = useState("");
  const [activeTab, setActiveTab] = useState<"subgoals" | "reasoning">("subgoals");
  const scrollRef = useRef<HTMLDivElement>(null);

  const [correctionHint, setCorrectionHint] = useState("");
  const isActive = runState.status === "running";
  const isWaiting = runState.status === "waiting_for_user";
  const steps = events.filter((e): e is StepEvent => e.type === "step");

  const proceedRun = async () => {
    if (!runState.runId) return;
    await fetch(`${API}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runState.runId }),
    });
    onRunChange({ ...runState, status: "running" });
  };

  const sendCorrection = () => {
    const hint = correctionHint.trim();
    if (!hint || !runState.runId) return;
    // Override stores the hint in ws_stream and calls resume_run on the backend.
    sendControl({ type: "override", hint });
    setCorrectionHint("");
    onRunChange({ ...runState, status: "running" });
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events]);

  const startRun = async () => {
    const trimmed = intent.trim();
    if (!trimmed) return;
    const res = await fetch(`${API}/run-task`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intent: trimmed, max_steps: 25 }),
    });
    const data = await res.json();
    onRunChange({ runId: data.run_id, intent: trimmed, status: data.status });
  };

  const stopRun = async () => {
    if (!runState.runId) return;
    await fetch(`${API}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runState.runId }),
    });
  };

  const sendOverride = () => {
    if (!overrideHint.trim()) return;
    sendControl({ type: "override", hint: overrideHint.trim() });
    setOverrideHint("");
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "#f8f9fb" }}>
      {/* Panel header */}
      <div style={{
        padding: "12px 16px 10px",
        background: "#fff", borderBottom: "1px solid #e5e7eb",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: "#374151" }}>
            TASK INTELLIGENCE
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: connected ? "#10b981" : "#ef4444",
            }} />
            <span style={{ fontSize: 10, color: "#9ca3af" }}>
              {connected ? "Bridge connected" : "Reconnecting…"}
            </span>
          </div>
        </div>
        <ThinkingPulse active={isActive} />
      </div>

      {/* Intent input */}
      <div style={{ padding: "12px 14px", background: "#fff", borderBottom: "1px solid #e5e7eb", flexShrink: 0 }}>
        <textarea
          value={intent}
          onChange={(e) => setIntent(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && e.metaKey) startRun(); }}
          placeholder="Describe the task… (⌘↵ to start)"
          rows={3}
          style={{
            width: "100%", resize: "none", outline: "none",
            background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 6,
            padding: "8px 10px", fontSize: 12, color: "#1e2030",
            fontFamily: "inherit", lineHeight: 1.5,
          }}
        />
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <PanelBtn onClick={startRun} disabled={isActive || !intent.trim()} color="#10b981" label="▶ Start" />
          <PanelBtn onClick={stopRun} disabled={!isActive} color="#ef4444" label="■ Stop" />
        </div>

        {/* Override hint */}
        <input
          value={overrideHint}
          onChange={(e) => setOverrideHint(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendOverride()}
          placeholder="Override hint — press Enter to inject"
          style={{
            marginTop: 8, width: "100%", background: "#f9fafb",
            border: "1px solid #e5e7eb", borderRadius: 6,
            padding: "6px 10px", fontSize: 11, color: "#374151",
            outline: "none", fontFamily: "inherit",
          }}
        />
      </div>

      {/* Tab switcher */}
      <div style={{
        display: "flex", background: "#fff", borderBottom: "1px solid #e5e7eb", flexShrink: 0,
      }}>
        {([["subgoals", "Subgoal Tree"], ["reasoning", "Reasoning Log"]] as const).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            style={{
              flex: 1, padding: "8px 0", border: "none", background: "none",
              fontFamily: "inherit", fontSize: 11, cursor: "pointer", fontWeight: activeTab === id ? 700 : 400,
              color: activeTab === id ? "#6366f1" : "#9ca3af",
              borderBottom: activeTab === id ? "2px solid #6366f1" : "2px solid transparent",
              transition: "color 0.15s",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* HITL approval card — shown when the agent is paused waiting for user */}
      {isWaiting && pendingDecision && (
        <div style={{
          margin: "10px 14px 0",
          border: "1px solid #fed7aa",
          borderRadius: 8,
          background: "#fff7ed",
          overflow: "hidden",
          flexShrink: 0,
        }}>
          {/* Header */}
          <div style={{
            padding: "8px 12px",
            background: "#ffedd5",
            borderBottom: "1px solid #fed7aa",
            display: "flex", alignItems: "center", gap: 7,
          }}>
            <span style={{ fontSize: 14 }}>⚠</span>
            <span style={{ fontSize: 11, fontWeight: 700, color: "#9a3412", letterSpacing: "0.05em" }}>
              APPROVAL REQUIRED
            </span>
            {pendingDecision.confidence != null && pendingDecision.threshold != null && (
              <span style={{ marginLeft: "auto", fontSize: 10, color: "#c2410c", fontWeight: 600 }}>
                {Math.round(pendingDecision.confidence * 100)}% conf / {Math.round(pendingDecision.threshold * 100)}% threshold
              </span>
            )}
          </div>

          {/* Pending decision summary */}
          <div style={{ padding: "10px 12px" }}>
            {pendingDecision.subgoal && (
              <div style={{ fontSize: 10, color: "#92400e", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>
                {pendingDecision.subgoal}
              </div>
            )}
            {pendingDecision.pending_action && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                <span style={{
                  fontSize: 10, padding: "2px 7px", borderRadius: 4, fontWeight: 700,
                  background: "#fef3c7", color: "#b45309", letterSpacing: "0.04em",
                }}>
                  {pendingDecision.pending_action.toUpperCase()}
                </span>
                {pendingDecision.rationale && (
                  <span style={{ fontSize: 10, color: "#78350f", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {pendingDecision.rationale}
                  </span>
                )}
              </div>
            )}
            {pendingDecision.message && (
              <div style={{ fontSize: 10, color: "#7c2d12", lineHeight: 1.5, marginBottom: 8 }}>
                {pendingDecision.message}
              </div>
            )}

            {/* Correction hint input */}
            <input
              value={correctionHint}
              onChange={(e) => setCorrectionHint(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendCorrection()}
              placeholder="Correction hint (optional) — press Enter to send"
              style={{
                width: "100%", background: "#fff", border: "1px solid #fed7aa",
                borderRadius: 5, padding: "5px 8px", fontSize: 11,
                color: "#374151", outline: "none", fontFamily: "inherit",
                marginBottom: 8,
              }}
            />

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={proceedRun}
                style={{
                  flex: 1, padding: "6px 0", border: "1px solid #10b98155",
                  borderRadius: 5, background: "#10b98111", color: "#065f46",
                  fontSize: 11, fontWeight: 700, fontFamily: "inherit", cursor: "pointer",
                }}
              >
                ▶ Proceed
              </button>
              <button
                onClick={sendCorrection}
                disabled={!correctionHint.trim()}
                style={{
                  flex: 1, padding: "6px 0",
                  border: `1px solid ${correctionHint.trim() ? "#6366f155" : "#e5e7eb"}`,
                  borderRadius: 5,
                  background: correctionHint.trim() ? "#6366f111" : "#f9fafb",
                  color: correctionHint.trim() ? "#4338ca" : "#d1d5db",
                  fontSize: 11, fontWeight: 700, fontFamily: "inherit",
                  cursor: correctionHint.trim() ? "pointer" : "not-allowed",
                }}
              >
                ↩ Send & Proceed
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Scrollable content */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "14px", minHeight: 0 }}>
        {activeTab === "subgoals" ? (
          <SubgoalTree intent={runState.intent} steps={steps} />
        ) : (
          <ReasoningLog steps={steps} />
        )}
      </div>

      {/* Bottom controls */}
      <div style={{
        padding: "8px 14px", background: "#fff", borderTop: "1px solid #e5e7eb",
        display: "flex", gap: 6, flexShrink: 0,
      }}>
        <PanelBtn onClick={() => sendControl({ type: "pause" })} disabled={!isActive} color="#f59e0b" label="⏸ Pause" />
        <PanelBtn onClick={() => sendControl({ type: "resume" })} disabled={isActive} color="#6366f1" label="⟳ Resume" />
        <PanelBtn onClick={() => sendControl({ type: "snapshot_ax" })} disabled={false} color="#6b7280" label="AX" />
      </div>
    </div>
  );
}

function PanelBtn({
  onClick, disabled, color, label,
}: { onClick: () => void; disabled: boolean; color: string; label: string }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        flex: 1, padding: "6px 0", border: `1px solid ${disabled ? "#e5e7eb" : color + "55"}`,
        borderRadius: 5, background: disabled ? "#f9fafb" : `${color}11`,
        color: disabled ? "#d1d5db" : color,
        fontSize: 11, fontWeight: 600, fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        letterSpacing: "0.04em",
      }}
    >
      {label}
    </button>
  );
}
