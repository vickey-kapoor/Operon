import { useState } from "react";
import type { ActionIntentEvent, ElementBoundsEvent } from "../hooks/useAgentStream";
import type { RunState } from "../App";
import { LiveMirror } from "./LiveMirror";
import { ConfidenceSlider } from "./ConfidenceSlider";

type Props = {
  registerFrameCallback: (cb: (blob: Blob) => void) => () => void;
  actionIntent: ActionIntentEvent | null;
  elementBounds: ElementBoundsEvent | null;
  runState: RunState;
  sendControl: (msg: object) => void;
  latestConfidence: number;
  hitlActive: boolean;
};

type InputMode = "click" | "type" | "scroll";

const API = "http://127.0.0.1:8080";

export function LiveExecution({
  registerFrameCallback, actionIntent, elementBounds, runState, sendControl,
  latestConfidence, hitlActive,
}: Props) {
  const [inputMode, setInputMode] = useState<InputMode>("click");
  const [typeText, setTypeText] = useState("");
  const [showOverlay, setShowOverlay] = useState(true);
  const [threshold, setThreshold] = useState(0);

  const onThresholdChange = (v: number) => {
    setThreshold(v);
    sendControl({ type: "set_confidence_threshold", threshold: v });
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "#f0f2f5" }}>
      {/* Portal header */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "10px 14px", background: "#fff", borderBottom: "1px solid #e5e7eb",
        flexShrink: 0,
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: "#374151" }}>
            LIVE EXECUTION PORTAL
          </div>
          {/* Fake browser address bar */}
          <div style={{
            display: "flex", alignItems: "center", gap: 6, marginTop: 5,
            background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 4,
            padding: "3px 8px",
          }}>
            <span style={{ fontSize: 10, color: "#9ca3af" }}>●●●</span>
            <span style={{
              fontSize: 10, color: "#6b7280", flex: 1, overflow: "hidden",
              textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {actionIntent ? "about:blank" : "No browser connected"}
            </span>
          </div>
        </div>
      </div>

      {/* Mirror toolbar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 5,
        padding: "5px 12px", background: "#fff", borderBottom: "1px solid #e5e7eb",
        flexShrink: 0,
      }}>
        {(["click", "type", "scroll"] as InputMode[]).map((m) => (
          <button key={m} onClick={() => setInputMode(m)} style={modeBtn(inputMode === m)}>
            {m === "click" ? "↖ Click" : m === "type" ? "T Type" : "↕ Scroll"}
          </button>
        ))}
        {inputMode === "type" && (
          <input
            value={typeText}
            onChange={(e) => setTypeText(e.target.value)}
            placeholder="text to type…"
            style={miniInput("110px")}
          />
        )}
        <span style={{ flex: 1 }} />
        <button
          onClick={() => setShowOverlay((v) => !v)}
          style={modeBtn(showOverlay)}
          title="Toggle element bounds overlay"
        >
          ◻ Overlay
        </button>
        <span style={{ fontSize: 9, color: "#9ca3af", marginLeft: 4 }}>
          {runState.runId ? `#${runState.runId.slice(0, 10)}` : "no run"}
        </span>
      </div>

      {/* Canvas viewport — flex: 1 so it fills remaining height */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
        <LiveMirror
          registerFrameCallback={registerFrameCallback}
          actionIntent={actionIntent}
          elementBounds={elementBounds}
          showOverlay={showOverlay}
          onInput={sendControl}
          inputMode={inputMode}
          typeText={typeText}
        />
        {/* HITL dim overlay — blocks interaction and signals agent is paused */}
        {hitlActive && (
          <div style={{
            position: "absolute", inset: 0,
            background: "rgba(0,0,0,0.55)",
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            backdropFilter: "blur(2px)",
            pointerEvents: "all",
          }}>
            <div style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: 10, padding: "20px 28px", textAlign: "center",
              maxWidth: 340,
            }}>
              <div style={{ fontSize: 28, marginBottom: 10 }}>⏸</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#fbbf24", letterSpacing: "0.06em", marginBottom: 6 }}>
                AGENT PAUSED
              </div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.65)", lineHeight: 1.6 }}>
                Confidence below threshold — awaiting your approval in the left panel.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Confidence slider + HITL + controls */}
      <ConfidenceSlider
        threshold={threshold}
        liveConfidence={latestConfidence}
        onChange={onThresholdChange}
        hitlActive={hitlActive}
        onOverride={() => sendControl({ type: "override", hint: "" })}
        onPause={() => sendControl({ type: "pause" })}
        onStop={async () => {
          if (!runState.runId) return;
          await fetch(`${API}/stop`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_id: runState.runId }),
          });
        }}
      />
    </div>
  );
}

// ── Small style helpers ───────────────────────────────────────────────────────

function miniInput(width: string): React.CSSProperties {
  return {
    width, background: "#f9fafb", border: "1px solid #e5e7eb",
    borderRadius: 4, color: "#6b7280", fontSize: 10,
    padding: "3px 6px", outline: "none", fontFamily: "inherit",
  };
}

function modeBtn(active: boolean): React.CSSProperties {
  return {
    padding: "4px 9px", background: active ? "#f0f4ff" : "transparent",
    border: `1px solid ${active ? "#c7d2fe" : "#e5e7eb"}`,
    borderRadius: 4, color: active ? "#4338ca" : "#6b7280",
    fontSize: 10, fontWeight: active ? 600 : 400,
    cursor: "pointer", fontFamily: "inherit", outline: "none",
  };
}
