import { invoke } from "@tauri-apps/api/core";
import { useState, useCallback } from "react";
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
  const [cdpPort, setCdpPort] = useState("9222");
  const [cdpFps, setCdpFps] = useState("15");
  const [cdpStatus, setCdpStatus] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0);
  const [currentUrl, setCurrentUrl] = useState<string | null>(null);

  const attachBrowser = useCallback(async () => {
    setCdpStatus("connecting…");
    try {
      await invoke<string>("attach_to_browser", { port: parseInt(cdpPort, 10) });
      const res = await fetch(`${API}/connect-cdp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port: parseInt(cdpPort, 10), fps: parseInt(cdpFps, 10) }),
      });
      const data = await res.json();
      setCdpStatus(data.connected ? `${data.fps}fps` : `failed: ${data.detail}`);
    } catch (e) {
      setCdpStatus(`error: ${String(e).slice(0, 30)}`);
    }
  }, [cdpPort, cdpFps]);

  const detachBrowser = useCallback(async () => {
    await fetch(`${API}/disconnect-cdp`, { method: "POST" });
    setCdpStatus(null);
    setCurrentUrl(null);
  }, []);

  const onThresholdChange = (v: number) => {
    setThreshold(v);
    sendControl({ type: "set_confidence_threshold", threshold: v });
  };

  const isStreaming = cdpStatus !== null && cdpStatus !== "connecting…" && !cdpStatus.startsWith("error") && !cdpStatus.startsWith("failed");

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
              {currentUrl ?? (actionIntent ? "about:blank" : "No browser connected")}
            </span>
          </div>
        </div>

        {/* CDP controls */}
        <div style={{ display: "flex", alignItems: "center", gap: 5, flexShrink: 0 }}>
          <input value={cdpPort} onChange={(e) => setCdpPort(e.target.value)} style={miniInput("52px")} placeholder="port" />
          <input value={cdpFps}  onChange={(e) => setCdpFps(e.target.value)}  style={miniInput("36px")} placeholder="fps" />
          {isStreaming ? (
            <HeaderBtn onClick={detachBrowser} color="#ef4444" label="Detach" />
          ) : (
            <HeaderBtn onClick={attachBrowser} color="#10b981" label="⬡ Attach" />
          )}
          {cdpStatus && (
            <span style={{ fontSize: 9, color: isStreaming ? "#10b981" : "#9ca3af" }}>{cdpStatus}</span>
          )}
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
        {/* Floating labels from mockup */}
        {isStreaming && (
          <OverlayLabels />
        )}
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

/** Floating info labels overlaid on the mirror — matches the mockup's callout labels. */
function OverlayLabels() {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      <div style={floatingLabel(12, 10)}>Browser Mirror</div>
      <div style={floatingLabel(12, 36)}>Action Overlay</div>
      <div style={{ ...floatingLabel(undefined, undefined), top: 10, right: 12 }}>Ghost Trail</div>
    </div>
  );
}

function floatingLabel(top?: number, left?: number): React.CSSProperties {
  return {
    position: "absolute",
    top, left,
    fontSize: 9, fontWeight: 700, letterSpacing: "0.07em",
    color: "#f59e0b",
    background: "rgba(0,0,0,0.45)",
    padding: "2px 6px", borderRadius: 3,
    backdropFilter: "blur(4px)",
  };
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

function HeaderBtn({ onClick, color, label }: { onClick: () => void; color: string; label: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "4px 10px", border: `1px solid ${color}44`,
        borderRadius: 4, background: `${color}11`, color,
        fontSize: 10, fontWeight: 600, cursor: "pointer",
        fontFamily: "inherit", outline: "none", whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}
