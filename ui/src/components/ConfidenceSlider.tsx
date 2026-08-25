type Props = {
  threshold: number;           // 0.0 – 1.0
  liveConfidence: number;      // most recent step confidence, 0.0 – 1.0
  onChange: (v: number) => void;
  hitlActive: boolean;
  onOverride: () => void;
  onPause: () => void;
  onStop: () => void;
};

export function ConfidenceSlider({
  threshold, liveConfidence, onChange, hitlActive, onOverride, onPause, onStop,
}: Props) {
  const pct = Math.round(threshold * 100);
  const confPct = Math.round(liveConfidence * 100);

  return (
    <div style={{ padding: "10px 14px", borderTop: "1px solid #e5e7eb", background: "#fff" }}>
      {/* Threshold slider row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <span style={{ fontSize: 10, color: "#6b7280", fontWeight: 600, whiteSpace: "nowrap", letterSpacing: "0.05em" }}>
          AUTONOMY THRESHOLD
        </span>
        <input
          type="range" min={0} max={100} value={pct}
          onChange={(e) => onChange(parseInt(e.target.value, 10) / 100)}
          style={{ flex: 1, accentColor: "#6366f1", cursor: "pointer", height: 4 }}
        />
        <span style={{ fontSize: 11, fontWeight: 700, color: "#1e2030", minWidth: 32, textAlign: "right" }}>
          {pct}%
        </span>
      </div>

      {/* Confidence meter row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span style={{ fontSize: 10, color: "#6b7280", fontWeight: 600, whiteSpace: "nowrap", letterSpacing: "0.05em" }}>
          CONFIDENCE METER
        </span>
        <div style={{ flex: 1, height: 6, background: "#f3f4f6", borderRadius: 3, overflow: "hidden" }}>
          <div style={{
            height: "100%", width: `${confPct}%`,
            background: confPct >= 80 ? "#10b981" : confPct >= 60 ? "#f59e0b" : "#ef4444",
            borderRadius: 3, transition: "width 0.4s ease",
          }} />
        </div>
        <span style={{ fontSize: 11, color: "#6b7280", minWidth: 28, textAlign: "right" }}>{confPct}%</span>
      </div>

      {/* HITL banner */}
      {(hitlActive || threshold > 0) && (
        <div style={{
          padding: "7px 10px", borderRadius: 5, marginBottom: 10,
          background: hitlActive ? "#fff7ed" : "#f0f4ff",
          border: `1px solid ${hitlActive ? "#fed7aa" : "#c7d2fe"}`,
          fontSize: 11,
          color: hitlActive ? "#9a3412" : "#4338ca",
        }}>
          {hitlActive
            ? "⚠ HUMAN-IN-THE-LOOP — agent paused, waiting for you"
            : `Agent pauses when confidence < ${pct}% — threshold active`}
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
        <ActionBtn label="Override" color="#6366f1" onClick={onOverride} />
        <ActionBtn label="Pause"    color="#f59e0b" onClick={onPause} />
        <ActionBtn label="Stop"     color="#ef4444" onClick={onStop} />
      </div>
    </div>
  );
}

function ActionBtn({ label, color, onClick }: { label: string; color: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "5px 14px", border: `1px solid ${color}55`,
        borderRadius: 5, background: `${color}11`, color,
        fontSize: 11, fontWeight: 600, fontFamily: "inherit",
        cursor: "pointer", letterSpacing: "0.04em",
      }}
    >
      {label}
    </button>
  );
}
