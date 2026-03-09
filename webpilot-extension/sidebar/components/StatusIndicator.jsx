import React from "react";

const STATUS_COLORS = {
  idle: "#68d391",     // green
  done: "#68d391",     // green
  thinking: "#f6ad55", // amber
  running: "#f6ad55",  // amber
  confirming: "#f6ad55",
  error: "#fc8181",    // red
  disconnected: "#718096", // grey
};

export default function StatusIndicator({ connected, status }) {
  const effectiveStatus = connected ? status : "disconnected";
  const color = STATUS_COLORS[effectiveStatus] || "#718096";
  const label = connected
    ? status.charAt(0).toUpperCase() + status.slice(1)
    : "Disconnected";

  return (
    <div style={styles.wrapper} title={label}>
      <div
        style={{
          ...styles.dot,
          background: color,
          boxShadow: color !== "#718096" ? `0 0 6px ${color}` : "none",
        }}
      />
      <span style={{ ...styles.label, color }}>{label}</span>
    </div>
  );
}

const styles = {
  wrapper: { display: "flex", alignItems: "center", gap: 6 },
  dot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  label: { fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 },
};
