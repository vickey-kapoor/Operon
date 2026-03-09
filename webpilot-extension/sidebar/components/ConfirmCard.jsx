import React from "react";

export default function ConfirmCard({ action, onConfirm }) {
  return (
    <div style={styles.card}>
      <p style={styles.title}>Confirm action</p>
      <p style={styles.narration}>
        {action.narration || action.action_label || action.action}
      </p>
      <div style={styles.buttons}>
        <button onClick={() => onConfirm(true)} style={styles.proceed}>
          ✅ Proceed
        </button>
        <button onClick={() => onConfirm(false)} style={styles.cancel}>
          ⛔ Cancel
        </button>
      </div>
    </div>
  );
}

const styles = {
  card: {
    background: "#1a202c",
    border: "1px solid #4a5568",
    borderRadius: 10,
    padding: "12px 16px",
    margin: "0 16px 8px",
    flexShrink: 0,
  },
  title: { fontSize: 12, color: "#a0aec0", marginBottom: 6, textTransform: "uppercase", letterSpacing: 0.5 },
  narration: { fontSize: 13, color: "#e2e8f0", marginBottom: 12, lineHeight: 1.5 },
  buttons: { display: "flex", gap: 8 },
  proceed: {
    flex: 1,
    background: "#276749",
    color: "#9ae6b4",
    border: "none",
    borderRadius: 6,
    padding: "8px",
    fontSize: 13,
    cursor: "pointer",
    fontWeight: 600,
  },
  cancel: {
    flex: 1,
    background: "#742a2a",
    color: "#fc8181",
    border: "none",
    borderRadius: 6,
    padding: "8px",
    fontSize: 13,
    cursor: "pointer",
    fontWeight: 600,
  },
};
