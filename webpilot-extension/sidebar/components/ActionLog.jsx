import React, { useEffect, useRef } from "react";

const ACTION_ICONS = {
  click: "🖱️",
  type: "⌨️",
  scroll: "📜",
  navigate: "🌐",
  wait: "⏳",
  key: "⌨️",
  done: "✅",
  interrupt: "⚡",
  error: "❌",
};

export default function ActionLog({ log, status }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log, status]);

  return (
    <div style={styles.container}>
      {log.length === 0 && status === "idle" && (
        <p style={styles.empty}>Ready. Describe a task above to get started.</p>
      )}

      {log.map((item, i) => (
        <div key={i} style={styles.item} title={item.narration || ""}>
          <span style={styles.icon}>{ACTION_ICONS[item.action] || "🔧"}</span>
          <div style={styles.body}>
            <span style={styles.label}>
              {item.action_label || item.narration || item.action}
            </span>
            {item.success === false && (
              <span style={styles.failed}>✗ failed</span>
            )}
            {item.success === true && item.action !== "done" && (
              <span style={styles.ok}>✓</span>
            )}
            {item.narration && item.action_label && (
              <span style={styles.narration}>{item.narration}</span>
            )}
          </div>
        </div>
      ))}

      {status === "thinking" && (
        <div style={styles.thinking}>
          <span style={styles.spinner}>⋯</span>
          <span style={{ color: "#a0aec0", fontSize: 13 }}>Thinking…</span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

const styles = {
  container: { padding: "4px 16px" },
  empty: { color: "#718096", fontSize: 13, padding: "20px 0", textAlign: "center" },
  item: {
    display: "flex",
    alignItems: "flex-start",
    gap: 8,
    padding: "6px 0",
    borderBottom: "1px solid #1a202c",
  },
  icon: { fontSize: 14, flexShrink: 0, marginTop: 2 },
  body: { display: "flex", flexDirection: "column", gap: 2, flex: 1 },
  label: { fontSize: 13, color: "#cbd5e0", lineHeight: 1.4 },
  narration: { fontSize: 11, color: "#718096", fontStyle: "italic" },
  failed: { fontSize: 11, color: "#fc8181", fontWeight: 600 },
  ok: { fontSize: 11, color: "#68d391", fontWeight: 600 },
  thinking: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 0",
  },
  spinner: { fontSize: 20, color: "#63b3ed" },
};
