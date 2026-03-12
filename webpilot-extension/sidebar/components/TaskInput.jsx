import React, { useRef, useState } from "react";
import { useVoiceInput } from "../hooks/useVoiceInput.js";

export default function TaskInput({ onSubmit, disabled, isRunning, placeholder }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  const { isListening, startListening, stopListening, supported, error: voiceError } = useVoiceInput((transcript) => {
    setText(transcript);
  });

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const val = text.trim();
    if (!val || disabled) return;
    onSubmit(val);
    setText("");
  }

  return (
    <div style={styles.wrapper}>
      {voiceError === "mic-denied" && (
        <p style={styles.voiceError}>
          Mic blocked — go to <strong>chrome://settings/content/microphone</strong> and allow this extension.
        </p>
      )}
      {voiceError && voiceError !== "mic-denied" && (
        <p style={styles.voiceError}>Mic error: {voiceError}</p>
      )}
      <div style={styles.row}>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Describe what to do… (Ctrl+Enter to send)"}
          disabled={disabled}
          rows={3}
          style={styles.textarea}
        />
        {supported && (
          <button
            onMouseDown={startListening}
            onMouseUp={stopListening}
            onTouchStart={startListening}
            onTouchEnd={stopListening}
            style={{ ...styles.micBtn, background: isListening ? "#2b6cb0" : "#2d3748" }}
            title="Hold to speak"
          >
            🎤
          </button>
        )}
      </div>
      <button
        onClick={submit}
        disabled={disabled || !text.trim()}
        title={isRunning ? "Send to interrupt current task" : "Start a new task"}
        style={{
          ...styles.sendBtn,
          background: isRunning ? "#744210" : "#2b6cb0",
          opacity: disabled || !text.trim() ? 0.5 : 1,
        }}
      >
        {isRunning ? "Interrupt" : "Send"}
      </button>
    </div>
  );
}

const styles = {
  wrapper: { display: "flex", flexDirection: "column", gap: 8 },
  row: { display: "flex", gap: 6, alignItems: "flex-start" },
  textarea: {
    flex: 1,
    background: "#1a202c",
    color: "#e2e8f0",
    border: "1px solid #4a5568",
    borderRadius: 8,
    padding: "8px 10px",
    fontSize: 13,
    resize: "vertical",
    outline: "none",
    fontFamily: "inherit",
  },
  micBtn: {
    border: "none",
    borderRadius: 8,
    padding: "8px 10px",
    cursor: "pointer",
    fontSize: 16,
    flexShrink: 0,
  },
  voiceError: { fontSize: 11, color: "#fc8181", marginBottom: 4 },
  sendBtn: {
    border: "none",
    borderRadius: 8,
    padding: "8px 16px",
    color: "#fff",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    width: "100%",
  },
};
