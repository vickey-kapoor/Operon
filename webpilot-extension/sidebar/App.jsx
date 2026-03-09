import React, { useCallback, useEffect, useReducer, useRef } from "react";
import TaskInput from "./components/TaskInput.jsx";
import ActionLog from "./components/ActionLog.jsx";
import ConfirmCard from "./components/ConfirmCard.jsx";
import StatusIndicator from "./components/StatusIndicator.jsx";
import { useWebSocket } from "./hooks/useWebSocket.js";
import { useVoiceOutput } from "./hooks/useVoiceOutput.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const initialState = {
  status: "idle",       // idle | thinking | running | confirming | done | error
  wsConnected: false,
  actionLog: [],
  currentTask: "",
  pendingAction: null,
  errorMessage: "",
  completionMessage: "",
};

function reducer(state, action) {
  switch (action.type) {
    case "WS_STATUS":
      return { ...state, wsConnected: action.connected };
    case "THINKING":
      return { ...state, status: "thinking" };
    case "ACTION":
      return {
        ...state,
        status: "running",
        actionLog: [...state.actionLog, { ...action.payload, success: null }],
      };
    case "STEP_RESULT": {
      const { step, success } = action;
      const log = state.actionLog.map((item) =>
        item._step === step ? { ...item, success } : item
      );
      return { ...state, actionLog: log };
    }
    case "CONFIRMATION_REQUIRED":
      return { ...state, status: "confirming", pendingAction: action.payload };
    case "DONE":
      return {
        ...state,
        status: "done",
        pendingAction: null,
        completionMessage: action.payload.narration || "Task complete.",
        actionLog: [
          ...state.actionLog,
          { action: "done", narration: action.payload.narration || "Task complete." },
        ],
      };
    case "STOPPED":
      return { ...state, status: "idle", pendingAction: null };
    case "ERROR":
      return { ...state, status: "error", errorMessage: action.message, pendingAction: null };
    case "SET_TASK":
      return { ...state, currentTask: action.task, actionLog: [], errorMessage: "", completionMessage: "" };
    case "RESET":
      return { ...initialState, wsConnected: state.wsConnected };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { speak } = useVoiceOutput();
  const { sendTask, sendInterrupt, sendConfirm, sendStop } = useWebSocket(dispatch);
  const prevStatusRef = useRef(state.status);
  const prevLogLenRef = useRef(0);

  // Check initial WS status on mount.
  useEffect(() => {
    chrome.runtime.sendMessage({ type: "GET_STATUS" }, (resp) => {
      if (resp) dispatch({ type: "WS_STATUS", connected: resp.connected });
    });
  }, []);

  const handleSubmit = useCallback(
    (intent) => {
      if (!intent.trim()) return;

      if (state.status === "running" || state.status === "thinking") {
        // Treat new input as interrupt.
        speak(`Got it, updating.`);
        sendInterrupt(intent);
        dispatch({ type: "ACTION", payload: { action: "interrupt", narration: `Interrupting: ${intent}` } });
      } else {
        speak(`Got it. ${intent}`);
        dispatch({ type: "SET_TASK", task: intent });
        dispatch({ type: "THINKING" });
        sendTask(intent);
      }
    },
    [state.status, sendTask, sendInterrupt, speak]
  );

  const handleConfirm = useCallback(
    (confirmed) => {
      sendConfirm(confirmed);
      // Only dispatch STOPPED on denial — let server drive state on confirmation.
      if (!confirmed) dispatch({ type: "STOPPED" });
    },
    [sendConfirm, dispatch]
  );

  const handleStop = useCallback(() => {
    sendStop();
    dispatch({ type: "STOPPED" });
  }, [sendStop]);

  // Voice narration: only fire when a NEW entry is added to the log, not on STEP_RESULT updates.
  useEffect(() => {
    if (state.actionLog.length <= prevLogLenRef.current) return;
    prevLogLenRef.current = state.actionLog.length;
    const last = state.actionLog[state.actionLog.length - 1];
    // Skip narrating the "done" entry — handled separately below with the follow-up prompt.
    if (last?.narration && last.action !== "done") speak(last.narration);
  }, [state.actionLog, speak]);

  // Narrate task completion with a follow-up prompt.
  useEffect(() => {
    if (state.status === "done" && state.completionMessage) {
      const task = state.currentTask || "your task";
      speak(`"${task}" has been completed. Would you like me to do anything else on this page?`);
    }
  }, [state.status, state.completionMessage, speak]);

  // Narrate confirmation prompt when agent pauses for user approval.
  useEffect(() => {
    if (state.status === "confirming" && state.pendingAction?.narration) {
      speak(state.pendingAction.narration);
    }
  }, [state.status, state.pendingAction, speak]);

  // Narrate stop: detect transition from running/thinking → idle (user-initiated stop).
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = state.status;
    if (state.status === "idle" && (prev === "running" || prev === "thinking")) {
      speak("Stopped. What would you like to do?");
    }
  }, [state.status, speak]);

  const isRunning = state.status === "running" || state.status === "thinking";

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>WebPilot</span>
        <StatusIndicator connected={state.wsConnected} status={state.status} />
      </div>

      {/* Action log */}
      <div style={styles.logArea}>
        <ActionLog log={state.actionLog} status={state.status} />
      </div>

      {/* Completion banner */}
      {state.status === "done" && state.completionMessage && (
        <div style={styles.doneBanner}>
          <div style={styles.doneText}>✓ "{state.currentTask}" has been completed</div>
          <div style={styles.donePrompt}>Would you like me to do anything else on this page?</div>
        </div>
      )}

      {/* Error banner */}
      {state.status === "error" && (
        <div style={styles.errorBanner}>{state.errorMessage || "An error occurred."}</div>
      )}

      {/* Confirmation card */}
      {state.status === "confirming" && state.pendingAction && (
        <ConfirmCard action={state.pendingAction} onConfirm={handleConfirm} />
      )}

      {/* Input */}
      <div style={styles.inputArea}>
        {isRunning && (
          <button onClick={handleStop} style={styles.stopBtn}>
            ⬛ Stop
          </button>
        )}
        <TaskInput
          onSubmit={handleSubmit}
          disabled={state.status === "confirming"}
          isRunning={isRunning}
          placeholder={isRunning ? "Interrupt with new instruction…" : "Describe what to do…"}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    background: "#0f1117",
    color: "#e2e8f0",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 16px",
    borderBottom: "1px solid #2d3748",
    flexShrink: 0,
  },
  title: {
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: 0.5,
    color: "#63b3ed",
  },
  logArea: {
    flex: 1,
    overflowY: "auto",
    padding: "8px 0",
  },
  doneBanner: {
    background: "#1a2e1a",
    borderTop: "1px solid #2f6b2f",
    padding: "10px 16px",
    flexShrink: 0,
  },
  doneText: {
    color: "#68d391",
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 4,
  },
  donePrompt: {
    color: "#a0aec0",
    fontSize: 12,
  },
  errorBanner: {
    background: "#742a2a",
    color: "#fc8181",
    padding: "8px 16px",
    fontSize: 13,
    flexShrink: 0,
  },
  inputArea: {
    borderTop: "1px solid #2d3748",
    padding: "12px 16px",
    flexShrink: 0,
  },
  stopBtn: {
    marginBottom: 8,
    background: "#742a2a",
    color: "#fc8181",
    border: "none",
    borderRadius: 6,
    padding: "6px 12px",
    fontSize: 13,
    cursor: "pointer",
    width: "100%",
  },
};
