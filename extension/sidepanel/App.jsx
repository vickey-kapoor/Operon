import { useState, useEffect, useRef, useCallback } from "react";
import VoiceButton from "./components/VoiceButton.jsx";
import StepLog from "./components/StepLog.jsx";
import ScreenshotView from "./components/ScreenshotView.jsx";

const DEFAULT_BACKEND = "https://your-cloud-run-url.run.app";
const MAX_STEPS = 20;
const ACTION_DELAY_MS = 800;
const NAVIGATE_DELAY_MS = 2500; // extra wait after navigate actions for page load

// ---------------------------------------------------------------------------
// Styles (inline — no CSS file needed for extension side panel)
// ---------------------------------------------------------------------------
const S = {
  app: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    padding: "12px",
    gap: "10px",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: { fontSize: "15px", fontWeight: "700", color: "#7c9ef8" },
  settingsBtn: {
    background: "none",
    border: "1px solid #333",
    borderRadius: "6px",
    color: "#aaa",
    cursor: "pointer",
    padding: "4px 8px",
    fontSize: "12px",
  },
  modal: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.7)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 100,
  },
  modalBox: {
    background: "#1a1a24",
    border: "1px solid #333",
    borderRadius: "10px",
    padding: "20px",
    width: "300px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  modalTitle: { fontWeight: "700", fontSize: "14px" },
  label: { fontSize: "12px", color: "#aaa", marginBottom: "4px" },
  input: {
    width: "100%",
    padding: "8px",
    background: "#0f0f1a",
    border: "1px solid #333",
    borderRadius: "6px",
    color: "#e8e8e8",
    fontSize: "13px",
  },
  row: { display: "flex", gap: "8px", alignItems: "flex-end" },
  textarea: {
    flex: 1,
    padding: "8px",
    background: "#1a1a24",
    border: "1px solid #333",
    borderRadius: "6px",
    color: "#e8e8e8",
    fontSize: "13px",
    resize: "none",
    minHeight: "64px",
    fontFamily: "inherit",
  },
  startBtn: (running) => ({
    padding: "8px 14px",
    borderRadius: "6px",
    border: "none",
    cursor: running ? "not-allowed" : "pointer",
    background: running ? "#555" : "#3d6cf0",
    color: "#fff",
    fontWeight: "700",
    fontSize: "13px",
    alignSelf: "flex-end",
  }),
  stopBtn: {
    padding: "8px 14px",
    borderRadius: "6px",
    border: "none",
    cursor: "pointer",
    background: "#c0392b",
    color: "#fff",
    fontWeight: "700",
    fontSize: "13px",
    alignSelf: "flex-end",
  },
  status: (color) => ({
    fontSize: "12px",
    color: color,
    minHeight: "18px",
  }),
  saveBtn: {
    padding: "8px 16px",
    borderRadius: "6px",
    border: "none",
    cursor: "pointer",
    background: "#3d6cf0",
    color: "#fff",
    fontWeight: "700",
    fontSize: "13px",
    alignSelf: "flex-end",
  },
};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState([]);
  const [screenshot, setScreenshot] = useState(null);
  const [status, setStatus] = useState("");
  const [statusColor, setStatusColor] = useState("#aaa");
  const [showSettings, setShowSettings] = useState(false);
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND);
  const [apiKey, setApiKey] = useState("");
  const stopRef = useRef(false);
  const sessionRef = useRef(null);

  // Load stored settings on mount.
  useEffect(() => {
    if (typeof chrome !== "undefined" && chrome.storage) {
      chrome.storage.sync.get(["backendUrl", "apiKey"], (data) => {
        if (data.backendUrl) setBackendUrl(data.backendUrl);
        if (data.apiKey) setApiKey(data.apiKey);
      });
    }
  }, []);

  const saveSettings = () => {
    if (typeof chrome !== "undefined" && chrome.storage) {
      chrome.storage.sync.set({ backendUrl, apiKey });
    }
    setShowSettings(false);
  };

  // ── Agent loop helpers ─────────────────────────────────────────────────

  const captureScreenshot = () =>
    new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: "CAPTURE_SCREENSHOT" }, (resp) => {
        if (chrome.runtime.lastError || resp?.error) {
          reject(new Error(resp?.error || chrome.runtime.lastError?.message));
        } else {
          resolve(resp.dataUrl);
        }
      });
    });

  const executeAction = (action) =>
    new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "EXECUTE_ACTION", action }, (resp) => {
        resolve(resp);
      });
    });

  const apiHeaders = () => {
    const h = { "Content-Type": "application/json" };
    if (apiKey) h["X-API-Key"] = apiKey;
    return h;
  };

  const addStep = useCallback((step) => {
    setSteps((prev) => [...prev, step]);
  }, []);

  // ── Main agent loop ────────────────────────────────────────────────────

  const startAgent = async () => {
    if (!task.trim()) {
      setStatus("Please enter a task.");
      setStatusColor("#f0a53d");
      return;
    }
    stopRef.current = false;
    setRunning(true);
    setSteps([]);
    setScreenshot(null);
    setStatus("Starting session…");
    setStatusColor("#7c9ef8");

    let sessionId = null;
    try {
      // 1. Create session.
      const res = await fetch(`${backendUrl}/sessions`, {
        method: "POST",
        headers: apiHeaders(),
      });
      if (!res.ok) throw new Error(`POST /sessions failed: ${res.status}`);
      const data = await res.json();
      sessionId = data.session_id;
      sessionRef.current = sessionId;

      let stepCount = 0;
      let lastObservation = "";
      let repeatCount = 0;

      // 2. Agent loop.
      while (stepCount < MAX_STEPS && !stopRef.current) {
        stepCount++;
        setStatus(`Step ${stepCount}/${MAX_STEPS} — capturing screenshot…`);

        // a. Capture screenshot.
        const dataUrl = await captureScreenshot();
        const base64 = dataUrl.replace(/^data:image\/png;base64,/, "");
        setScreenshot(dataUrl);

        // b. Send to backend.
        setStatus(`Step ${stepCount}/${MAX_STEPS} — thinking…`);
        const stepRes = await fetch(`${backendUrl}/sessions/${sessionId}/step`, {
          method: "POST",
          headers: apiHeaders(),
          body: JSON.stringify({ image_b64: base64, task }),
        });
        if (!stepRes.ok) {
          const errText = await stepRes.text();
          throw new Error(`Step failed (${stepRes.status}): ${errText}`);
        }
        const plan = await stepRes.json();

        // c. Log the step.
        addStep({
          step: stepCount,
          observation: plan.observation,
          reasoning: plan.reasoning,
          actions: plan.actions || [],
          done: plan.done,
          result: plan.result,
        });

        if (plan.done) {
          setStatus(`Done: ${plan.result || "Task completed."}`);
          setStatusColor("#2ecc71");
          break;
        }

        // Detect stuck loop — same observation 3 times in a row.
        if (plan.observation === lastObservation) {
          repeatCount++;
          if (repeatCount >= 3) {
            setStatus("Agent appears stuck (same observation 3× in a row). Stopping.");
            setStatusColor("#f0a53d");
            fetch(`${backendUrl}/sessions/${sessionId}/events`, {
              method: "POST", headers: apiHeaders(),
              body: JSON.stringify({ event: "loop_detected", step: stepCount }),
            }).catch(() => {});
            break;
          }
        } else {
          lastObservation = plan.observation;
          repeatCount = 0;
        }

        // d. Execute actions.
        for (const action of plan.actions || []) {
          if (stopRef.current) break;
          if (action.type === "done") break;
          if (action.type === "screenshot") continue;
          await executeAction(action);
          // Navigate needs extra time for page load.
          await sleep(action.type === "navigate" ? NAVIGATE_DELAY_MS : ACTION_DELAY_MS);
        }

        if (stopRef.current) {
          setStatus("Stopped by user.");
          setStatusColor("#f0a53d");
          fetch(`${backendUrl}/sessions/${sessionId}/events`, {
            method: "POST", headers: apiHeaders(),
            body: JSON.stringify({ event: "user_stopped", step: stepCount }),
          }).catch(() => {});
          break;
        }
      }

      if (stepCount >= MAX_STEPS && !stopRef.current) {
        setStatus(`Reached max ${MAX_STEPS} steps.`);
        setStatusColor("#f0a53d");
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      setStatusColor("#e74c3c");
      addStep({ step: -1, observation: `Error: ${err.message}`, actions: [], done: true });
    } finally {
      // Cleanup session.
      if (sessionId) {
        fetch(`${backendUrl}/sessions/${sessionId}`, {
          method: "DELETE",
          headers: apiHeaders(),
        }).catch(() => {});
        sessionRef.current = null;
      }
      setRunning(false);
    }
  };

  const stopAgent = () => {
    stopRef.current = true;
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <span style={S.title}>UI Navigator</span>
        <button style={S.settingsBtn} onClick={() => setShowSettings(true)}>
          ⚙ Settings
        </button>
      </div>

      {/* Settings modal */}
      {showSettings && (
        <div style={S.modal} onClick={() => setShowSettings(false)}>
          <div style={S.modalBox} onClick={(e) => e.stopPropagation()}>
            <div style={S.modalTitle}>Settings</div>
            <div>
              <div style={S.label}>Backend URL</div>
              <input
                style={S.input}
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="https://your-cloud-run-url.run.app"
              />
            </div>
            <div>
              <div style={S.label}>API Key</div>
              <input
                style={S.input}
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="X-API-Key value"
              />
            </div>
            <button style={S.saveBtn} onClick={saveSettings}>
              Save
            </button>
          </div>
        </div>
      )}

      {/* Task input */}
      <div style={S.row}>
        <textarea
          style={S.textarea}
          placeholder="Enter task… e.g. 'Go to example.com and tell me the page title'"
          value={task}
          onChange={(e) => setTask(e.target.value)}
          disabled={running}
        />
        <VoiceButton onTranscript={setTask} disabled={running} />
      </div>

      {/* Controls */}
      <div style={S.row}>
        <button
          style={S.startBtn(running)}
          onClick={startAgent}
          disabled={running}
        >
          ▶ Start
        </button>
        {running && (
          <button style={S.stopBtn} onClick={stopAgent}>
            ■ Stop
          </button>
        )}
      </div>

      {/* Status */}
      <div style={S.status(statusColor)}>{status}</div>

      {/* Screenshot */}
      {screenshot && <ScreenshotView src={screenshot} />}

      {/* Step log */}
      <StepLog steps={steps} />
    </div>
  );
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
