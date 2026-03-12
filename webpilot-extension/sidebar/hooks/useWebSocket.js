import { useCallback, useEffect } from "react";

/**
 * Provides send helpers and wires up the background → sidebar message listener.
 *
 * The actual WebSocket is owned by background.js. The sidebar communicates
 * with it via chrome.runtime.sendMessage / chrome.runtime.onMessage.
 */
export function useWebSocket(dispatch) {
  // Listen for messages forwarded from background.js.
  useEffect(() => {
    function listener(msg) {
      if (msg.type === "WS_STATUS") {
        dispatch({ type: "WS_STATUS", connected: msg.connected });
        return;
      }
      if (msg.type !== "WP_MSG") return;

      const { payload } = msg;
      switch (payload.type) {
        case "thinking":
          dispatch({ type: "THINKING" });
          break;
        case "action":
          dispatch({ type: "ACTION", payload });
          break;
        case "step_result":
          dispatch({ type: "STEP_RESULT", step: payload.step, success: payload.success });
          break;
        case "confirmation_required":
          dispatch({ type: "CONFIRMATION_REQUIRED", payload: payload.action });
          break;
        case "done":
          dispatch({ type: "DONE", payload });
          break;
        case "paused":
          dispatch({ type: "PAUSED", reason: payload.reason, narration: payload.narration });
          break;
        case "stopped":
          dispatch({ type: "STOPPED" });
          break;
        case "error":
          dispatch({ type: "ERROR", message: payload.message });
          break;
        default:
          break;
      }
    }

    chrome.runtime.onMessage.addListener(listener);
    return () => chrome.runtime.onMessage.removeListener(listener);
  }, [dispatch]);

  const sendTask = useCallback((intent) => {
    chrome.runtime.sendMessage({ type: "TASK", intent }, (resp) => {
      if (chrome.runtime.lastError) {
        dispatch({ type: "ERROR", message: "Failed to send task: " + chrome.runtime.lastError.message });
      }
    });
  }, []);

  const sendInterrupt = useCallback((instruction) => {
    chrome.runtime.sendMessage({ type: "INTERRUPT", instruction }, (resp) => {
      if (chrome.runtime.lastError) {
        dispatch({ type: "ERROR", message: "Failed to send interrupt: " + chrome.runtime.lastError.message });
      }
    });
  }, []);

  const sendConfirm = useCallback((confirmed) => {
    chrome.runtime.sendMessage({ type: "CONFIRM", confirmed }, (resp) => {
      if (chrome.runtime.lastError) {
        dispatch({ type: "ERROR", message: "Failed to send confirm: " + chrome.runtime.lastError.message });
      }
    });
  }, []);

  const sendStop = useCallback(() => {
    chrome.runtime.sendMessage({ type: "STOP" }, (resp) => {
      if (chrome.runtime.lastError) {
        dispatch({ type: "ERROR", message: "Failed to send stop: " + chrome.runtime.lastError.message });
      }
    });
  }, []);

  const sendResume = useCallback(() => {
    chrome.runtime.sendMessage({ type: "RESUME" }, (resp) => {
      if (chrome.runtime.lastError) {
        dispatch({ type: "ERROR", message: "Failed to send resume: " + chrome.runtime.lastError.message });
      }
    });
  }, []);

  return { sendTask, sendInterrupt, sendConfirm, sendStop, sendResume };
}
