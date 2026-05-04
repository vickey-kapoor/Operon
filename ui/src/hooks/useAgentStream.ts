import { useEffect, useRef, useState, useCallback } from "react";

// ── Event types ───────────────────────────────────────────────────────────────

export type StepEvent = {
  type: "step";
  run_id: string;
  step: number;
  status: string;
  subgoal?: string;
  action_type?: string;
  rule_name?: string | null;
  confidence?: number;
  rationale?: string;
  perception?: string;
  receivedAt?: number;
};

export type RunEvent = {
  type: "run_started" | "run_completed";
  run_id: string;
  intent?: string;
  status: string;
};

export type ControlAck = {
  type: "control_ack";
  action: string;
  hint?: string;
  threshold?: number;
};

export type ActionIntentEvent = {
  type: "action_intent";
  action_type: string;
  x: number | null;
  y: number | null;
  viewport_w: number;
  viewport_h: number;
  confidence: number;
  rule_name: string | null;
  subgoal: string | null;
  rationale: string;
};

export type ElementBound = {
  tag: string;
  role: string;
  text: string;
  x: number;
  y: number;
  w: number;
  h: number;
};

export type ElementBoundsEvent = {
  type: "element_bounds";
  viewport_w: number;
  viewport_h: number;
  elements: ElementBound[];
};

export type HitlRequiredEvent = {
  type: "hitl_required";
  run_id: string;
  message: string | null;
  confidence: number | null;
  threshold: number | null;
  pending_action: string | null;
  subgoal: string | null;
  rationale: string | null;
};

export type AgentEvent = StepEvent | RunEvent | ControlAck | ActionIntentEvent | ElementBoundsEvent | HitlRequiredEvent;

// ── Hook state ────────────────────────────────────────────────────────────────

export type StreamState = {
  connected: boolean;
  events: AgentEvent[];
  frameUrl: string | null;
  elementBounds: ElementBoundsEvent | null;
  latestActionIntent: ActionIntentEvent | null;
  pendingDecision: HitlRequiredEvent | null;
};

const WS_URL = "ws://127.0.0.1:9001";
const RECONNECT_MS = 2000;

export function useAgentStream() {
  const [state, setState] = useState<StreamState>({
    connected: false,
    events: [],
    frameUrl: null,
    elementBounds: null,
    latestActionIntent: null,
    pendingDecision: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const frameUrlRef = useRef<string | null>(null);
  // Frame callbacks: registered by canvas components to avoid React re-renders on every frame.
  const frameCallbacksRef = useRef<Set<(blob: Blob) => void>>(new Set());

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setState((s) => ({ ...s, connected: true }));

    ws.onclose = () => {
      setState((s) => ({ ...s, connected: false }));
      setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (evt) => {
      if (evt.data instanceof Blob) {
        // Binary JPEG frame — deliver to registered canvas callbacks directly (zero React overhead).
        const blob = evt.data as Blob;
        frameCallbacksRef.current.forEach((cb) => cb(blob));
        // Also maintain frameUrl for any fallback <img> consumers.
        const url = URL.createObjectURL(blob);
        if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
        frameUrlRef.current = url;
        setState((s) => ({ ...s, frameUrl: url }));
        return;
      }

      try {
        const event: AgentEvent = JSON.parse(evt.data as string);
        if (event.type === "step") {
          (event as StepEvent).receivedAt = Date.now();
        }
        if (event.type === "element_bounds") {
          setState((s) => ({ ...s, elementBounds: event as ElementBoundsEvent }));
        } else if (event.type === "action_intent") {
          setState((s) => ({ ...s, latestActionIntent: event as ActionIntentEvent }));
        } else if (event.type === "hitl_required") {
          setState((s) => ({ ...s, pendingDecision: event as HitlRequiredEvent }));
        } else if (event.type === "run_completed" || event.type === "run_started") {
          // Clear any pending decision when a run transitions.
          setState((s) => ({
            ...s,
            pendingDecision: null,
            events: [...s.events.slice(-299), event],
          }));
        } else {
          setState((s) => ({
            ...s,
            events: [...s.events.slice(-299), event],
          }));
        }
      } catch {
        // ignore malformed
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
    };
  }, [connect]);

  /** Register a callback that receives every JPEG Blob frame without triggering React renders. */
  const registerFrameCallback = useCallback((cb: (blob: Blob) => void) => {
    frameCallbacksRef.current.add(cb);
    return () => { frameCallbacksRef.current.delete(cb); };
  }, []);

  const sendControl = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { ...state, sendControl, registerFrameCallback };
}
