/**
 * Global agent state — managed by Zustand 5.
 *
 * Single source of truth for: run lifecycle, confidence score, subgoal tree,
 * HITL state, and navigation.  Components subscribe to specific slices via
 * selector functions so unrelated re-renders are minimised.
 */

import { create } from "zustand";
import type { HitlRequiredEvent, StepEvent } from "../hooks/useAgentStream";

// ── Types ────────────────────────────────────────────────────────────────────

export interface Subgoal {
  id: string;
  label: string;
  status: "pending" | "active" | "complete" | "failed";
}

interface AgentState {
  // Run context
  runId: string | null;
  currentTask: string;
  runStatus: string;       // "idle" | "running" | "waiting_for_user" | "succeeded" | "failed"

  // Agent intelligence
  isAgentActive: boolean;  // true when runStatus === "running"
  confidenceScore: number; // 0.0 – 1.0, latest known confidence
  subgoals: Subgoal[];
  stepCount: number;

  // HITL
  hitlActive: boolean;
  pendingDecision: HitlRequiredEvent | null;

  // UI
  activeNav: string;

  // Settings / Moat Builder
  disabledRules: string[];
  sessionMode: "fresh" | "observable";
}

interface AgentActions {
  // Nav
  setActiveNav: (nav: string) => void;

  // Run lifecycle
  setRunStarted: (runId: string, intent: string) => void;
  /** Direct state sync — used by child components after e.g. POST /resume. */
  applyRunState: (s: { runId: string | null; intent: string; status: string }) => void;
  setRunCompleted: (status: string) => void;

  // Per-step updates
  updateFromStep: (step: StepEvent) => void;
  setConfidenceScore: (score: number) => void;

  // HITL
  updateFromHitl: (decision: HitlRequiredEvent) => void;
  setPendingDecision: (d: HitlRequiredEvent | null) => void;

  // Settings
  setDisabledRules: (rules: string[]) => void;
  setSessionMode: (mode: "fresh" | "observable") => void;

  // Reset
  reset: () => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const IDLE: AgentState = {
  runId: null,
  currentTask: "",
  runStatus: "idle",
  isAgentActive: false,
  confidenceScore: 0,
  subgoals: [],
  stepCount: 0,
  hitlActive: false,
  pendingDecision: null,
  activeNav: "tasks",
  disabledRules: [],
  sessionMode: "fresh",
};

/** Derive next subgoal list from a step event, incrementally. */
function applyStepSubgoals(current: Subgoal[], step: StepEvent): Subgoal[] {
  const newLabel = step.subgoal?.trim();
  if (!newLabel) return current;

  const activeNode = current.find((s) => s.status === "active");
  if (activeNode?.label === newLabel) return current; // same subgoal, no change

  // Mark previous active as complete (or failed if step failed).
  const updated = current.map((s): Subgoal =>
    s.status === "active"
      ? { ...s, status: step.status === "failed" ? "failed" : "complete" }
      : s,
  );

  const exists = updated.find((s) => s.label === newLabel);
  if (exists) {
    return updated.map((s): Subgoal =>
      s.label === newLabel ? { ...s, status: "active" } : s,
    );
  }
  return [...updated, { id: newLabel, label: newLabel, status: "active" }];
}

// ── Store ─────────────────────────────────────────────────────────────────────

export const useAgentStore = create<AgentState & AgentActions>()((set) => ({
  ...IDLE,

  setActiveNav: (nav) => set({ activeNav: nav }),

  setRunStarted: (runId, intent) =>
    set({
      ...IDLE,
      runId,
      currentTask: intent,
      runStatus: "running",
      isAgentActive: true,
    }),

  applyRunState: (s) =>
    set({
      runId: s.runId,
      currentTask: s.intent,
      runStatus: s.status,
      isAgentActive: s.status === "running",
      hitlActive: s.status === "waiting_for_user",
    }),

  setRunCompleted: (status) =>
    set({ isAgentActive: false, hitlActive: false, runStatus: status }),

  updateFromStep: (step) =>
    set((state) => ({
      subgoals: applyStepSubgoals(state.subgoals, step),
      confidenceScore: step.confidence ?? state.confidenceScore,
      isAgentActive: step.status === "running",
      hitlActive: step.status === "waiting_for_user",
      runStatus: step.status,
      stepCount: state.stepCount + 1,
    })),

  setConfidenceScore: (score) => set({ confidenceScore: score }),

  updateFromHitl: (decision) =>
    set({
      hitlActive: true,
      isAgentActive: false,
      pendingDecision: decision,
      runStatus: "waiting_for_user",
    }),

  setPendingDecision: (d) => set({ pendingDecision: d }),

  setDisabledRules: (rules) => set({ disabledRules: rules }),
  setSessionMode: (mode) => set({ sessionMode: mode }),

  reset: () => set(IDLE),
}));
