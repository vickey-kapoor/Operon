import { useMemo } from "react";
import type { StepEvent } from "../hooks/useAgentStream";

export type SubgoalNode = {
  id: string;
  label: string;
  status: "pending" | "active" | "complete" | "failed";
  stepCount: number;
};

type Props = {
  intent: string;
  steps: StepEvent[];
};

const STATUS_META: Record<SubgoalNode["status"], { bg: string; color: string; border: string; label: string }> = {
  active:   { bg: "#eef2ff", color: "#4338ca", border: "#c7d2fe", label: "Active" },
  complete: { bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0", label: "Done" },
  pending:  { bg: "#f9fafb", color: "#6b7280", border: "#e5e7eb", label: "Pending" },
  failed:   { bg: "#fff1f2", color: "#be123c", border: "#fecdd3", label: "Failed" },
};

function deriveSubgoals(steps: StepEvent[]): SubgoalNode[] {
  const seenOrder: string[] = [];
  const stepCounts = new Map<string, number>();
  const finalStatus = new Map<string, "complete" | "failed">();
  let active: string | null = null;

  for (const s of steps) {
    const label = s.subgoal?.trim() || "";
    if (!label) continue;
    if (!stepCounts.has(label)) {
      seenOrder.push(label);
      stepCounts.set(label, 0);
    }
    if (active && active !== label) {
      finalStatus.set(active, s.status === "failed" ? "failed" : "complete");
    }
    active = label;
    stepCounts.set(label, (stepCounts.get(label) ?? 0) + 1);
  }

  return seenOrder.map((label) => ({
    id: label,
    label,
    status: finalStatus.has(label)
      ? finalStatus.get(label)!
      : label === active
        ? "active"
        : "pending",
    stepCount: stepCounts.get(label) ?? 0,
  }));
}

export function SubgoalTree({ intent, steps }: Props) {
  const nodes = useMemo(() => deriveSubgoals(steps), [steps]);
  const doneCount = nodes.filter((n) => n.status === "complete").length;
  const isActive = nodes.some((n) => n.status === "active");

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#374151", letterSpacing: "0.07em" }}>
          SUBGOAL TREE
        </span>
        {nodes.length > 0 && (
          <span style={{
            fontSize: 10, padding: "1px 8px", borderRadius: 10, fontWeight: 600,
            background: isActive ? "#ede9fe" : "#f0fdf4",
            color: isActive ? "#6d28d9" : "#15803d",
          }}>
            {doneCount}/{nodes.length}
          </span>
        )}
      </div>

      {/* Root intent */}
      {intent && (
        <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
          <div style={{
            marginTop: 2, width: 16, height: 16, borderRadius: 4, flexShrink: 0,
            background: "#f0f4ff", border: "1px solid #c7d2fe",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M2 5h6M5 2l3 3-3 3" stroke="#6366f1" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <span style={{ fontSize: 12, fontWeight: 700, color: "#1e2030", lineHeight: 1.4, flex: 1 }}>
            {intent}
          </span>
        </div>
      )}

      {nodes.length === 0 && !intent && (
        <div style={{ fontSize: 11, color: "#9ca3af", padding: "8px 0", fontStyle: "italic" }}>
          No subgoals yet — start a run to see the plan.
        </div>
      )}

      {/* Subgoal list */}
      <div style={{ marginLeft: intent ? 8 : 0 }}>
        {/* Vertical connector */}
        <div style={{ position: "relative" }}>
          {intent && nodes.length > 0 && (
            <div style={{
              position: "absolute", left: 7, top: 0, bottom: 0,
              width: 1.5, background: "#e5e7eb", borderRadius: 1,
            }} />
          )}
          <div style={{ paddingLeft: intent ? 18 : 0 }}>
            {nodes.map((node, idx) => (
              <SubgoalRow key={node.id} node={node} isLast={idx === nodes.length - 1} />
            ))}
            {/* Future steps ghost */}
            {isActive && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", opacity: 0.35 }}>
                <PendingIcon />
                <span style={{ fontSize: 11, color: "#6b7280", fontStyle: "italic" }}>more steps ahead…</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SubgoalRow({ node, isLast }: { node: SubgoalNode; isLast: boolean }) {
  const m = STATUS_META[node.status];
  const isActive = node.status === "active";

  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 8,
      padding: "5px 0",
      marginBottom: isLast ? 0 : 2,
    }}>
      {/* Status icon */}
      <div style={{ flexShrink: 0, marginTop: 1 }}>
        <StatusIcon status={node.status} />
      </div>

      {/* Label + badge */}
      <div style={{
        flex: 1, minWidth: 0,
        padding: isActive ? "4px 8px" : "0",
        background: isActive ? m.bg : "transparent",
        border: isActive ? `1px solid ${m.border}` : "none",
        borderRadius: isActive ? 6 : 0,
        marginLeft: isActive ? -8 : 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
          <span style={{
            fontSize: 12,
            color: isActive ? "#1e2030" : node.status === "failed" ? "#be123c" : "#374151",
            fontWeight: isActive ? 600 : 400,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            flex: 1,
          }}>
            {node.label}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
            {node.stepCount > 0 && (
              <span style={{ fontSize: 9, color: "#9ca3af" }}>{node.stepCount}s</span>
            )}
            <span style={{
              fontSize: 9, padding: "1px 6px", borderRadius: 8, fontWeight: 700,
              background: m.bg, color: m.color, border: `1px solid ${m.border}`,
              letterSpacing: "0.04em",
            }}>
              {m.label}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: SubgoalNode["status"] }) {
  if (status === "complete") {
    return (
      <div style={{
        width: 16, height: 16, borderRadius: "50%", background: "#dcfce7",
        border: "1.5px solid #86efac", display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
          <path d="M2 4.5l2 2 3-3.5" stroke="#16a34a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    );
  }
  if (status === "failed") {
    return (
      <div style={{
        width: 16, height: 16, borderRadius: "50%", background: "#fee2e2",
        border: "1.5px solid #fca5a5", display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
          <path d="M2 2l4 4M6 2L2 6" stroke="#dc2626" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </div>
    );
  }
  if (status === "active") {
    return <PulsingDot />;
  }
  return <PendingIcon />;
}

function PulsingDot() {
  return (
    <div style={{ position: "relative", width: 16, height: 16, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{
        position: "absolute", width: 14, height: 14, borderRadius: "50%",
        background: "#818cf8", opacity: 0.25,
        animation: "operon-ping 1.1s ease-out infinite",
      }} />
      <span style={{
        width: 10, height: 10, borderRadius: "50%", background: "#6366f1",
        display: "block", position: "relative",
      }} />
    </div>
  );
}

function PendingIcon() {
  return (
    <div style={{
      width: 16, height: 16, borderRadius: "50%", background: "#f3f4f6",
      border: "1.5px solid #d1d5db", display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#d1d5db" }} />
    </div>
  );
}
