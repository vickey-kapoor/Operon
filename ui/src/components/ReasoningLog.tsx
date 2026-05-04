import type { StepEvent } from "../hooks/useAgentStream";

type Props = { steps: StepEvent[] };

// ── Confidence badge ──────────────────────────────────────────────────────────

function confStyle(pct: number): { bg: string; color: string; dot: string } {
  if (pct >= 90) return { bg: "#dcfce7", color: "#15803d", dot: "#22c55e" };
  if (pct >= 70) return { bg: "#fef9c3", color: "#854d0e", dot: "#eab308" };
  return { bg: "#fee2e2", color: "#b91c1c", dot: "#ef4444" };
}

function ConfBadge({ confidence }: { confidence: number | undefined }) {
  if (confidence == null) return null;
  const pct = Math.round(confidence * 100);
  const { bg, color, dot } = confStyle(pct);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      fontSize: 10, padding: "2px 7px", borderRadius: 10, fontWeight: 700,
      background: bg, color,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: dot, display: "inline-block" }} />
      {pct}%
    </span>
  );
}

// ── Decision badge ────────────────────────────────────────────────────────────

function DecisionBadge({ ruleName }: { ruleName?: string | null }) {
  const isRule = Boolean(ruleName);
  const label = isRule
    ? ruleName!.replace(/^_/, "").replace(/_rule$/, "").replace(/_/g, " ")
    : "LLM";
  return (
    <span style={{
      fontSize: 9, padding: "2px 6px", borderRadius: 4, fontWeight: 700,
      background: isRule ? "#ede9fe" : "#e0f2fe",
      color: isRule ? "#6d28d9" : "#0369a1",
      letterSpacing: "0.04em", flexShrink: 0,
    }}>
      {isRule ? "RULE" : "LLM"}{isRule ? ` · ${label}` : ""}
    </span>
  );
}

// ── Timestamp ─────────────────────────────────────────────────────────────────

function fmtTime(ts: number | undefined): string {
  if (!ts) return "--:--:--";
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, "0");
  const m = d.getMinutes().toString().padStart(2, "0");
  const s = d.getSeconds().toString().padStart(2, "0");
  return `${h}:${m}:${s}`;
}

// ── Status color ──────────────────────────────────────────────────────────────

function statusColor(st: string): string {
  if (st === "succeeded") return "#15803d";
  if (st === "failed")    return "#b91c1c";
  if (st === "waiting_for_user") return "#d97706";
  return "#6366f1";
}

// ── Thought Card ──────────────────────────────────────────────────────────────

function ThoughtCard({ step }: { step: StepEvent }) {
  const actionLabel = step.action_type ? step.action_type.toUpperCase() : null;

  return (
    <div style={{
      marginBottom: 8,
      background: "#fff",
      border: "1px solid #e5e7eb",
      borderRadius: 8,
      overflow: "hidden",
    }}>
      {/* ── Header row: time · step · decision badge · confidence badge ── */}
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        padding: "7px 10px 6px",
        background: "#fafafa",
        borderBottom: "1px solid #f3f4f6",
        flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 10, color: "#6b7280", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
          {fmtTime(step.receivedAt)}
        </span>
        <span style={{ fontSize: 10, color: "#d1d5db" }}>·</span>
        <span style={{ fontSize: 10, color: "#9ca3af", flexShrink: 0 }}>
          Step {step.step}
        </span>
        <div style={{ flex: 1 }} />
        <DecisionBadge ruleName={step.rule_name} />
        <ConfBadge confidence={step.confidence} />
      </div>

      {/* ── Body ── */}
      <div style={{ padding: "8px 10px", display: "flex", flexDirection: "column", gap: 5 }}>

        {/* Perception row */}
        {step.perception ? (
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
              <ellipse cx="6" cy="6" rx="5" ry="3.5" stroke="#9ca3af" strokeWidth="1.2"/>
              <circle cx="6" cy="6" r="1.5" fill="#9ca3af"/>
            </svg>
            <span style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.4, flex: 1 }}>
              {step.perception}
            </span>
          </div>
        ) : step.subgoal ? (
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
              <path d="M2 6h8M6 2l4 4-4 4" stroke="#9ca3af" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.4, fontStyle: "italic", flex: 1 }}>
              {step.subgoal}
            </span>
          </div>
        ) : null}

        {/* Decision row */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
          {actionLabel && (
            <span style={{
              fontSize: 9, padding: "2px 6px", borderRadius: 4, fontWeight: 700,
              background: "#f0f4ff", color: "#4338ca", letterSpacing: "0.04em",
              flexShrink: 0, marginTop: 1,
            }}>
              {actionLabel}
            </span>
          )}
          {step.rationale && (
            <span style={{
              fontSize: 11, color: "#6b7280", lineHeight: 1.4, flex: 1,
              overflow: "hidden", display: "-webkit-box",
              WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
            }}>
              {step.rationale}
            </span>
          )}
        </div>

        {/* Status pill */}
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <span style={{
            fontSize: 9, color: statusColor(step.status), fontWeight: 700,
            letterSpacing: "0.05em", textTransform: "uppercase",
          }}>
            {step.status}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function ReasoningLog({ steps }: Props) {
  const visible = steps.slice(-30).reverse();

  return (
    <div>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12,
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#374151", letterSpacing: "0.07em" }}>
          REASONING LOG
        </span>
        {steps.length > 0 && (
          <span style={{ fontSize: 10, color: "#9ca3af" }}>{steps.length} steps</span>
        )}
      </div>

      {visible.length === 0 ? (
        <div style={{ fontSize: 11, color: "#9ca3af", fontStyle: "italic", padding: "8px 0" }}>
          Waiting for steps…
        </div>
      ) : (
        visible.map((step) => (
          <ThoughtCard key={`${step.run_id}-${step.step}`} step={step} />
        ))
      )}
    </div>
  );
}
