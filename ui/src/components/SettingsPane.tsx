/**
 * SettingsPane — shown in the Task Intelligence slot when "Moat Builder" is active.
 *
 * Sections:
 *   1. Rule Manager       — toggle engine primitives on/off
 *   2. Session Persistence — Fresh Playwright vs observable
 */

import { useAgentStore } from "../store/agentStore";

type Props = {
  sendControl: (msg: object) => void;
};

// ── Engine rule catalogue ─────────────────────────────────────────────────────

const ENGINE_RULES: { name: string; label: string; desc: string; danger?: boolean }[] = [
  {
    name: "_human_intervention_rule",
    label: "Human Intervention",
    desc: "Pauses for CAPTCHA, login walls, 2FA, and bot-detection pages",
    danger: true,
  },
  {
    name: "_task_success_stop_rule",
    label: "Task Success Stop",
    desc: "Ends the run when form submission or task completion is visually confirmed",
    danger: true,
  },
  {
    name: "_prefer_launch_app_rule",
    label: "Prefer Launch App",
    desc: "Launches the target application directly instead of navigating via menus",
  },
  {
    name: "_form_visible_field_fill_rule",
    label: "Form Field Fill",
    desc: "Auto-fills visible form fields (name, email, message) on detected form pages",
  },
  {
    name: "_dropdown_menu_select_rule",
    label: "Dropdown Select",
    desc: "Selects an unselected option when a dropdown menu is open",
  },
  {
    name: "_avoid_identical_type_retry",
    label: "Avoid Identical Type Retry",
    desc: "Re-establishes focus instead of retrying a failed TYPE on the same element",
  },
  {
    name: "_no_progress_recovery_rule",
    label: "No-Progress Recovery",
    desc: "Triggers recovery when no progress is detected after several consecutive steps",
  },
  {
    name: "_dismiss_blocking_overlay_rule",
    label: "Dismiss Blocking Overlay",
    desc: "Dismisses dialogs and banners that obstruct the main content area",
  },
  {
    name: "_search_query_rule",
    label: "Search Query",
    desc: "Uses search bars directly when a search-oriented intent is detected",
  },
];

// ── Toggle switch ─────────────────────────────────────────────────────────────

function Toggle({ on, onChange, disabled }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <div
      role="switch"
      aria-checked={on}
      onClick={() => !disabled && onChange(!on)}
      style={{
        width: 34, height: 19, borderRadius: 10, flexShrink: 0,
        background: on ? "#6366f1" : "#d1d5db",
        position: "relative",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "background 0.18s",
      }}
    >
      <div style={{
        position: "absolute", top: 2.5, left: on ? 15 : 2.5,
        width: 14, height: 14, borderRadius: "50%",
        background: "#fff",
        transition: "left 0.18s",
        boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
      }} />
    </div>
  );
}

// ── Section header ────────────────────────────────────────────────────────────

function SectionHeader({ icon, title, subtitle }: { icon: string; title: string; subtitle?: string }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 13 }}>{icon}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#1e2030", letterSpacing: "0.07em" }}>
          {title}
        </span>
      </div>
      {subtitle && (
        <p style={{ margin: "4px 0 0 21px", fontSize: 10, color: "#9ca3af", lineHeight: 1.4 }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}

// ── Rule Manager section ──────────────────────────────────────────────────────

function RuleManager({ sendControl }: { sendControl: (msg: object) => void }) {
  const disabledRules = useAgentStore((s) => s.disabledRules);
  const setDisabledRules = useAgentStore((s) => s.setDisabledRules);

  const toggle = (name: string, enabled: boolean) => {
    const next = enabled
      ? disabledRules.filter((r) => r !== name)
      : [...disabledRules, name];
    setDisabledRules(next);
    sendControl({ type: "set_disabled_rules", rules: next });
  };

  const enabledCount = ENGINE_RULES.length - disabledRules.length;

  return (
    <div>
      <SectionHeader
        icon="⚡"
        title="RULE MANAGER"
        subtitle="Engine primitives run before the LLM. Disable to let the model decide instead."
      />

      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 8,
      }}>
        <span style={{ fontSize: 10, color: "#6b7280" }}>
          {enabledCount}/{ENGINE_RULES.length} rules active
        </span>
        <button
          onClick={() => {
            setDisabledRules([]);
            sendControl({ type: "set_disabled_rules", rules: [] });
          }}
          style={{
            fontSize: 10, color: "#6366f1", background: "none", border: "none",
            cursor: "pointer", padding: 0, fontFamily: "inherit",
          }}
        >
          Enable all
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {ENGINE_RULES.map((rule) => {
          const isEnabled = !disabledRules.includes(rule.name);
          return (
            <div
              key={rule.name}
              style={{
                display: "flex", alignItems: "flex-start", gap: 10,
                padding: "8px 10px",
                background: isEnabled ? "#fff" : "#f9fafb",
                borderRadius: 6,
                border: `1px solid ${isEnabled ? "#e5e7eb" : "#f3f4f6"}`,
                opacity: isEnabled ? 1 : 0.65,
                transition: "opacity 0.15s, background 0.15s",
              }}
            >
              <Toggle on={isEnabled} onChange={(v) => toggle(rule.name, v)} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "#1e2030" }}>
                    {rule.label}
                  </span>
                  {rule.danger && (
                    <span style={{
                      fontSize: 8, padding: "1px 5px", borderRadius: 4,
                      background: "#fef9c3", color: "#854d0e", fontWeight: 700,
                      letterSpacing: "0.05em",
                    }}>
                      CRITICAL
                    </span>
                  )}
                </div>
                <p style={{ margin: "2px 0 0", fontSize: 10, color: "#6b7280", lineHeight: 1.4 }}>
                  {rule.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Session Persistence section ───────────────────────────────────────────────

const SESSION_MODES = [
  {
    id: "fresh" as const,
    label: "Fresh Playwright Session",
    desc: "Operon launches and manages its own browser instance. Isolated, reproducible.",
  },
  {
    id: "observable" as const,
    label: "Observable Mode",
    desc: "Persistent headed browser — runs are visible on screen and streamed to the Command Center.",
  },
];

function SessionPersistence() {
  const sessionMode = useAgentStore((s) => s.sessionMode);
  const setSessionMode = useAgentStore((s) => s.setSessionMode);

  return (
    <div>
      <SectionHeader
        icon="🔄"
        title="SESSION PERSISTENCE"
        subtitle="Choose how Operon acquires the browser for each run."
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {SESSION_MODES.map((mode) => {
          const active = sessionMode === mode.id;
          return (
            <button
              key={mode.id}
              onClick={() => setSessionMode(mode.id)}
              style={{
                display: "flex", alignItems: "flex-start", gap: 10, width: "100%",
                padding: "10px 12px", textAlign: "left",
                border: `1.5px solid ${active ? "#6366f1" : "#e5e7eb"}`,
                borderRadius: 8, background: active ? "#f0f0ff" : "#fff",
                cursor: "pointer", fontFamily: "inherit",
                transition: "border-color 0.15s, background 0.15s",
              }}
            >
              {/* Radio ring */}
              <div style={{
                marginTop: 1, width: 15, height: 15, borderRadius: "50%", flexShrink: 0,
                border: `2px solid ${active ? "#6366f1" : "#d1d5db"}`,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                {active && (
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#6366f1" }} />
                )}
              </div>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: active ? "#4338ca" : "#374151" }}>
                  {mode.label}
                </div>
                <div style={{ fontSize: 10, color: "#9ca3af", marginTop: 2, lineHeight: 1.4 }}>
                  {mode.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Root component ────────────────────────────────────────────────────────────

export function SettingsPane({ sendControl }: Props) {
  const disabledRules = useAgentStore((s) => s.disabledRules);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "#f8f9fb" }}>
      {/* Header */}
      <div style={{
        padding: "12px 16px 10px",
        background: "#fff", borderBottom: "1px solid #e5e7eb",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: "#374151" }}>
            SETTINGS / MOAT BUILDER
          </div>
          <div style={{ fontSize: 10, color: "#9ca3af", marginTop: 2 }}>
            Deterministic intelligence layer &amp; session config
          </div>
        </div>
        {disabledRules.length > 0 && (
          <span style={{
            fontSize: 10, padding: "2px 8px", borderRadius: 10, fontWeight: 700,
            background: "#fef9c3", color: "#854d0e",
          }}>
            {disabledRules.length} rule{disabledRules.length !== 1 ? "s" : ""} off
          </span>
        )}
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 14px", display: "flex", flexDirection: "column", gap: 24 }}>
        <RuleManager sendControl={sendControl} />

        <Divider />
        <SessionPersistence />

        {/* Bottom padding */}
        <div style={{ height: 8 }} />
      </div>
    </div>
  );
}

function Divider() {
  return <div style={{ height: 1, background: "#e5e7eb", flexShrink: 0 }} />;
}
