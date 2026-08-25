type NavItem = { icon: string; label: string; id: string };

const NAV: NavItem[] = [
  { icon: "⊞", label: "Dashboard", id: "dashboard" },
  { icon: "≡", label: "Tasks",     id: "tasks" },
  { icon: "∿", label: "Reasoning", id: "reasoning" },
  { icon: "◇", label: "Moat Builder", id: "moat" },
];

type Props = { active?: string; onNav?: (id: string) => void };

export function Sidebar({ active = "tasks", onNav }: Props) {
  return (
    <aside style={{
      width: 200, flexShrink: 0, height: "100%",
      background: "#1a1d2e", display: "flex", flexDirection: "column",
      borderRight: "1px solid #252840",
    }}>
      {/* Logo */}
      <div style={{
        padding: "18px 16px 14px", borderBottom: "1px solid #252840",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <HexLogo />
        <div>
          <div style={{ color: "#e2e8f0", fontSize: 12, fontWeight: 700, letterSpacing: "0.1em" }}>OPERON</div>
          <div style={{ color: "#64748b", fontSize: 9, letterSpacing: "0.08em", marginTop: 1 }}>COMMAND CENTER</div>
        </div>
      </div>

      {/* Nav items */}
      <nav style={{ flex: 1, padding: "10px 8px" }}>
        {NAV.map((item) => {
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              onClick={() => onNav?.(item.id)}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                width: "100%", padding: "9px 12px", marginBottom: 2,
                background: isActive ? "#2d3256" : "transparent",
                border: "none", borderRadius: 6,
                color: isActive ? "#818cf8" : "#64748b",
                fontSize: 12, fontFamily: "inherit", cursor: "pointer",
                transition: "background 0.12s, color 0.12s",
                textAlign: "left",
              }}
              onMouseEnter={(e) => { if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = "#222540"; }}
              onMouseLeave={(e) => { if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
            >
              <span style={{ fontSize: 14, width: 18, textAlign: "center" }}>{item.icon}</span>
              <span style={{ fontWeight: isActive ? 600 : 400 }}>{item.label}</span>
              {isActive && <span style={{ marginLeft: "auto", width: 3, height: 18, background: "#818cf8", borderRadius: 2 }} />}
            </button>
          );
        })}
      </nav>

      {/* Settings at bottom */}
      <div style={{ padding: "8px 8px 16px" }}>
        {(["settings"] as const).map((id) => {
          const isActive = id === active;
          return (
            <button
              key={id}
              onClick={() => onNav?.(id)}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                width: "100%", padding: "9px 12px",
                background: isActive ? "#2d3256" : "transparent",
                border: "none", borderRadius: 6,
                color: isActive ? "#818cf8" : "#64748b",
                fontSize: 12, fontFamily: "inherit", cursor: "pointer",
                transition: "background 0.12s, color 0.12s",
                textAlign: "left",
              }}
              onMouseEnter={(e) => { if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = "#222540"; }}
              onMouseLeave={(e) => { if (!isActive) (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
            >
              <span style={{ fontSize: 14, width: 18, textAlign: "center" }}>⚙</span>
              <span style={{ fontWeight: isActive ? 600 : 400 }}>Settings</span>
              {isActive && <span style={{ marginLeft: "auto", width: 3, height: 18, background: "#818cf8", borderRadius: 2 }} />}
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function HexLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28">
      <polygon
        points="14,2 25,8 25,20 14,26 3,20 3,8"
        fill="#4338ca"
        stroke="#6366f1"
        strokeWidth="1.5"
      />
      <text x="14" y="18" textAnchor="middle" fill="#e2e8f0" fontSize="11" fontWeight="700" fontFamily="monospace">O</text>
    </svg>
  );
}
