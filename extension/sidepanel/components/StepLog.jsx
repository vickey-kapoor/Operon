import { useEffect, useRef } from "react";

export default function StepLog({ steps }) {
  const bot = useRef(null);
  useEffect(() => bot.current?.scrollIntoView({ behavior: "smooth" }), [steps]);

  return (
    <div style={{ flex:1, overflowY:"auto", border:"1px solid #222", borderRadius:"8px",
      padding:"8px", background:"#0d0d18", display:"flex", flexDirection:"column", gap:"8px" }}>
      {steps.length === 0 && <div style={{ color:"#555", fontSize:"12px", textAlign:"center", marginTop:"12px" }}>Steps appear here once the agent starts.</div>}
      {steps.map((s, i) => (
        <div key={i} style={{ background:"#161625", border:"1px solid #252535", borderRadius:"6px", padding:"8px", fontSize:"12px", lineHeight:1.5 }}>
          {s.step === -1
            ? <div style={{ color:"#e74c3c", fontWeight:"700" }}>{s.observation}</div>
            : <>
                <div style={{ color:"#7c9ef8", fontWeight:"700" }}>Step {s.step}</div>
                <div style={{ color:"#ccc" }}>{s.observation}</div>
                {s.actions?.length > 0 && (
                  <div style={{ color:"#89d9a0", marginTop:"4px" }}>
                    {s.actions.map((a, j) => <div key={j} style={{ marginLeft:"10px" }}>• {a.type}{a.description ? ` — ${a.description}` : ""}</div>)}
                  </div>
                )}
                {s.done && <div style={{ color:"#2ecc71", fontWeight:"700", marginTop:"4px" }}>✓ {s.result || "Done"}</div>}
              </>}
        </div>
      ))}
      <div ref={bot} />
    </div>
  );
}
