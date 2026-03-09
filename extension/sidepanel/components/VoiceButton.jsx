import { useState, useRef } from "react";

const SR = typeof window !== "undefined"
  ? (window.SpeechRecognition || window.webkitSpeechRecognition)
  : null;

export default function VoiceButton({ onTranscript, disabled }) {
  const [on, setOn] = useState(false);
  const ref = useRef(null);

  const toggle = () => {
    if (!SR) { alert("Speech recognition not supported."); return; }
    if (on) { ref.current?.stop(); setOn(false); return; }
    const r = new SR();
    r.continuous = false; r.interimResults = true; r.lang = "en-US";
    r.onresult = (e) => {
      let t = "";
      for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
      onTranscript(t);
    };
    r.onend = () => setOn(false);
    r.onerror = () => setOn(false);
    ref.current = r; r.start(); setOn(true);
  };

  return (
    <button onClick={toggle} disabled={disabled && !on} title={on ? "Stop" : "Voice input"}
      style={{ padding:"8px 10px", borderRadius:"6px", border:`2px solid ${on?"#e74c3c":"#333"}`,
        background:on?"#3a0a0a":"#1a1a24", color:on?"#e74c3c":"#aaa", cursor:"pointer", fontSize:"18px" }}>
      {on ? "🔴" : "🎤"}
    </button>
  );
}
