import { useEffect, useRef } from "react";
import type { ActionIntentEvent } from "../hooks/useAgentStream";

type Props = {
  registerFrameCallback: (cb: (blob: Blob) => void) => () => void;
  actionIntent: ActionIntentEvent | null;
  onInput: (msg: object) => void;
  inputMode: "click" | "type" | "scroll";
  typeText: string;
};

const SVG_NS = "http://www.w3.org/2000/svg";
const TRAIL_MAX = 6;
const NATIVE_FALLBACK_W = 1920;
const NATIVE_FALLBACK_H = 1080;

/** Inject global CSS keyframes once into the document head. */
function ensureKeyframes() {
  const id = "operon-keyframes";
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = `
    @keyframes operon-pulse {
      0%   { transform: scale(1);   opacity: 0.85; }
      100% { transform: scale(9);   opacity: 0; }
    }
    @keyframes operon-dot-pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.5; }
    }
    @keyframes operon-ping {
      0%   { transform: scale(1); opacity: 0.6; }
      100% { transform: scale(2); opacity: 0; }
    }
    @keyframes operon-bar-wave {
      0%   { transform: scaleY(0.6); opacity: 0.7; }
      100% { transform: scaleY(1.4); opacity: 1; }
    }
    @keyframes operon-wave-flow {
      0%   { stroke-dashoffset: 0; }
      100% { stroke-dashoffset: -160; }
    }
  `;
  document.head.appendChild(style);
}

/** Add a glowing amber pulse animation at (cx, cy) in SVG viewBox coords. */
function addPulse(svg: SVGSVGElement, cx: number, cy: number) {
  // Outer group translated to the click center — so transform-origin works at local (0,0).
  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("transform", `translate(${cx} ${cy})`);

  // Inner static dot
  const dot = document.createElementNS(SVG_NS, "circle");
  dot.setAttribute("cx", "0");
  dot.setAttribute("cy", "0");
  dot.setAttribute("r", "7");
  dot.setAttribute("fill", "#f59e0b");
  dot.style.animation = "operon-dot-pulse 0.6s ease-in-out 3";
  g.appendChild(dot);

  // Three concentric rings with staggered delays
  for (let i = 0; i < 3; i++) {
    const ring = document.createElementNS(SVG_NS, "circle");
    ring.setAttribute("cx", "0");
    ring.setAttribute("cy", "0");
    ring.setAttribute("r", "7");
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "#f59e0b");
    ring.setAttribute("stroke-width", "2.5");
    ring.style.animation = `operon-pulse 1.3s ease-out ${i * 0.22}s forwards`;
    ring.style.transformOrigin = "0px 0px";
    g.appendChild(ring);
  }

  svg.appendChild(g);
  setTimeout(() => { if (svg.contains(g)) svg.removeChild(g); }, 2200);
}

/** Redraw the ghost trail connecting the last N click positions. */
function updateGhostTrail(svg: SVGSVGElement, trail: [number, number][]) {
  const old = svg.querySelector("#operon-trail");
  if (old) svg.removeChild(old);
  if (trail.length < 2) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.id = "operon-trail";

  const d = trail.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", d);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#f59e0b");
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-dasharray", "6 5");
  path.setAttribute("opacity", "0.45");
  g.appendChild(path);

  trail.slice(0, -1).forEach(([x, y], i) => {
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", String(x));
    c.setAttribute("cy", String(y));
    c.setAttribute("r", "4");
    c.setAttribute("fill", "#f59e0b");
    c.setAttribute("opacity", String(0.15 + (i / trail.length) * 0.3));
    g.appendChild(c);
  });

  svg.insertBefore(g, svg.firstChild);
}

export function LiveMirror({
  registerFrameCallback, actionIntent, onInput, inputMode, typeText,
}: Props) {
  const boxRef  = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef  = useRef<SVGSVGElement>(null);
  const trailRef = useRef<[number, number][]>([]);
  const intentSeen = useRef<ActionIntentEvent | null>(null);

  // Inject CSS keyframes once.
  useEffect(() => { ensureKeyframes(); }, []);

  // Sync canvas resolution to container size via ResizeObserver.
  useEffect(() => {
    const box = boxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = box.clientWidth;
      canvas.height = box.clientHeight;
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, []);

  // Register the frame callback — draws directly to canvas, zero React re-renders.
  useEffect(() => {
    return registerFrameCallback(async (blob: Blob) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      try {
        const bm = await createImageBitmap(blob);
        ctx.drawImage(bm, 0, 0, canvas.width, canvas.height);
        bm.close();
      } catch { /* ignore decode errors */ }
    });
  }, [registerFrameCallback]);

  // When actionIntent changes: add pulse + update ghost trail imperatively.
  useEffect(() => {
    if (!actionIntent || actionIntent === intentSeen.current) return;
    intentSeen.current = actionIntent;

    const svg = svgRef.current;
    if (!svg || actionIntent.x == null || actionIntent.y == null) return;

    const x = actionIntent.x;
    const y = actionIntent.y;

    trailRef.current = [...trailRef.current.slice(-(TRAIL_MAX - 1)), [x, y]];
    updateGhostTrail(svg, trailRef.current);

    const at = actionIntent.action_type.toLowerCase();
    if (["click", "dblclick", "type"].includes(at)) {
      addPulse(svg, x, y);
    }
  }, [actionIntent]);

  // User input: translate canvas click → normalised coords → WS inject_input.
  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x_ratio = (e.clientX - rect.left) / rect.width;
    const y_ratio = (e.clientY - rect.top) / rect.height;
    if (inputMode === "click") {
      onInput({ type: "inject_input", input_type: "click", x_ratio, y_ratio, button: e.button === 2 ? "right" : "left" });
    } else if (inputMode === "type" && typeText) {
      onInput({ type: "inject_input", input_type: "type", x_ratio, y_ratio, text: typeText });
    } else if (inputMode === "scroll") {
      onInput({ type: "inject_input", input_type: "scroll", x_ratio, y_ratio, delta_y: e.shiftKey ? -300 : 300 });
    }
  };

  return (
    <div
      ref={boxRef}
      style={{ position: "relative", width: "100%", height: "100%", background: "#0f111a", overflow: "hidden" }}
    >
      {/* Frame canvas */}
      <canvas
        ref={canvasRef}
        onClick={onCanvasClick}
        onContextMenu={(e) => e.preventDefault()}
        style={{
          position: "absolute", inset: 0,
          cursor: inputMode === "click" ? "crosshair" : inputMode === "type" ? "text" : "ns-resize",
          userSelect: "none",
        }}
      />

      {/* SVG overlay — pulse animations + ghost trail */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${NATIVE_FALLBACK_W} ${NATIVE_FALLBACK_H}`}
        preserveAspectRatio="none"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      >
        {/* Pulse + ghost trail elements are added imperatively above */}
      </svg>
    </div>
  );
}

