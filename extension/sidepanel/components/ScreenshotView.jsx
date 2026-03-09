export default function ScreenshotView({ src }) {
  if (!src) return null;
  return (
    <div style={{ border:"1px solid #222", borderRadius:"8px", overflow:"hidden",
      maxHeight:"180px", background:"#0d0d18", flexShrink:0 }}>
      <img src={src} alt="screenshot" style={{ width:"100%", objectFit:"contain", display:"block" }} />
    </div>
  );
}
