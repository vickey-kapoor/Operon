import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The API requires a key. Unless ui/.env.local sets VITE_OPERON_API_KEY, use the one
// the API generated into .var/api-key (OPERON_RUNTIME_ROOT moves it), so start the
// API before the dev server or the build.
function localApiKey(mode: string): string {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  if (env.VITE_OPERON_API_KEY) return env.VITE_OPERON_API_KEY;
  const runtimeRoot = process.env.OPERON_RUNTIME_ROOT ?? resolve(__dirname, "..", ".var");
  try {
    return readFileSync(resolve(runtimeRoot, "api-key"), "utf-8").trim();
  } catch {
    return "";
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  define: {
    "import.meta.env.VITE_OPERON_API_KEY": JSON.stringify(localApiKey(mode)),
  },
  build: {
    target: ["es2022", "chrome120"],
    minify: !process.env.TAURI_DEBUG ? "oxc" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
}));
