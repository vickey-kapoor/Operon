import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Output directly into extension/sidepanel/dist/
    outDir: "dist",
    emptyOutDir: true,
    // Chrome extension CSP forbids eval — Vite's default dev transform uses it.
    // In production builds this is already disabled; set it explicitly.
    minify: "esbuild",
    rollupOptions: {
      input: "index.html",
    },
  },
  // No server-side dynamic imports needed for extension context.
  base: "./",
});
