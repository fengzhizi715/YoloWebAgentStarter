import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Konva's package main points at its Node adapter, which requires the
  // optional native canvas package. Browser UI and jsdom tests use this build.
  resolve: { alias: [{ find: /^konva$/, replacement: "konva/lib/index.js" }] },
  server: { host: "127.0.0.1", port: 5173 },
});
