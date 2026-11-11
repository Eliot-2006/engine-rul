import { defineConfig } from "vite";

// GitHub Pages serves the site from https://<user>.github.io/engine-rul/, so every asset and data
// URL is prefixed with this base. Override with DASHBOARD_BASE=/ for a root deployment.
export default defineConfig({
  base: process.env.DASHBOARD_BASE ?? "/engine-rul/",
  build: { target: "es2022", sourcemap: false },
});
