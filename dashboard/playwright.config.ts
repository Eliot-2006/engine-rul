import { defineConfig, devices } from "@playwright/test";

// Browser tests run against the production build (`npm run build`) served by `vite preview` under
// the same /engine-rul/ base path as GitHub Pages. Chromium only: desktop for everything, a
// 375-px touch phone for the tests tagged @mobile.
// E2E_BASE_URL=https://eliot-2006.github.io/engine-rul/ runs the same tests against the deployed site.
const PORT = 4173;
const remote = process.env.E2E_BASE_URL;
export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: remote ?? `http://localhost:${PORT}/engine-rul/`, trace: "retain-on-failure" },
  webServer: remote
    ? undefined
    : { command: `npx vite preview --port ${PORT} --strictPort`, url: `http://localhost:${PORT}/engine-rul/`, reuseExistingServer: !process.env.CI },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    {
      name: "mobile",
      grep: /@mobile/,
      use: { ...devices["Desktop Chrome"], viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
    },
  ],
});
