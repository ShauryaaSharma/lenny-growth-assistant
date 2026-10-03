import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests for the critical user journeys (e2e/).
 *
 * They need the backend running with the deterministic mock LLM and the
 * fixture corpus -- see e2e/README.md. Playwright starts the frontend itself:
 * the production build in CI, the dev server locally (reusing one that is
 * already up).
 */
const CI = !!process.env.CI;

export default defineConfig({
  testDir: "./e2e",
  // One backend and one database: journeys run one at a time, each in its
  // own new chat, so they cannot see each other's messages.
  workers: 1,
  // No retries: a test that only passes on the second try is reported as
  // failing rather than hidden.
  retries: 0,
  reporter: CI
    ? [["list"], ["html", { open: "never" }], ["junit", { outputFile: "test-results/junit.xml" }]]
    : [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: CI ? "npm run start" : "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !CI,
    timeout: 120_000,
  },
});
