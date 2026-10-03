import { expect, test, type Page } from "@playwright/test";

/**
 * The critical user journeys, through the real UI, real API and real
 * database. Only the model is scripted (backend/devtools/mock_llm.py), which
 * is what makes the answers below exact. Each test starts its own chat.
 *
 * Selectors are the accessible names a user would perceive -- button labels,
 * headings, placeholders -- not test ids, so a test breaks when the UI
 * changes for a user, not when markup is refactored.
 */

const COMPOSER = "Ask about product, growth, retention, pricing, hiring...";

async function newChat(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "+ New chat" }).click();
  await expect(page.getByRole("heading", { name: "Ask anything about product and growth" })).toBeVisible();
}

async function ask(page: Page, question: string) {
  const composer = page.getByPlaceholder(COMPOSER);
  await composer.fill(question);
  await composer.press("Enter");
}

/** The latest assistant reply's source list, opened. */
async function openSources(page: Page) {
  const toggle = page.getByRole("button", { name: /sources? from Lenny's Podcast/ }).last();
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  return toggle.locator("xpath=following-sibling::ol[1]");
}

test("start a session", async ({ page }) => {
  await newChat(page);

  const current = page.getByRole("navigation", { name: "Chat sessions" }).locator('[aria-current="page"]');
  await expect(current).toContainText("New chat");
  await expect(current).toContainText("0 messages");
  await expect(page.getByPlaceholder(COMPOSER)).toBeEnabled();
});

test("ask a question and see the answer with its citation", async ({ page }) => {
  await newChat(page);
  await ask(page, "How do I improve user retention through onboarding?");

  await expect(page.getByText("Adam Fishman argues that onboarding is the lever for retention [1].")).toBeVisible();
  const sources = await openSources(page);
  await expect(sources.getByRole("listitem")).toHaveCount(1);
  await expect(sources.getByRole("listitem")).toContainText("[1]");
  await expect(sources.getByRole("listitem")).toContainText("Adam Fishman");
  await expect(sources.getByRole("link")).toHaveAttribute("href", /youtube\.com\/watch\?v=fixture-fishman/);

  // The first question names the chat in the sidebar.
  await expect(
    page.getByRole("navigation", { name: "Chat sessions" }).locator('[aria-current="page"]'),
  ).toContainText("How do I improve user retention");
});

test("asking about a named guest searches only that guest", async ({ page }) => {
  await newChat(page);
  await ask(page, "What does Casey Winters say about growth loops compounding?");

  await expect(page.getByText(/^Casey Winters argues/)).toBeVisible();
  const sources = await openSources(page);
  for (const item of await sources.getByRole("listitem").all()) {
    await expect(item).toContainText("Casey Winters");
  }
});

test("asking about a period searches only that period", async ({ page }) => {
  await newChat(page);
  await ask(page, "What was said about investing in new acquisition channels since 2024?");

  // Only Adam Grenier's episode is from 2024 in the fixture corpus.
  await expect(page.getByText(/^Adam Grenier argues/)).toBeVisible();
  const sources = await openSources(page);
  for (const item of await sources.getByRole("listitem").all()) {
    await expect(item).toContainText("Adam Grenier");
  }
});

test("asking for a document renders it in the viewer", async ({ page }) => {
  await newChat(page);
  await ask(page, "Write a checklist document on how onboarding improves user retention");

  await expect(page.getByText("I put a one-page onboarding checklist in the panel beside the chat.")).toBeVisible();
  const viewer = page.getByRole("complementary", { name: "Artifact viewer" });
  await expect(viewer).toBeVisible();
  await expect(viewer.getByRole("heading", { name: "Onboarding checklist" }).last()).toBeVisible();
  await expect(viewer.getByRole("listitem").first()).toContainText("Start with the first session");

  // The source tab shows the raw markdown, not a render of it.
  await viewer.getByRole("tab", { name: /source/i }).click();
  await expect(viewer.locator("pre")).toContainText("# Onboarding checklist");

  // The chip on the reply reopens it after closing.
  await viewer.getByRole("button", { name: "Close artifact viewer" }).click();
  await expect(viewer).toBeHidden();
  await page.getByRole("button", { name: "Onboarding checklist" }).click();
  await expect(page.getByRole("complementary", { name: "Artifact viewer" })).toBeVisible();
});

test("an off-topic question is refused with no sources", async ({ page }) => {
  await newChat(page);
  await ask(page, "What is the best sourdough starter recipe?");

  await expect(page.getByText(/transcripts don't cover that/)).toBeVisible();
  await expect(page.getByRole("button", { name: /sources? from Lenny's Podcast/ })).toHaveCount(0);
});
