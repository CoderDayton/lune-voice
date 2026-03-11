import { afterAll, afterEach, beforeAll, beforeEach, describe, test } from "bun:test";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { expectLocatorToMatchSnapshot } from "./helpers/snapshot";
import { PREVIEW_URL, startPreviewServer, stopPreviewServer, type PreviewServer } from "./helpers/server";

const VIEWPORT = { width: 1512, height: 982 };
const AUTH_STORAGE_KEY = "lune.desktop.auth";
const MOTION_RESET = `
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
    caret-color: transparent !important;
  }
`;

let previewServer: PreviewServer;
let browser: Browser;
let context: BrowserContext;
let page: Page;

describe("desktop-ui visual snapshots", () => {
  beforeAll(async () => {
    previewServer = await startPreviewServer();
    browser = await chromium.launch({ headless: true });
  });

  beforeEach(async () => {
    context = await browser.newContext({
      colorScheme: "dark",
      deviceScaleFactor: 1,
      viewport: VIEWPORT,
    });
    page = await context.newPage();
    await page.emulateMedia({ reducedMotion: "reduce" });
  });

  afterEach(async () => {
    await context.close();
  });

  afterAll(async () => {
    await browser.close();
    await stopPreviewServer(previewServer);
  });

  test("auth entry", async () => {
    await openRoute("/");
    await page.locator(".auth-root").waitFor({ state: "visible" });
    await expectLocatorToMatchSnapshot(page.locator(".auth-root"), "auth-entry");
  });

  test("home page", async () => {
    await authenticatePreviewSession();
    await openRoute("/");
    await page.locator(".app-shell-root").waitFor({ state: "visible" });
    await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "home");
  });

  test("dictionary page", async () => {
    await authenticatePreviewSession();
    await openRoute("/dictionary");
    await page.locator(".app-shell-root").waitFor({ state: "visible" });
    await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "dictionary");
  });

  test("delete confirmation modal", async () => {
    await authenticatePreviewSession();
    await openRoute("/dictionary");
    await page.locator(".app-shell-root").waitFor({ state: "visible" });
    await page.getByRole("button", { name: "Remove" }).first().click();
    await page.getByRole("dialog").waitFor();
    await expectLocatorToMatchSnapshot(page.locator('[role="dialog"]'), "delete-confirmation-modal");
  });
});

async function openRoute(pathname: string) {
  await page.goto(new URL(pathname, PREVIEW_URL).toString(), { waitUntil: "networkidle" });
  await page.addStyleTag({ content: MOTION_RESET });
}

async function authenticatePreviewSession() {
  await page.addInitScript(([storageKey, profile]) => {
    window.localStorage.setItem(storageKey, JSON.stringify(profile));
  }, [AUTH_STORAGE_KEY, { email: "preview@lune.local", name: "Preview user" }]);
}
