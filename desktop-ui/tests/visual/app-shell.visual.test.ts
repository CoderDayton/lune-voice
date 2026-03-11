import { afterAll, afterEach, beforeAll, beforeEach, describe, test } from "bun:test";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { expectLocatorToMatchSnapshot } from "./helpers/snapshot";
import { PREVIEW_URL, startPreviewServer, stopPreviewServer, type PreviewServer } from "./helpers/server";

const VIEWPORT = { width: 1512, height: 982 };
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

  test("home page", async () => {
    await openRoute("/");
    await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "home");
  });

  test("dictionary page", async () => {
    await openRoute("/dictionary");
    await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "dictionary");
  });

  test("delete confirmation modal", async () => {
    await openRoute("/dictionary");
    await page.getByRole("button", { name: "Remove" }).first().click();
    await page.getByRole("dialog").waitFor();
    await expectLocatorToMatchSnapshot(page.locator('[role="dialog"]'), "delete-confirmation-modal");
  });
});

async function openRoute(pathname: string) {
  await page.goto(new URL(pathname, PREVIEW_URL).toString(), { waitUntil: "networkidle" });
  await page.addStyleTag({ content: MOTION_RESET });
  await page.locator(".app-shell-root").waitFor({ state: "visible" });
}
