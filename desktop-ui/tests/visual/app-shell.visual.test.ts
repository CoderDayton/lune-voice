import { afterAll, beforeAll, describe, test } from "bun:test";
import { chromium, type Browser, type Page } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { expectLocatorToMatchSnapshot } from "./helpers/snapshot";
import { PREVIEW_URL, startPreviewServer, stopPreviewServer, type PreviewServer } from "./helpers/server";

const THIS_DIR = dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = join(THIS_DIR, "__screenshots__");

const AUTH_STORAGE_KEY = "lune.desktop.auth";
const MOTION_RESET = `
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
    caret-color: transparent !important;
  }
`;

const VIEWPORTS = {
  mobile: { width: 375, height: 812 },
  tablet: { width: 768, height: 1024 },
  desktop: { width: 1280, height: 900 },
  wide: { width: 1512, height: 982 },
} as const;

const AUTHED_ROUTES = [
  { path: "/", name: "home" },
  { path: "/dictionary", name: "dictionary" },
  { path: "/history", name: "history" },
  { path: "/recording", name: "recording" },
  { path: "/shortcuts", name: "shortcuts" },
];

let previewServer: PreviewServer;
let browser: Browser;

beforeAll(async () => {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  previewServer = await startPreviewServer();
  browser = await chromium.launch({ headless: true });
});

afterAll(async () => {
  await browser.close();
  await stopPreviewServer(previewServer);
});


async function createPage(viewport: { width: number; height: number }) {
  const context = await browser.newContext({
    colorScheme: "dark",
    deviceScaleFactor: 1,
    viewport,
  });
  const page = await context.newPage();
  await page.emulateMedia({ reducedMotion: "reduce" });
  return { page, context };
}

async function openRoute(page: Page, pathname: string) {
  await page.goto(new URL(pathname, PREVIEW_URL).toString(), { waitUntil: "networkidle" });
  await page.addStyleTag({ content: MOTION_RESET });
}

async function authenticate(page: Page) {
  await page.addInitScript(
    ([storageKey, profile]) => {
      window.localStorage.setItem(storageKey, JSON.stringify(profile));
    },
    [AUTH_STORAGE_KEY, { email: "preview@lune.local", name: "Preview user" }],
  );
}

async function saveScreenshot(page: Page, name: string) {
  const buffer = await page.screenshot({ fullPage: true, animations: "disabled" });
  writeFileSync(join(SCREENSHOT_DIR, `${name}.png`), buffer);
}


describe("baseline snapshots", () => {
  test("auth entry", async () => {
    const { page, context } = await createPage(VIEWPORTS.wide);
    try {
      await openRoute(page, "/");
      await page.locator(".auth-root").waitFor({ state: "visible" });
      await expectLocatorToMatchSnapshot(page.locator(".auth-root"), "auth-entry");
    } finally {
      await context.close();
    }
  });

  test("home page", async () => {
    const { page, context } = await createPage(VIEWPORTS.wide);
    try {
      await authenticate(page);
      await openRoute(page, "/");
      await page.locator(".app-shell-root").waitFor({ state: "visible" });
      await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "home");
    } finally {
      await context.close();
    }
  });

  test("dictionary page", async () => {
    const { page, context } = await createPage(VIEWPORTS.wide);
    try {
      await authenticate(page);
      await openRoute(page, "/dictionary");
      await page.locator(".app-shell-root").waitFor({ state: "visible" });
      await expectLocatorToMatchSnapshot(page.locator(".app-shell-root"), "dictionary");
    } finally {
      await context.close();
    }
  });

  test("delete confirmation modal", async () => {
    const { page, context } = await createPage(VIEWPORTS.wide);
    try {
      await authenticate(page);
      await openRoute(page, "/dictionary");
      await page.locator(".app-shell-root").waitFor({ state: "visible" });
      await page.getByRole("button", { name: "Remove" }).first().click();
      await page.getByRole("dialog").waitFor();
      await expectLocatorToMatchSnapshot(page.locator('[role="dialog"]'), "delete-confirmation-modal");
    } finally {
      await context.close();
    }
  });
});


describe("responsive screenshots", () => {
  // Auth screen (unauthenticated) at all viewports
  for (const [vpName, vp] of Object.entries(VIEWPORTS)) {
    test(`auth @ ${vpName} (${vp.width}x${vp.height})`, async () => {
      const { page, context } = await createPage(vp);
      try {
        await openRoute(page, "/");
        await page.locator(".auth-root").waitFor({ state: "visible", timeout: 8_000 });
        await saveScreenshot(page, `auth-${vpName}`);
      } finally {
        await context.close();
      }
    });
  }

  // Authenticated pages at each viewport
  for (const route of AUTHED_ROUTES) {
    for (const [vpName, vp] of Object.entries(VIEWPORTS)) {
      test(`${route.name} @ ${vpName} (${vp.width}x${vp.height})`, async () => {
        const { page, context } = await createPage(vp);
        try {
          await authenticate(page);
          await openRoute(page, route.path);
          await page.locator(".app-shell-root").waitFor({ state: "visible", timeout: 8_000 });
          await page.waitForTimeout(300);
          await saveScreenshot(page, `${route.name}-${vpName}`);
        } finally {
          await context.close();
        }
      });
    }
  }

  // Mobile nav drawer
  test("mobile nav drawer open", async () => {
    const { page, context } = await createPage(VIEWPORTS.mobile);
    try {
      await authenticate(page);
      await openRoute(page, "/");
      await page.locator(".app-shell-root").waitFor({ state: "visible", timeout: 8_000 });
      const menuBtn = page.getByRole("button", { name: "Open navigation menu" });
      await menuBtn.waitFor({ state: "visible", timeout: 3_000 });
      await menuBtn.click();
      await page.waitForTimeout(400);
      await saveScreenshot(page, "mobile-nav-open");
    } finally {
      await context.close();
    }
  });

  // Delete modal at mobile and desktop
  for (const vpName of ["mobile", "desktop"] as const) {
    test(`delete modal @ ${vpName}`, async () => {
      const { page, context } = await createPage(VIEWPORTS[vpName]);
      try {
        await authenticate(page);
        await openRoute(page, "/dictionary");
        await page.locator(".app-shell-root").waitFor({ state: "visible", timeout: 8_000 });
        const removeBtn = page.getByRole("button", { name: "Remove" }).first();
        await removeBtn.waitFor({ state: "visible", timeout: 3_000 });
        await removeBtn.click();
        await page.getByRole("dialog").waitFor({ timeout: 3_000 });
        await page.waitForTimeout(200);
        await saveScreenshot(page, `delete-modal-${vpName}`);
      } finally {
        await context.close();
      }
    });
  }
});
