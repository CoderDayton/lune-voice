import { afterAll, beforeAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { setTimeout as sleep } from "node:timers/promises";

const ROOT_DIR = resolve(import.meta.dir, "../../..");
const APP_BINARY = resolve(
  ROOT_DIR,
  "target/debug",
  `lune-ui${process.platform === "win32" ? ".exe" : ""}`,
);
const DRIVER_URL = "http://127.0.0.1:4444";
const DEFAULT_DRIVER_BIN = process.platform === "win32" ? "tauri-driver.exe" : "tauri-driver";
setDefaultTimeout(120_000);

const ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf";

type WebDriverResponse<T> = {
  value: T;
};

type WebDriverElement = {
  [ELEMENT_KEY]: string;
};

let driverProcess: ChildProcessWithoutNullStreams | null = null;
let sessionId: string | null = null;

describe("tauri smoke", () => {
  beforeAll(async () => {
    assertNativeWebDriverAvailable();
    expect(existsSync(APP_BINARY), `missing built Tauri app binary at ${APP_BINARY}`).toBe(true);

    driverProcess = spawn(resolveDriverBinary(), [], {
      stdio: ["ignore", "inherit", "inherit"],
      env: {
        ...process.env,
        // Allow WebKitWebDriver to run without a compositor (headless-friendly).
        WEBKIT_DISABLE_COMPOSITING_MODE: "1",
      },
    });

    driverProcess.once("error", (error) => {
      throw error;
    });

    await waitForDriver();
    sessionId = await createSession();
  });

  afterAll(async () => {
    if (sessionId) {
      await webdriverDelete(`/session/${sessionId}`).catch(() => {});
      sessionId = null;
    }

    if (driverProcess) {
      driverProcess.kill();
      driverProcess = null;
    }
  });

  test("launches the main window and renders the shell", async () => {
    expect(sessionId).not.toBeNull();

    const title = await webdriverGet<string>(`/session/${sessionId}/title`);
    expect(title).toBe("Lune Voice");

    const appShell = await waitForElement(sessionId!, "css selector", ".app-shell-root");
    expect(appShell).not.toBeNull();

    const mainContent = await webdriverFindElement(sessionId!, "css selector", "main#main-content");
    expect(mainContent).not.toBeNull();
  });
});

function resolveDriverBinary() {
  const envPath = process.env.TAURI_DRIVER_BIN?.trim();
  if (envPath) return envPath;

  const localCargoBin = resolve(homedir(), ".cargo/bin", DEFAULT_DRIVER_BIN);
  if (existsSync(localCargoBin)) return localCargoBin;

  return DEFAULT_DRIVER_BIN;
}

function assertNativeWebDriverAvailable() {
  if (process.platform !== "linux") return;

  const configured = process.env.WEBKIT_WEBDRIVER_BIN?.trim();
  if (configured && existsSync(configured)) return;

  const result = spawnSync("sh", ["-lc", "command -v WebKitWebDriver"], {
    encoding: "utf8",
  });

  if (result.status === 0 && result.stdout.trim()) {
    return;
  }

  throw new Error(
    [
      "Linux Tauri smoke tests require `WebKitWebDriver` in PATH.",
      "Install the `webkit2gtk-driver` package, or set `WEBKIT_WEBDRIVER_BIN` to the binary path.",
    ].join(" "),
  );
}

async function waitForDriver(timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${DRIVER_URL}/status`);
      if (response.ok) {
        return;
      }
    } catch {
      // Retry until the driver opens its port.
    }

    await sleep(250);
  }

  throw new Error(`timed out waiting for tauri-driver at ${DRIVER_URL}`);
}

async function createSession() {
  const response = await fetch(`${DRIVER_URL}/session`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      capabilities: {
        alwaysMatch: {
          browserName: "wry",
          "tauri:options": {
            application: APP_BINARY,
          },
          "webkitgtk:browserOptions": {
            // Run without a visible window when possible.
            args: ["--headless"],
          },
        },
      },
    }),
  });

  const payload = (await response.json()) as WebDriverResponse<{
    sessionId?: string;
    capabilities?: Record<string, unknown>;
  }>;

  if (!response.ok) {
    throw new Error(`failed to create Tauri session: ${JSON.stringify(payload)}`);
  }

  const id = payload.value.sessionId;
  if (!id) {
    throw new Error(`tauri-driver response did not include a session id: ${JSON.stringify(payload)}`);
  }

  return id;
}

async function webdriverGet<T>(path: string): Promise<T> {
  const payload = await webdriverRequest<T>("GET", path);
  return payload.value;
}

async function webdriverDelete(path: string) {
  await webdriverRequest("DELETE", path);
}

async function webdriverRequest<T = unknown>(method: string, path: string, body?: unknown) {
  const response = await fetch(`${DRIVER_URL}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  const rawBody = await response.text();
  const payload = rawBody ? (JSON.parse(rawBody) as WebDriverResponse<T>) : undefined;
  if (!response.ok) {
    throw new Error(`webdriver ${method} ${path} failed: ${rawBody || "<empty response>"}`);
  }

  return (payload ?? { value: undefined }) as WebDriverResponse<T>;
}

async function waitForElement(session: string, using: string, value: string, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      return await webdriverFindElement(session, using, value);
    } catch {
      // App shell may still be mounting.
    }

    await sleep(250);
  }

  throw new Error(`timed out waiting for element ${value}`);
}

async function webdriverFindElement(session: string, using: string, value: string) {
  const payload = await webdriverRequest<WebDriverElement>("POST", `/session/${session}/element`, {
    using,
    value,
  });

  const element = payload.value;
  if (!element || typeof element[ELEMENT_KEY] !== "string") {
    throw new Error(`webdriver did not return an element for ${value}`);
  }

  return element;
}
