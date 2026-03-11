import { fileURLToPath } from "node:url";

const PROJECT_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PREVIEW_PORT = 4173;

export const PREVIEW_URL = `http://127.0.0.1:${PREVIEW_PORT}`;

export interface PreviewServer {
  process: Bun.Subprocess<"ignore", "ignore", "pipe">;
  url: string;
}

export async function startPreviewServer() {
  const process = Bun.spawn({
    cmd: [
      Bun.which("bun") ?? globalThis.process.execPath,
      "x",
      "vite",
      "preview",
      "--host",
      "127.0.0.1",
      "--port",
      String(PREVIEW_PORT),
      "--strictPort",
    ],
    cwd: PROJECT_ROOT,
    stdout: "ignore",
    stderr: "pipe",
  });

  await waitForPreview(process);

  return { process, url: PREVIEW_URL } satisfies PreviewServer;
}

export async function stopPreviewServer(server: PreviewServer) {
  server.process.kill();
  await server.process.exited;
}

async function waitForPreview(process: Bun.Subprocess<"ignore", "ignore", "pipe">) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (process.exitCode !== null) {
      const stderr = await readProcessOutput(process);
      throw new Error(`Preview server exited before becoming ready.\n${stderr}`);
    }

    try {
      const response = await fetch(PREVIEW_URL);
      if (response.ok) {
        return;
      }
    } catch {
      // Keep polling until the server responds or exits.
    }

    await Bun.sleep(250);
  }

  process.kill();
  const stderr = await readProcessOutput(process);
  throw new Error(`Timed out waiting for preview server at ${PREVIEW_URL}.\n${stderr}`);
}

async function readProcessOutput(process: Bun.Subprocess<"ignore", "ignore", "pipe">) {
  return process.stderr ? await new Response(process.stderr).text() : "";
}
