import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { isTauriRuntime } from "./runtime";
import type { SessionStreamEvent } from "./types";

async function retryInvoke<T>(fn: () => Promise<T>, maxAttempts = 3, delayMs = 1000): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (attempt === maxAttempts) throw e;
      await new Promise((r) => setTimeout(r, delayMs * attempt));
    }
  }
  throw new Error("unreachable");
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timeoutId) clearTimeout(timeoutId);
  });
}

/**
 * Clean up a Tauri Channel's JS-side callback from the window object.
 * Tauri's Channel registers a callback via `transformCallback` on
 * `window.__TAURI_INTERNALS__` but never removes it (known issue #13133).
 * On HMR reload this causes "Couldn't find callback id" warnings.
 */
// biome-ignore lint/suspicious/noExplicitAny: accessing Tauri internals
function cleanupChannel(channel: Channel<any>) {
  try {
    const id = (channel as { id: number }).id;
    // biome-ignore lint/suspicious/noExplicitAny: Tauri internal window property
    const internals = (window as any).__TAURI_INTERNALS__ as
      | Record<string, unknown>
      | undefined;
    if (internals && typeof id === "number") {
      delete internals[`_${id}`];
    }
  } catch {
    // Best-effort cleanup
  }
}

export async function attachSessionStream(
  onEvent: (event: SessionStreamEvent) => void,
): Promise<() => void> {
  if (!isTauriRuntime()) {
    return () => {};
  }

  const channel = new Channel<SessionStreamEvent>((event) => {
    try {
      onEvent(event);
    } catch (e) {
      console.error("[stream] session event handler threw", e);
    }
  });

  let subscriptionId = 0;
  try {
    subscriptionId = await retryInvoke(() =>
      withTimeout(
        invoke<number>("subscribe_session_stream", { channel }),
        10_000,
        "subscribe_session_stream",
      ),
    );
  } catch (error) {
    cleanupChannel(channel);
    throw error;
  }

  return () => {
    invoke("detach_session_stream", { subscriptionId }).catch((e) => {
      console.warn("[stream] detach_session_stream failed", e);
    });
    cleanupChannel(channel);
  };
}

export async function listenToWorkspaceInvalidation(
  eventName: "lune://history-changed",
  onInvalidate: () => void,
): Promise<() => void> {
  if (!isTauriRuntime()) {
    return () => {};
  }

  return listen(eventName, () => {
    onInvalidate();
  });
}
