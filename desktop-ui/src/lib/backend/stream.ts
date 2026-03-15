import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { isTauriRuntime } from "./runtime";
import type { SessionStreamEvent } from "./types";

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

  await invoke("subscribe_session_stream", { channel });

  return () => {
    invoke("detach_session_stream").catch((e) => {
      console.warn("[stream] detach_session_stream failed", e);
    });
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
