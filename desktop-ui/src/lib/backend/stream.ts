import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { SessionStreamEvent } from "./types";

function isTauriRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function attachSessionStream(
  onEvent: (event: SessionStreamEvent) => void,
) {
  if (!isTauriRuntime()) {
    return () => {};
  }

  const channel = new Channel<SessionStreamEvent>((event) => {
    onEvent(event);
  });

  await invoke("subscribe_session_stream", { channel });

  return () => {
    void invoke("detach_session_stream");
  };
}

export async function listenToWorkspaceInvalidation(
  eventName: "lune://history-changed",
  onInvalidate: () => void,
) {
  if (!isTauriRuntime()) {
    return () => {};
  }

  return listen(eventName, () => {
    onInvalidate();
  });
}
