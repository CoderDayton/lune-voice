import type { SessionStreamEvent } from "../../lib/backend/types";
import { attachSessionStream, listenToWorkspaceInvalidation } from "../../lib/backend/stream";
import { loadWorkspaceSnapshot } from "../../store/workspace-store";

const DEFAULT_RETRY_DELAY_MS = 5000;

export type WorkspaceBridgeDeps = {
  onSessionEvent: (event: SessionStreamEvent) => void;
  onSnapshotError: (error: unknown) => void;
  onConnectionError?: (scope: "session" | "history", error: unknown) => void;
  attachSessionStream?: typeof attachSessionStream;
  listenToWorkspaceInvalidation?: typeof listenToWorkspaceInvalidation;
  loadWorkspaceSnapshot?: typeof loadWorkspaceSnapshot;
  retryDelayMs?: number;
  setTimeoutFn?: typeof setTimeout;
  clearTimeoutFn?: typeof clearTimeout;
};

export function startWorkspaceBridge(deps: WorkspaceBridgeDeps) {
  const {
    onSessionEvent,
    onSnapshotError,
    onConnectionError,
    attachSessionStream: attachSessionStreamFn = attachSessionStream,
    listenToWorkspaceInvalidation: listenToWorkspaceInvalidationFn = listenToWorkspaceInvalidation,
    loadWorkspaceSnapshot: loadWorkspaceSnapshotFn = loadWorkspaceSnapshot,
    retryDelayMs = DEFAULT_RETRY_DELAY_MS,
    setTimeoutFn = setTimeout,
    clearTimeoutFn = clearTimeout,
  } = deps;

  const disposers: Array<() => void> = [];
  const timers = new Set<ReturnType<typeof setTimeoutFn>>();
  let cancelled = false;
  let snapshotSyncInFlight: Promise<void> | null = null;

  function scheduleRetry(task: () => Promise<void>) {
    const timer = setTimeoutFn(() => {
      timers.delete(timer);
      void task();
    }, retryDelayMs);
    timers.add(timer);
  }

  function requestSnapshotSync() {
    if (snapshotSyncInFlight) return snapshotSyncInFlight;

    snapshotSyncInFlight = (async () => {
      try {
        await loadWorkspaceSnapshotFn();
      } catch (error) {
        onSnapshotError(error);
      } finally {
        snapshotSyncInFlight = null;
      }
    })();

    return snapshotSyncInFlight;
  }

  async function connectSessionStream() {
    try {
      const detach = await attachSessionStreamFn(onSessionEvent);
      if (cancelled) {
        detach();
        return;
      }
      disposers.push(detach);
    } catch (error) {
      onConnectionError?.("session", error);
      if (!cancelled) scheduleRetry(connectSessionStream);
    }
  }

  async function connectHistoryInvalidation() {
    try {
      const detach = await listenToWorkspaceInvalidationFn("lune://history-changed", () => {
        void requestSnapshotSync();
      });
      if (cancelled) {
        detach();
        return;
      }
      disposers.push(detach);
      void requestSnapshotSync();
    } catch (error) {
      onConnectionError?.("history", error);
      if (!cancelled) scheduleRetry(connectHistoryInvalidation);
    }
  }

  void (async () => {
    await Promise.all([connectSessionStream(), connectHistoryInvalidation()]);
  })();

  return () => {
    cancelled = true;
    for (const timer of timers) clearTimeoutFn(timer);
    timers.clear();
    for (const dispose of disposers) dispose();
    disposers.length = 0;
  };
}
