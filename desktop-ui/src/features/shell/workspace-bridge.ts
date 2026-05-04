import { attachSessionStream, listenToWorkspaceInvalidation } from '../../lib/backend/stream';
import type { SessionStreamEvent } from '../../lib/backend/types';
import { loadWorkspaceSnapshot } from '../../store/workspace-store';

const DEFAULT_RETRY_DELAY_MS = 5000;
const HEALTH_CHECK_INTERVAL_MS = 15_000;
const SESSION_STREAM_SILENCE_LIMIT_MS = 30_000;

export type WorkspaceBridgeDeps = {
  onSessionEvent: (event: SessionStreamEvent) => void;
  onSnapshotError: (error: unknown) => void;
  onConnectionError?: (scope: 'session' | 'history', error: unknown) => void;
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

  async function connectHistoryInvalidation() {
    try {
      const detach = await listenToWorkspaceInvalidationFn('lune://history-changed', () => {
        void requestSnapshotSync();
      });
      if (cancelled) {
        detach();
        return;
      }
      disposers.push(detach);
      void requestSnapshotSync();
    } catch (error) {
      onConnectionError?.('history', error);
      if (!cancelled) scheduleRetry(connectHistoryInvalidation);
    }
  }

  // Track last event from session stream to detect silence (backend crash/hang).
  let lastStreamEventAt = Date.now();
  const originalOnSessionEvent = onSessionEvent;
  const wrappedOnSessionEvent = (event: SessionStreamEvent) => {
    lastStreamEventAt = Date.now();
    originalOnSessionEvent(event);
  };

  async function connectSessionStreamWrapped() {
    try {
      const detach = await attachSessionStreamFn(wrappedOnSessionEvent);
      if (cancelled) {
        detach();
        return;
      }
      lastStreamEventAt = Date.now();
      disposers.push(detach);
    } catch (error) {
      onConnectionError?.('session', error);
      if (!cancelled) scheduleRetry(connectSessionStreamWrapped);
    }
  }

  // Periodic health check: if stream has been silent too long while we think
  // a session is active, reconcile by reloading the workspace snapshot.
  // Also detects backend crash/restart where the stream just stops.
  const healthInterval = setInterval(() => {
    if (cancelled) return;
    const silenceMs = Date.now() - lastStreamEventAt;
    if (silenceMs > SESSION_STREAM_SILENCE_LIMIT_MS) {
      console.warn(
        `[workspace-bridge] session stream silent for ${Math.round(silenceMs / 1000)}s — reconciling state`,
      );
      lastStreamEventAt = Date.now(); // prevent repeated triggers
      void requestSnapshotSync();
    }
  }, HEALTH_CHECK_INTERVAL_MS);

  void (async () => {
    await Promise.all([connectSessionStreamWrapped(), connectHistoryInvalidation()]);
  })();

  return () => {
    cancelled = true;
    clearInterval(healthInterval);
    for (const timer of timers) clearTimeoutFn(timer);
    timers.clear();
    for (const dispose of disposers) dispose();
    disposers.length = 0;
  };
}
