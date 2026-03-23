import { describe, expect, test } from "bun:test";
import { startWorkspaceBridge } from "../../src/features/shell/workspace-bridge";
import type { SessionStreamEvent } from "../../src/lib/backend/types";

function createFlushMicrotasks() {
  return async function flushMicrotasks() {
    await Promise.resolve();
    await Promise.resolve();
  };
}

describe("workspace bridge", () => {
  test("loads the snapshot after initial listener setup even if the session stream retries", async () => {
    const flush = createFlushMicrotasks();
    const events: SessionStreamEvent[] = [];
    const scheduled: Array<() => void> = [];
    const snapshots: string[] = [];
    let sessionAttempts = 0;
    let historyAttempts = 0;
    let detachCount = 0;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: (event) => {
        events.push(event);
      },
      onSnapshotError: (error) => {
        throw error;
      },
      onConnectionError: () => {},
      attachSessionStream: async () => {
        sessionAttempts += 1;
        if (sessionAttempts === 1) {
          throw new Error("session stream unavailable");
        }
        return () => {
          detachCount += 1;
        };
      },
      listenToWorkspaceInvalidation: async () => {
        historyAttempts += 1;
        return () => {
          detachCount += 1;
        };
      },
      loadWorkspaceSnapshot: async () => {
        snapshots.push("sync");
      },
      setTimeoutFn: ((task) => {
        scheduled.push(task);
        return task as unknown as ReturnType<typeof setTimeout>;
      }) as typeof setTimeout,
      clearTimeoutFn: ((timer) => {
        const index = scheduled.indexOf(timer as unknown as () => void);
        if (index >= 0) {
          scheduled.splice(index, 1);
        }
      }) as typeof clearTimeout,
    });

    await flush();

    expect(sessionAttempts).toBe(1);
    expect(historyAttempts).toBe(1);
    expect(snapshots).toEqual(["sync"]);
    expect(scheduled).toHaveLength(1);

    scheduled[0]?.();
    await flush();

    expect(sessionAttempts).toBe(2);
    expect(snapshots).toEqual(["sync"]);

    cleanup();
    expect(detachCount).toBe(2);
    expect(events).toEqual([]);
  });

  test("resyncs when the history listener only becomes available on retry", async () => {
    const flush = createFlushMicrotasks();
    const scheduled: Array<() => void> = [];
    const snapshots: string[] = [];
    let sessionAttempts = 0;
    let historyAttempts = 0;
    let detachCount = 0;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: () => {},
      onSnapshotError: (error) => {
        throw error;
      },
      onConnectionError: () => {},
      attachSessionStream: async () => {
        sessionAttempts += 1;
        return () => {
          detachCount += 1;
        };
      },
      listenToWorkspaceInvalidation: async () => {
        historyAttempts += 1;
        if (historyAttempts === 1) {
          throw new Error("listener unavailable");
        }
        return () => {
          detachCount += 1;
        };
      },
      loadWorkspaceSnapshot: async () => {
        snapshots.push("sync");
      },
      setTimeoutFn: ((task) => {
        scheduled.push(task);
        return task as unknown as ReturnType<typeof setTimeout>;
      }) as typeof setTimeout,
      clearTimeoutFn: ((timer) => {
        const index = scheduled.indexOf(timer as unknown as () => void);
        if (index >= 0) {
          scheduled.splice(index, 1);
        }
      }) as typeof clearTimeout,
    });

    await flush();

    expect(sessionAttempts).toBe(1);
    expect(historyAttempts).toBe(1);
    expect(snapshots).toEqual([]);
    expect(scheduled).toHaveLength(1);

    scheduled[0]?.();
    await flush();

    expect(historyAttempts).toBe(2);
    expect(snapshots).toEqual(["sync"]);

    cleanup();
    expect(detachCount).toBe(2);
  });
});
