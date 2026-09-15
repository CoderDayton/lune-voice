// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { startWorkspaceBridge } from "../../src/features/shell/workspace-bridge";
import type { SessionStreamEvent } from "../../src/lib/backend/types";

function createFlushMicrotasks() {
  return async function flushMicrotasks() {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  };
}

describe("workspace bridge fortification", () => {
  test("wraps session events to track last event timestamp", async () => {
    const flush = createFlushMicrotasks();
    const events: SessionStreamEvent[] = [];
    let emitEvent: ((event: SessionStreamEvent) => void) | null = null;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: (event) => {
        events.push(event);
      },
      onSnapshotError: () => {},
      attachSessionStream: async (cb) => {
        emitEvent = cb;
        return () => {};
      },
      listenToWorkspaceInvalidation: async () => () => {},
      loadWorkspaceSnapshot: async () => {},
    });

    await flush();

    expect(emitEvent).not.toBeNull();
    emitEvent!({ type: "serviceStatus", status: "ready" });
    emitEvent!({ type: "sessionStarted", sessionId: "s1" });

    expect(events).toHaveLength(2);
    expect(events[0]).toEqual({ type: "serviceStatus", status: "ready" });
    expect(events[1]).toEqual({ type: "sessionStarted", sessionId: "s1" });

    cleanup();
  });

  test("does not call connectSessionStream after cleanup even if retry was scheduled", async () => {
    const flush = createFlushMicrotasks();
    const scheduled: Array<() => void> = [];
    let sessionAttempts = 0;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: () => {},
      onSnapshotError: () => {},
      attachSessionStream: async () => {
        sessionAttempts += 1;
        if (sessionAttempts === 1) throw new Error("fail");
        return () => {};
      },
      listenToWorkspaceInvalidation: async () => () => {},
      loadWorkspaceSnapshot: async () => {},
      setTimeoutFn: ((task) => {
        scheduled.push(task);
        return task as unknown as ReturnType<typeof setTimeout>;
      }) as typeof setTimeout,
      clearTimeoutFn: ((timer) => {
        const index = scheduled.indexOf(timer as unknown as () => void);
        if (index >= 0) scheduled.splice(index, 1);
      }) as typeof clearTimeout,
    });

    await flush();
    expect(sessionAttempts).toBe(1);
    expect(scheduled).toHaveLength(1);

    // Cleanup before retry fires
    cleanup();
    expect(scheduled).toHaveLength(0);
  });

  test("detaches session stream if cancelled during connection", async () => {
    const flush = createFlushMicrotasks();
    let detached = false;
    let resolveAttach: (() => void) | null = null;

    const attachPromise = new Promise<() => void>((resolve) => {
      resolveAttach = () =>
        resolve(() => {
          detached = true;
        });
    });

    const cleanup = startWorkspaceBridge({
      onSessionEvent: () => {},
      onSnapshotError: () => {},
      attachSessionStream: async () => attachPromise,
      listenToWorkspaceInvalidation: async () => () => {},
      loadWorkspaceSnapshot: async () => {},
    });

    await flush();

    // Cancel before attach resolves
    cleanup();
    resolveAttach!();
    await flush();

    expect(detached).toBe(true);
  });

  test("multiple events delivered through wrapped handler in order", async () => {
    const flush = createFlushMicrotasks();
    let emitEvent: ((event: SessionStreamEvent) => void) | null = null;
    const events: string[] = [];

    const cleanup = startWorkspaceBridge({
      onSessionEvent: (event) => {
        events.push(event.type);
      },
      onSnapshotError: () => {},
      attachSessionStream: async (cb) => {
        emitEvent = cb;
        return () => {};
      },
      listenToWorkspaceInvalidation: async () => () => {},
      loadWorkspaceSnapshot: async () => {},
    });

    await flush();

    emitEvent!({ type: "serviceStatus", status: "ready" });
    emitEvent!({ type: "sessionStarted", sessionId: "s1" });
    emitEvent!({ type: "audioLevel", level: 0.5 });
    emitEvent!({ type: "processing" });
    emitEvent!({ type: "finalizedText", text: "hello" });
    emitEvent!({ type: "sessionStopped" });

    expect(events).toEqual([
      "serviceStatus",
      "sessionStarted",
      "audioLevel",
      "processing",
      "finalizedText",
      "sessionStopped",
    ]);
    cleanup();
  });

  test("snapshot sync deduplicates concurrent requests", async () => {
    const flush = createFlushMicrotasks();
    let loadCount = 0;
    let resolveLoad: (() => void) | null = null;
    let historyCallback: (() => void) | null = null;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: () => {},
      onSnapshotError: () => {},
      attachSessionStream: async () => () => {},
      listenToWorkspaceInvalidation: async (_event, cb) => {
        historyCallback = cb;
        return () => {};
      },
      loadWorkspaceSnapshot: async () => {
        loadCount++;
        await new Promise<void>((r) => {
          resolveLoad = r;
        });
      },
    });

    await flush();

    // First load from connectHistoryInvalidation
    expect(loadCount).toBe(1);

    // Trigger another invalidation while first is in-flight
    historyCallback!();
    await flush();

    // Should not have started a second load
    expect(loadCount).toBe(1);

    // Resolve first load
    resolveLoad!();
    await flush();

    // Now trigger again — should start a new load
    historyCallback!();
    await flush();
    expect(loadCount).toBe(2);

    resolveLoad!();
    await flush();
    cleanup();
  });

  test("cleanup clears health interval (no leaks)", async () => {
    const flush = createFlushMicrotasks();
    const clearedIntervals: number[] = [];
    const originalClearInterval = globalThis.clearInterval;
    globalThis.clearInterval = ((id: number) => {
      clearedIntervals.push(id);
      originalClearInterval(id);
    }) as typeof clearInterval;

    const cleanup = startWorkspaceBridge({
      onSessionEvent: () => {},
      onSnapshotError: () => {},
      attachSessionStream: async () => () => {},
      listenToWorkspaceInvalidation: async () => () => {},
      loadWorkspaceSnapshot: async () => {},
    });

    await flush();
    cleanup();

    expect(clearedIntervals.length).toBeGreaterThan(0);
    globalThis.clearInterval = originalClearInterval;
  });
});
