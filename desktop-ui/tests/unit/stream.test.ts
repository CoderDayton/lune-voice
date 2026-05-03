import { describe, expect, test } from "bun:test";

// We can't directly import the private functions from stream.ts,
// but we can test the public attachSessionStream behavior in non-Tauri env.
import { attachSessionStream, listenToWorkspaceInvalidation } from "../../src/lib/backend/stream";
import { isTauriRuntime } from "../../src/lib/backend/runtime";

describe("stream module (non-Tauri environment)", () => {
  test("attachSessionStream returns no-op detach when not in Tauri", async () => {
    expect(isTauriRuntime()).toBe(false);

    const events: unknown[] = [];
    const detach = await attachSessionStream((event) => {
      events.push(event);
    });

    expect(typeof detach).toBe("function");
    // Calling detach is safe
    detach();
    // No events emitted in non-Tauri mode
    expect(events).toHaveLength(0);
  });

  test("listenToWorkspaceInvalidation returns no-op when not in Tauri", async () => {
    expect(isTauriRuntime()).toBe(false);

    let called = false;
    const detach = await listenToWorkspaceInvalidation("lune://history-changed", () => {
      called = true;
    });

    expect(typeof detach).toBe("function");
    detach();
    expect(called).toBe(false);
  });

  test("attachSessionStream detach can be called multiple times safely", async () => {
    const detach = await attachSessionStream(() => {});
    detach();
    detach();
    detach();
    // Should not throw
  });
});

// Test the retry/timeout logic patterns used in the stream module
describe("retry and timeout patterns", () => {
  test("exponential backoff pattern respects max attempts", async () => {
    let attempts = 0;
    const maxAttempts = 3;

    async function retryInvoke<T>(fn: () => Promise<T>): Promise<T> {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await fn();
        } catch (e) {
          if (attempt === maxAttempts) throw e;
        }
      }
      throw new Error("unreachable");
    }

    await expect(
      retryInvoke(async () => {
        attempts++;
        throw new Error("always fails");
      }),
    ).rejects.toThrow("always fails");

    expect(attempts).toBe(3);
  });

  test("retryInvoke succeeds on second attempt", async () => {
    let attempts = 0;

    async function retryInvoke<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await fn();
        } catch (e) {
          if (attempt === maxAttempts) throw e;
        }
      }
      throw new Error("unreachable");
    }

    const result = await retryInvoke(async () => {
      attempts++;
      if (attempts < 2) throw new Error("transient");
      return "success";
    });

    expect(result).toBe("success");
    expect(attempts).toBe(2);
  });

  test("timeout rejects when promise is too slow", async () => {
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

    const slow = new Promise<string>((resolve) => setTimeout(() => resolve("late"), 5000));

    await expect(withTimeout(slow, 10, "test_op")).rejects.toThrow(
      "test_op timed out after 10ms",
    );
  });

  test("timeout resolves when promise is fast enough", async () => {
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

    const fast = Promise.resolve("quick");
    const result = await withTimeout(fast, 5000, "test_op");
    expect(result).toBe("quick");
  });
});
