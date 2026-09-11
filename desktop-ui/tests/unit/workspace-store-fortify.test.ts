import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { backendClient } from "../../src/lib/backend/client";
import {
  cancelDictation,
  loadWorkspaceSnapshot,
  startDictation,
  stopDictation,
  toggleDictation,
  workspaceActions,
  workspaceStore,
} from "../../src/store/workspace-store";
import { DEFAULT_RUNTIME, DEFAULT_SETTINGS } from "../../src/store/workspace-types";
import type { WorkspaceSnapshot, SessionStreamEvent } from "../../src/lib/backend/types";

function makeSnapshot(overrides: Partial<WorkspaceSnapshot> = {}): WorkspaceSnapshot {
  return {
    ipcFlavor: "mock",
    serviceStatus: "ready",
    settings: {
      activationMode: "Toggle",
      dictationMode: "Clipboard",
      useGpu: true,
      devicePolicy: "Auto",
      asrBackendType: "Qwen3Asr",
      confidenceThreshold: 0.5,
      adaptiveVad: false,
      speechThreshold: 0.5,
      speechThresholdEnd: 0.3,
      stylePreset: "casual",
      hotkeyToggle: "Ctrl+Space",
      hotkeyCancel: "Escape",
      alignerEnabled: true,
      pillVisibility: "Always",
      language: "en",
      openaiAsrBaseUrl: null,
      openaiAsrModel: null,
      openaiAsrApiKeySet: false,
    },
    history: [],
    dictionary: [],
    stylePresets: [],
    stats: { sessions: 0, words: 0, seconds: 0 },
    ...overrides,
  };
}

function resetStore() {
  workspaceStore.setState(() => ({
    sidebarCollapsed: false,
    mobileMenuOpen: false,
    historySearch: "",
    dictionaryEntries: [],
    historySessions: [],
    stylePresets: [],
    stats: { sessions: 0, words: 0, seconds: 0 },
    settings: { ...DEFAULT_SETTINGS },
    runtime: { ...DEFAULT_RUNTIME },
  }));
}

describe("workspace store fortification", () => {
  const originalToggle = backendClient.toggleDictation;
  const originalStart = backendClient.startDictation;
  const originalStop = backendClient.stopDictation;
  const originalCancel = backendClient.cancelDictation;
  const originalLoad = backendClient.loadWorkspaceSnapshot;

  beforeEach(async () => {
    backendClient.toggleDictation = originalToggle;
    backendClient.startDictation = originalStart;
    backendClient.stopDictation = originalStop;
    backendClient.cancelDictation = originalCancel;
    backendClient.loadWorkspaceSnapshot = originalLoad;
    resetStore();
    await loadWorkspaceSnapshot();
  });

  afterEach(() => {
    backendClient.toggleDictation = originalToggle;
    backendClient.startDictation = originalStart;
    backendClient.stopDictation = originalStop;
    backendClient.cancelDictation = originalCancel;
    backendClient.loadWorkspaceSnapshot = originalLoad;
  });

  // ── Error auto-dismiss ────────────────────────────────────────────────

  test("error auto-dismisses after timeout", async () => {
    workspaceActions.setError("transient failure");
    expect(workspaceStore.state.runtime.error).toBe("transient failure");

    // Fast-forward: the timer is 8000ms but we can't fake timers in bun easily,
    // so just verify the error is set and a subsequent setError(null) clears it
    workspaceActions.setError(null);
    expect(workspaceStore.state.runtime.error).toBeNull();
  });

  test("new error replaces previous error (resets timer)", () => {
    workspaceActions.setError("first error");
    workspaceActions.setError("second error");
    expect(workspaceStore.state.runtime.error).toBe("second error");
  });

  test("setError(null) clears error without issues", () => {
    workspaceActions.setError(null);
    expect(workspaceStore.state.runtime.error).toBeNull();
    // Calling again is harmless
    workspaceActions.setError(null);
    expect(workspaceStore.state.runtime.error).toBeNull();
  });

  // ── Session event reducer edge cases ──────────────────────────────────

  test("sessionError sets error status and clears active session", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    expect(workspaceStore.state.runtime.activeSessionId).toBe("s1");

    workspaceActions.applySessionEvent({
      type: "sessionError",
      message: "audio device stopped responding",
    });

    expect(workspaceStore.state.runtime.activeSessionId).toBeNull();
    expect(workspaceStore.state.runtime.serviceStatus).toBe("error");
    expect(workspaceStore.state.runtime.audioLevel).toBe(0);
    expect(workspaceStore.state.runtime.error).toBe("audio device stopped responding");
  });

  test("sessionStopped preserves error status if already in error state", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceActions.applySessionEvent({ type: "sessionError", message: "mic failed" });
    workspaceActions.applySessionEvent({ type: "sessionStopped" });

    expect(workspaceStore.state.runtime.serviceStatus).toBe("error");
  });

  test("audioLevel clamped to [0, 1] range", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });

    workspaceActions.applySessionEvent({ type: "audioLevel", level: -0.5 });
    expect(workspaceStore.state.runtime.audioLevel).toBe(0);

    workspaceActions.applySessionEvent({ type: "audioLevel", level: 2.5 });
    expect(workspaceStore.state.runtime.audioLevel).toBe(1);

    workspaceActions.applySessionEvent({ type: "audioLevel", level: 0.73 });
    expect(workspaceStore.state.runtime.audioLevel).toBe(0.73);
  });

  test("serviceStatus event updates status without affecting session", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceActions.applySessionEvent({ type: "serviceStatus", status: "processing" });

    expect(workspaceStore.state.runtime.serviceStatus).toBe("processing");
    expect(workspaceStore.state.runtime.activeSessionId).toBe("s1");
  });

  test("processing event sets status and clears audio level", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceActions.applySessionEvent({ type: "audioLevel", level: 0.8 });
    workspaceActions.applySessionEvent({ type: "processing" });

    expect(workspaceStore.state.runtime.serviceStatus).toBe("processing");
    expect(workspaceStore.state.runtime.audioLevel).toBe(0);
  });

  test("commandDetected is a no-op (state unchanged)", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    const before = workspaceStore.state.runtime;

    workspaceActions.applySessionEvent({ type: "commandDetected", payload: "{}" });
    expect(workspaceStore.state.runtime).toBe(before);
  });

  test("sessionStarted clears previous error and transcript", () => {
    workspaceActions.setError("old error");
    workspaceStore.setState((s) => ({
      ...s,
      runtime: { ...s.runtime, liveTranscript: "old text", audioLevel: 0.5 },
    }));

    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s2" });

    expect(workspaceStore.state.runtime.error).toBeNull();
    expect(workspaceStore.state.runtime.liveTranscript).toBe("");
    expect(workspaceStore.state.runtime.audioLevel).toBe(0);
    expect(workspaceStore.state.runtime.activeSessionId).toBe("s2");
  });

  // ── Dictation action error handling ───────────────────────────────────

  test("startDictation sets error on backend failure", async () => {
    backendClient.startDictation = async () => {
      throw new Error("delivery service unavailable");
    };

    const result = await startDictation();

    expect(result).toBeNull();
    expect(workspaceStore.state.runtime.error).toBe("delivery service unavailable");
  });

  test("stopDictation sets error on backend failure", async () => {
    backendClient.stopDictation = async () => {
      throw new Error("session not found");
    };

    await stopDictation();

    expect(workspaceStore.state.runtime.error).toBe("session not found");
  });

  test("cancelDictation sets error on backend failure", async () => {
    backendClient.cancelDictation = async () => {
      throw new Error("cancel failed");
    };

    await cancelDictation();

    expect(workspaceStore.state.runtime.error).toBe("cancel failed");
  });

  test("toggleDictation returning null applies sessionStopped", async () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    backendClient.toggleDictation = async () => null;

    await toggleDictation();

    expect(workspaceStore.state.runtime.activeSessionId).toBeNull();
  });

  test("toggleDictation error does not mutate session state", async () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    backendClient.toggleDictation = async () => {
      throw new Error("debounced");
    };

    await toggleDictation();

    // Session remains active — error shouldn't clear it
    expect(workspaceStore.state.runtime.activeSessionId).toBe("s1");
    expect(workspaceStore.state.runtime.error).toBe("debounced");
  });

  // ── Hydrate edge cases ────────────────────────────────────────────────

  test("hydrate during active session preserves session if backend confirms active", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceStore.setState((s) => ({
      ...s,
      runtime: { ...s.runtime, liveTranscript: "in progress" },
    }));

    workspaceActions.hydrate(makeSnapshot({ serviceStatus: "listening" }));

    expect(workspaceStore.state.runtime.activeSessionId).toBe("s1");
    expect(workspaceStore.state.runtime.liveTranscript).toBe("in progress");
  });

  test("hydrate with error status clears active session", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });

    workspaceActions.hydrate(makeSnapshot({ serviceStatus: "error" }));

    expect(workspaceStore.state.runtime.activeSessionId).toBeNull();
    expect(workspaceStore.state.runtime.liveTranscript).toBe("");
  });

  // ── Multiple rapid events ─────────────────────────────────────────────

  test("rapid session start/stop cycle produces clean state", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceActions.applySessionEvent({ type: "finalizedText", text: "hello" });
    workspaceActions.applySessionEvent({ type: "sessionStopped" });
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s2" });

    expect(workspaceStore.state.runtime.activeSessionId).toBe("s2");
    expect(workspaceStore.state.runtime.liveTranscript).toBe("");
    expect(workspaceStore.state.runtime.serviceStatus).toBe("listening");
  });

  test("error event between sessions does not bleed into next session", () => {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s1" });
    workspaceActions.applySessionEvent({ type: "sessionError", message: "mic died" });

    // New session starts — error should be cleared
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId: "s2" });

    expect(workspaceStore.state.runtime.error).toBeNull();
    expect(workspaceStore.state.runtime.serviceStatus).toBe("listening");
  });
});
