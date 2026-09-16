// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { backendClient } from "../../src/lib/backend/client";
import { resetMockBackend } from "../../src/lib/backend/mock";
import {
  addDictionaryEntry,
  loadWorkspaceSnapshot,
  removeDictionaryEntry,
  saveWorkspaceSetting,
  toggleDictation,
  workspaceActions,
  workspaceStore,
} from "../../src/store/workspace-store";
import { DEFAULT_RUNTIME, DEFAULT_SETTINGS } from "../../src/store/workspace-types";
import type { WorkspaceSnapshot } from "../../src/lib/backend/types";

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
    capabilities: { pillAllWorkspaces: true },
    history: [],
    dictionary: [],
    stylePresets: [],
    stats: { sessions: 0, words: 0, seconds: 0 },
    ...overrides,
  };
}

describe("workspace store", () => {
  const originalSaveWorkspaceSettings = backendClient.saveWorkspaceSettings;
  const originalLoadWorkspaceSnapshot = backendClient.loadWorkspaceSnapshot;
  const originalToggleDictation = backendClient.toggleDictation;

  beforeEach(async () => {
    resetMockBackend();
    backendClient.saveWorkspaceSettings = originalSaveWorkspaceSettings;
    backendClient.loadWorkspaceSnapshot = originalLoadWorkspaceSnapshot;
    backendClient.toggleDictation = originalToggleDictation;
    workspaceStore.setState(() => ({
      sidebarCollapsed: false,
      mobileMenuOpen: false,
      historySearch: "",
      dictionaryEntries: [],
      historySessions: [],
      settings: { ...DEFAULT_SETTINGS },
      runtime: { ...DEFAULT_RUNTIME },
    }));
    await loadWorkspaceSnapshot();
  });

  afterEach(() => {
    backendClient.saveWorkspaceSettings = originalSaveWorkspaceSettings;
    backendClient.loadWorkspaceSnapshot = originalLoadWorkspaceSnapshot;
    backendClient.toggleDictation = originalToggleDictation;
  });

  test("keeps dictionary entry ids unique for duplicate phrases", async () => {
    const addedFirst = await addDictionaryEntry("github", "GitHub");
    const addedSecond = await addDictionaryEntry("github", "GitHub Enterprise");

    expect(addedFirst).toBe(true);
    expect(addedSecond).toBe(true);

    const githubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");

    expect(githubEntries).toHaveLength(3);
    expect(new Set(githubEntries.map((entry) => entry.id)).size).toBe(3);

    await removeDictionaryEntry(githubEntries[1]!.id);

    const remainingGithubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");
    expect(remainingGithubEntries).toHaveLength(2);
  });

  test("rejects incomplete dictionary entries without mutating state", async () => {
    const before = workspaceStore.state.dictionaryEntries.length;

    expect(await addDictionaryEntry("github", "")).toBe(false);
    expect(await addDictionaryEntry("", "GitHub")).toBe(false);
    expect(workspaceStore.state.dictionaryEntries).toHaveLength(before);
  });

  test("hydrate clears stale session state when backend is idle", () => {
    workspaceStore.setState((s) => ({
      ...s,
      runtime: {
        ...s.runtime,
        activeSessionId: "session-123",
        liveTranscript: "stale transcript",
        audioLevel: 0.7,
        serviceStatus: "processing",
      },
    }));

    workspaceActions.hydrate(makeSnapshot({ serviceStatus: "ready" }));

    expect(workspaceStore.state.runtime.activeSessionId).toBeNull();
    expect(workspaceStore.state.runtime.liveTranscript).toBe("");
    expect(workspaceStore.state.runtime.audioLevel).toBe(0);
  });

  // Windows' window backend cannot show a window on every virtual desktop, so
  // the snapshot carries the capability and the UI drops the control rather
  // than offering a toggle that does nothing.
  test("hydrate carries the platform capability flags", () => {
    workspaceActions.hydrate(makeSnapshot({ capabilities: { pillAllWorkspaces: false } }));
    expect(workspaceStore.state.runtime.capabilities.pillAllWorkspaces).toBe(false);

    workspaceActions.hydrate(makeSnapshot({ capabilities: { pillAllWorkspaces: true } }));
    expect(workspaceStore.state.runtime.capabilities.pillAllWorkspaces).toBe(true);
  });

  test("saveWorkspaceSetting resyncs from backend on failed optimistic save", async () => {
    backendClient.saveWorkspaceSettings = async () => {
      throw new Error("backend rejected change");
    };
    backendClient.loadWorkspaceSnapshot = async () =>
      makeSnapshot({
        settings: {
          ...makeSnapshot().settings,
          dictationMode: "Clipboard",
        },
      });

    await saveWorkspaceSetting("dictationMode", "Both");

    expect(workspaceStore.state.settings.dictationMode).toBe("Clipboard");
    expect(workspaceStore.state.runtime.error).toContain("backend rejected change");
  });

  test("hydrate keeps OpenAI api keys out of store state", () => {
    const snapshotSettings = makeSnapshot().settings;
    workspaceActions.hydrate(
      makeSnapshot({
        settings: {
          ...snapshotSettings,
          openaiAsrApiKeySet: true,
        },
      }),
    );

    expect(workspaceStore.state.settings.openaiAsrApiKey).toBe("••••••••");
  });

  test("finalized text appends to live transcript", () => {
    workspaceActions.applySessionEvent({
      type: "sessionStarted",
      sessionId: "session-123",
    });
    workspaceActions.applySessionEvent({
      type: "finalizedText",
      text: "hello",
    });
    workspaceActions.applySessionEvent({
      type: "finalizedText",
      text: "world",
    });

    expect(workspaceStore.state.runtime.liveTranscript).toBe("hello world");
  });

  test("audio level events are clamped and reset when session stops", () => {
    workspaceActions.applySessionEvent({
      type: "sessionStarted",
      sessionId: "session-123",
    });
    workspaceActions.applySessionEvent({
      type: "audioLevel",
      level: 1.8,
    });

    expect(workspaceStore.state.runtime.audioLevel).toBe(1);

    workspaceActions.applySessionEvent({ type: "sessionStopped" });

    expect(workspaceStore.state.runtime.audioLevel).toBe(0);
  });

  test("toggleDictation mirrors backend command result into runtime state", async () => {
    backendClient.toggleDictation = async () => "session-from-main";

    const sessionId = await toggleDictation();

    expect(sessionId).toBe("session-from-main");
    expect(workspaceStore.state.runtime.activeSessionId).toBe("session-from-main");
    expect(workspaceStore.state.runtime.serviceStatus).toBe("listening");
  });
});
