import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { backendClient } from "../../src/lib/backend/client";
import {
  addDictionaryEntry,
  loadWorkspaceSnapshot,
  removeDictionaryEntry,
  saveWorkspaceSetting,
  workspaceActions,
  workspaceStore,
} from "../../src/store/workspace-store";
import { DEFAULT_RUNTIME, DEFAULT_SETTINGS } from "../../src/store/workspace-types";
import type { BackendWorkspaceSnapshot } from "../../src/lib/backend/types";

function makeSnapshot(overrides: Partial<BackendWorkspaceSnapshot> = {}): BackendWorkspaceSnapshot {
  return {
    ipcFlavor: "mock",
    serviceStatus: "ready",
    settings: {
      activationMode: "Toggle",
      dictationMode: "Clipboard",
      useGpu: true,
      devicePolicy: "Auto",
      asrBackendType: "OrtQwenAsr",
      llmBackendType: "LlamaCppLocal",
      confidenceThreshold: 0.5,
      adaptiveVad: false,
      speechThreshold: 0.5,
      stylePreset: "casual",
      hotkeyToggle: "Ctrl+Space",
      hotkeyCancel: "Escape",
      alignerEnabled: true,
      pillVisibility: "Always",
      language: "en",
      openaiAsrBaseUrl: null,
      openaiAsrModel: null,
      openaiLlmBaseUrl: null,
      openaiLlmModel: null,
    },
    history: [],
    dictionary: [],
    ...overrides,
  };
}

describe("workspace store", () => {
  const originalSaveWorkspaceSettings = backendClient.saveWorkspaceSettings;
  const originalLoadWorkspaceSnapshot = backendClient.loadWorkspaceSnapshot;

  beforeEach(async () => {
    backendClient.saveWorkspaceSettings = originalSaveWorkspaceSettings;
    backendClient.loadWorkspaceSnapshot = originalLoadWorkspaceSnapshot;
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
        serviceStatus: "processing",
      },
    }));

    workspaceActions.hydrate(makeSnapshot({ serviceStatus: "ready" }));

    expect(workspaceStore.state.runtime.activeSessionId).toBeNull();
    expect(workspaceStore.state.runtime.liveTranscript).toBe("");
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
          openaiAsrApiKey: "secret-asr",
        },
      }),
    );

    expect(workspaceStore.state.settings.openaiAsrApiKey).toBe("");
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
});
