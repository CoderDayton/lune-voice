import { describe, expect, test } from "bun:test";
import { mockBackendClient } from "../../src/lib/backend/mock";
import { isTauriRuntime } from "../../src/lib/backend/runtime";

describe("isTauriRuntime", () => {
  test("returns false in test environment (no __TAURI_INTERNALS__)", () => {
    expect(isTauriRuntime()).toBe(false);
  });
});

describe("mock backend client", () => {
  test("loadWorkspaceSnapshot returns a valid snapshot", async () => {
    const snapshot = await mockBackendClient.loadWorkspaceSnapshot();

    expect(snapshot.ipcFlavor).toBe("mock");
    expect(snapshot.serviceStatus).toBe("ready");
    expect(snapshot.settings).toBeDefined();
    expect(snapshot.settings.activationMode).toBe("Toggle");
    expect(snapshot.history).toBeInstanceOf(Array);
    expect(snapshot.dictionary).toBeInstanceOf(Array);
    expect(snapshot.stylePresets).toBeInstanceOf(Array);
  });

  test("loadPillSnapshot returns pill-relevant fields only", async () => {
    const snapshot = await mockBackendClient.loadPillSnapshot();

    expect(snapshot.serviceStatus).toBe("ready");
    expect(snapshot.activationMode).toBe("Toggle");
    expect(snapshot.pillVisibility).toBe("Always");
  });

  test("loadWorkspaceSnapshot returns deep clones (no shared references)", async () => {
    const snap1 = await mockBackendClient.loadWorkspaceSnapshot();
    const snap2 = await mockBackendClient.loadWorkspaceSnapshot();

    expect(snap1).toEqual(snap2);
    expect(snap1).not.toBe(snap2);
    expect(snap1.settings).not.toBe(snap2.settings);
    expect(snap1.history).not.toBe(snap2.history);
  });

  test("createDictionaryEntry adds entry and returns updated snapshot", async () => {
    const before = await mockBackendClient.loadWorkspaceSnapshot();
    const beforeCount = before.dictionary.length;

    const after = await mockBackendClient.createDictionaryEntry("vscode", "VS Code", 5);

    expect(after.dictionary.length).toBe(beforeCount + 1);
    const added = after.dictionary[after.dictionary.length - 1]!;
    expect(added.phrase).toBe("vscode");
    expect(added.replacement).toBe("VS Code");
    expect(added.priority).toBe(5);
  });

  test("deleteDictionaryEntry removes by id", async () => {
    const snap = await mockBackendClient.loadWorkspaceSnapshot();
    const firstId = snap.dictionary[0]!.id;
    const before = snap.dictionary.length;

    const after = await mockBackendClient.deleteDictionaryEntry(firstId);

    expect(after.dictionary.length).toBe(before - 1);
    expect(after.dictionary.find((d) => d.id === firstId)).toBeUndefined();
  });

  test("deleteDictionaryEntry with non-existent id is safe", async () => {
    const before = await mockBackendClient.loadWorkspaceSnapshot();
    const after = await mockBackendClient.deleteDictionaryEntry(999999);
    expect(after.dictionary.length).toBe(before.dictionary.length);
  });

  test("saveWorkspaceSettings updates settings and returns snapshot", async () => {
    const before = await mockBackendClient.loadWorkspaceSnapshot();
    const input = {
      ...before.settings,
      asrBackendType: "OpenAiCompatible" as const,
      activationMode: "Hold" as const,
      dictationMode: "Clipboard" as const,
      pillVisibility: "Never" as const,
      devicePolicy: "Cuda" as const,
    };

    const after = await mockBackendClient.saveWorkspaceSettings(input);

    expect(after.settings.activationMode).toBe("Hold");
    expect(after.settings.asrBackendType).toBe("OpenAiCompatible");
    expect(after.settings.pillVisibility).toBe("Never");
  });

  test("saveWorkspaceSettings forces CPU when GPU disabled", async () => {
    const before = await mockBackendClient.loadWorkspaceSnapshot();
    const input = {
      ...before.settings,
      useGpu: false,
      devicePolicy: "Cuda" as const,
      activationMode: "Toggle" as const,
      dictationMode: "Clipboard" as const,
      pillVisibility: "Always" as const,
    };

    const after = await mockBackendClient.saveWorkspaceSettings(input);

    expect(after.settings.useGpu).toBe(false);
    expect(after.settings.devicePolicy).toBe("Cpu");
  });

  test("startDictation returns a session id", async () => {
    const id = await mockBackendClient.startDictation();
    expect(id).toBe("mock-session");
  });

  test("stopDictation resolves without error", async () => {
    await expect(mockBackendClient.stopDictation()).resolves.toBeUndefined();
  });

  test("cancelDictation resolves without error", async () => {
    await expect(mockBackendClient.cancelDictation()).resolves.toBeUndefined();
  });

  test("toggleDictation returns a session id", async () => {
    const id = await mockBackendClient.toggleDictation();
    expect(id).toBe("mock-session");
  });

  test("gpuBackendStatus reports no native backend outside a real build", async () => {
    const status = await mockBackendClient.gpuBackendStatus();
    expect(status.state).toBe("unsupported");
    expect(status.devices).toEqual([]);
  });

  test("history entries have valid ISO timestamps and transcripts", async () => {
    const snapshot = await mockBackendClient.loadWorkspaceSnapshot();

    for (const entry of snapshot.history) {
      expect(entry.startedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      if (entry.endedAt) {
        expect(entry.endedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
        expect(new Date(entry.endedAt).getTime()).toBeGreaterThan(
          new Date(entry.startedAt).getTime(),
        );
      }
      expect(entry.transcript).not.toBeNull();
      expect(entry.transcript!.length).toBeGreaterThan(0);
    }
  });
});
