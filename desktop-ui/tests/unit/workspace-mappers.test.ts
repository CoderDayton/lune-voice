import { describe, expect, test } from "bun:test";
import {
  mapDictionaryEntry,
  mapHistoryEntry,
  mapWorkspaceSettings,
  settingsToSaveInput,
  summarizeAsrBackend,
  summarizeDictationMode,
} from "../../src/store/workspace-mappers";
import { DEFAULT_SETTINGS } from "../../src/store/workspace-types";
import type { HistoryItemDto, WorkspaceSnapshot } from "../../src/lib/backend/types";

function makeBackendSettings() {
  return {
    activationMode: "Toggle" as const,
    dictationMode: "Clipboard" as const,
    useGpu: true,
    devicePolicy: "Auto" as const,
    asrBackendType: "Qwen3Asr" as const,
    confidenceThreshold: 0.75,
    llmSkipThreshold: 0.8,
    adaptiveVad: false,
    speechThreshold: 0.6,
    speechThresholdEnd: 0.3,
    stylePreset: "professional",
    hotkeyToggle: "Ctrl+Space",
    hotkeyCancel: "Escape",
    pillVisibility: "WhenDictating" as const,
    language: "en",
    openaiAsrBaseUrl: null,
    openaiAsrModel: null,
    openaiAsrApiKeySet: false,
  };
}

describe("workspace mappers", () => {
  // ── mapHistoryEntry ─────────────────────────────────────────────────

  test("maps a valid history entry with all fields", () => {
    const entry: HistoryItemDto = {
      id: "s1",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: "2026-01-15T10:00:22.000Z",
      transcript: "Hello world from the test",
      backend: "Qwen3Asr",
      style: "casual",
      status: "completed",
    };

    const result = mapHistoryEntry(entry);

    expect(result).not.toBeNull();
    expect(result!.id).toBe("s1");
    expect(result!.words).toBe(5);
    expect(result!.seconds).toBe(22);
    expect(result!.transcript).toBe("Hello world from the test");
  });

  test("returns null for entries without transcript", () => {
    const entry: HistoryItemDto = {
      id: "s2",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: "2026-01-15T10:00:05.000Z",
      transcript: null,
      backend: "Qwen3Asr",
      style: "casual",
      status: "cancelled",
    };

    expect(mapHistoryEntry(entry)).toBeNull();
  });

  test("returns null for empty transcript", () => {
    const entry: HistoryItemDto = {
      id: "s3",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: "2026-01-15T10:00:05.000Z",
      transcript: "",
      backend: "Qwen3Asr",
      style: "casual",
      status: "completed",
    };

    // empty string is falsy
    expect(mapHistoryEntry(entry)).toBeNull();
  });

  test("seconds defaults to 0 when endedAt is null", () => {
    const entry: HistoryItemDto = {
      id: "s4",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: null,
      transcript: "Still running",
      backend: "Qwen3Asr",
      style: "casual",
      status: "active",
    };

    const result = mapHistoryEntry(entry);
    expect(result!.seconds).toBe(0);
  });

  test("seconds is at least 1 for very short sessions", () => {
    const entry: HistoryItemDto = {
      id: "s5",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: "2026-01-15T10:00:00.100Z",
      transcript: "Quick",
      backend: "Qwen3Asr",
      style: "casual",
      status: "completed",
    };

    const result = mapHistoryEntry(entry);
    expect(result!.seconds).toBe(1);
  });

  test("word count handles multiple spaces and tabs", () => {
    const entry: HistoryItemDto = {
      id: "s6",
      startedAt: "2026-01-15T10:00:00.000Z",
      endedAt: "2026-01-15T10:00:05.000Z",
      transcript: "  hello   world\t  foo  ",
      backend: "Qwen3Asr",
      style: "casual",
      status: "completed",
    };

    const result = mapHistoryEntry(entry);
    expect(result!.words).toBe(3);
  });

  // ── mapDictionaryEntry ──────────────────────────────────────────────

  test("passes through dictionary entry unchanged", () => {
    const entry = { id: 42, phrase: "lol", replacement: "LOL", priority: 5 };
    expect(mapDictionaryEntry(entry)).toEqual(entry);
  });

  // ── mapWorkspaceSettings ────────────────────────────────────────────

  test("maps backend settings to workspace settings", () => {
    const snapshot = {
      settings: makeBackendSettings(),
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);

    expect(result.activationMode).toBe("Toggle");
    expect(result.asrBackend).toBe("Qwen3Asr");
    expect(result.confidenceThreshold).toBe("0.75");
    expect(result.llmSkipThreshold).toBe("0.80");
    expect(result.speechThreshold).toBe("0.60");
    expect(result.pillVisibility).toBe("WhenDictating");
  });

  test("masks API key when set", () => {
    const snapshot = {
      settings: { ...makeBackendSettings(), openaiAsrApiKeySet: true },
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);
    expect(result.openaiAsrApiKey).toBe("••••••••");
  });

  test("uses empty string for API key when not set", () => {
    const snapshot = {
      settings: { ...makeBackendSettings(), openaiAsrApiKeySet: false },
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);
    expect(result.openaiAsrApiKey).toBe("");
  });

  test("uses defaults for null optional fields", () => {
    const snapshot = {
      settings: {
        ...makeBackendSettings(),
        openaiAsrBaseUrl: null,
        openaiAsrModel: null,
        pillVisibility: null as any,
      },
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);
    expect(result.openaiAsrBaseUrl).toBe(DEFAULT_SETTINGS.openaiAsrBaseUrl);
    expect(result.openaiAsrModel).toBe(DEFAULT_SETTINGS.openaiAsrModel);
    expect(result.pillVisibility).toBe("Always");
  });

  test("null confidenceThreshold defaults to 0.55", () => {
    const snapshot = {
      settings: { ...makeBackendSettings(), confidenceThreshold: null },
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);
    expect(result.confidenceThreshold).toBe("0.55");
  });

  test("null llmSkipThreshold defaults to 0.40", () => {
    const snapshot = {
      settings: { ...makeBackendSettings(), llmSkipThreshold: null },
    } as WorkspaceSnapshot;

    const result = mapWorkspaceSettings(snapshot);
    expect(result.llmSkipThreshold).toBe("0.40");
  });

  // ── settingsToSaveInput ─────────────────────────────────────────────

  test("converts workspace settings to save input", () => {
    const settings = { ...DEFAULT_SETTINGS };
    const input = settingsToSaveInput(settings);

    expect(input.activationMode).toBe("Toggle");
    expect(input.confidenceThreshold).toBe(0.55);
    expect(input.llmSkipThreshold).toBe(0.4);
    expect(input.speechThreshold).toBe(0.5);
    expect(input.openaiAsrApiKey).toBeNull();
  });

  test("preserves llmSkipThreshold below confidenceThreshold on save", () => {
    const settings = { ...DEFAULT_SETTINGS, confidenceThreshold: "0.75", llmSkipThreshold: "0.40" };
    const input = settingsToSaveInput(settings);
    expect(input.llmSkipThreshold).toBe(0.4);
  });

  test("does not send masked API key back to backend", () => {
    const settings = { ...DEFAULT_SETTINGS, openaiAsrApiKey: "••••••••" };
    const input = settingsToSaveInput(settings);
    expect(input.openaiAsrApiKey).toBeNull();
  });

  test("sends real API key to backend when user provides one", () => {
    const settings = { ...DEFAULT_SETTINGS, openaiAsrApiKey: "sk-abc123" };
    const input = settingsToSaveInput(settings);
    expect(input.openaiAsrApiKey).toBe("sk-abc123");
  });

  test("empty optional strings become null", () => {
    const settings = { ...DEFAULT_SETTINGS, openaiAsrBaseUrl: "", openaiAsrModel: "" };
    const input = settingsToSaveInput(settings);
    expect(input.openaiAsrBaseUrl).toBeNull();
    expect(input.openaiAsrModel).toBeNull();
  });

  // ── summarize helpers ───────────────────────────────────────────────

  test("summarizeAsrBackend strips suffixes", () => {
    const result = summarizeAsrBackend("Qwen3Asr");
    expect(result).not.toContain("(recommended)");
    expect(result.length).toBeGreaterThan(0);
  });

  test("summarizeDictationMode strips suffix", () => {
    const result = summarizeDictationMode("Clipboard");
    expect(result).not.toContain("into app");
    expect(result.length).toBeGreaterThan(0);
  });
});
