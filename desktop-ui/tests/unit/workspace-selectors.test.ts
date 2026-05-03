import { beforeEach, describe, expect, test } from "bun:test";
import {
  getSessionStats,
  getTodayStats,
  getVisibleHistorySessions,
  getVisibleSessionCount,
  loadWorkspaceSnapshot,
  workspaceActions,
  workspaceStore,
} from "../../src/store/workspace-store";
import { DEFAULT_RUNTIME, DEFAULT_SETTINGS, type WorkspaceState } from "../../src/store/workspace-types";
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
      asrBackendType: "Qwen3Asr",
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
      openaiAsrApiKeySet: false,
    },
    history: [
      {
        id: "s1",
        startedAt: new Date().toISOString(),
        endedAt: new Date(Date.now() + 10_000).toISOString(),
        transcript: "Hello world",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
      {
        id: "s2",
        startedAt: new Date().toISOString(),
        endedAt: new Date(Date.now() + 60_000).toISOString(),
        transcript: "Testing the search feature is important",
        backend: "Qwen3Asr",
        style: "professional",
        status: "completed",
      },
      {
        id: "s3",
        startedAt: "2025-01-01T00:00:00.000Z",
        endedAt: "2025-01-01T00:00:30.000Z",
        transcript: "Old session from last year",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
    ],
    dictionary: [],
    stylePresets: [],
    ...overrides,
  };
}

describe("workspace selectors", () => {
  beforeEach(async () => {
    workspaceStore.setState(() => ({
      sidebarCollapsed: false,
      mobileMenuOpen: false,
      historySearch: "",
      dictionaryEntries: [],
      historySessions: [],
      stylePresets: [],
      settings: { ...DEFAULT_SETTINGS },
      runtime: { ...DEFAULT_RUNTIME },
    }));
  });

  // ── getVisibleHistorySessions ─────────────────────────────────────────

  test("returns all sessions when search is empty", () => {
    workspaceActions.hydrate(makeSnapshot());
    const visible = getVisibleHistorySessions(workspaceStore.state);
    expect(visible).toHaveLength(3);
  });

  test("filters sessions by transcript text (case insensitive)", () => {
    workspaceActions.hydrate(makeSnapshot());

    workspaceActions.setHistorySearch("search");
    const visible = getVisibleHistorySessions(workspaceStore.state);
    expect(visible).toHaveLength(1);
    expect(visible[0]!.id).toBe("s2");
  });

  test("returns empty array when no sessions match search", () => {
    workspaceActions.hydrate(makeSnapshot());

    workspaceActions.setHistorySearch("zzzzzznotfound");
    const visible = getVisibleHistorySessions(workspaceStore.state);
    expect(visible).toHaveLength(0);
  });

  test("getVisibleSessionCount matches filtered length", () => {
    workspaceActions.hydrate(makeSnapshot());

    workspaceActions.setHistorySearch("hello");
    expect(getVisibleSessionCount(workspaceStore.state)).toBe(1);
  });

  test("search trims whitespace", () => {
    workspaceActions.hydrate(makeSnapshot());

    workspaceActions.setHistorySearch("  hello  ");
    const visible = getVisibleHistorySessions(workspaceStore.state);
    expect(visible).toHaveLength(1);
  });

  // ── getSessionStats ───────────────────────────────────────────────────

  test("computes aggregate stats across all sessions", () => {
    workspaceActions.hydrate(makeSnapshot());
    const stats = getSessionStats(workspaceStore.state);

    expect(stats.count).toBe(3);
    expect(stats.totalWords).toBeGreaterThan(0);
    expect(stats.totalMinutes).toBeGreaterThanOrEqual(0);
  });

  test("returns zero stats for empty history", () => {
    workspaceActions.hydrate(makeSnapshot({ history: [] }));
    const stats = getSessionStats(workspaceStore.state);

    expect(stats.count).toBe(0);
    expect(stats.totalWords).toBe(0);
    expect(stats.totalMinutes).toBe(0);
  });

  test("memoizes stats (same reference for same sessions)", () => {
    workspaceActions.hydrate(makeSnapshot());
    const stats1 = getSessionStats(workspaceStore.state);
    const stats2 = getSessionStats(workspaceStore.state);
    expect(stats1).toBe(stats2);
  });

  // ── getTodayStats ─────────────────────────────────────────────────────

  test("counts only today's sessions", () => {
    workspaceActions.hydrate(makeSnapshot());
    const today = getTodayStats(workspaceStore.state);

    // s1 and s2 have today's date, s3 is from 2025
    expect(today.count).toBe(2);
    expect(today.words).toBeGreaterThan(0);
  });

  test("returns zeros when no sessions are from today", () => {
    workspaceActions.hydrate(
      makeSnapshot({
        history: [
          {
            id: "old",
            startedAt: "2020-01-01T00:00:00.000Z",
            endedAt: "2020-01-01T00:00:10.000Z",
            transcript: "Ancient session",
            backend: "Qwen3Asr",
            style: "casual",
            status: "completed",
          },
        ],
      }),
    );

    const today = getTodayStats(workspaceStore.state);
    expect(today.count).toBe(0);
    expect(today.words).toBe(0);
  });

  // ── Memoization invalidation ──────────────────────────────────────────

  test("selectors recompute when sessions change", () => {
    workspaceActions.hydrate(makeSnapshot());
    const stats1 = getSessionStats(workspaceStore.state);

    // Hydrate with different data
    workspaceActions.hydrate(makeSnapshot({ history: [] }));
    const stats2 = getSessionStats(workspaceStore.state);

    expect(stats1).not.toBe(stats2);
    expect(stats2.count).toBe(0);
  });
});
