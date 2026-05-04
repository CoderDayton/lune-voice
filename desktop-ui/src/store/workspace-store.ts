import { Store } from "@tanstack/store";
import { backendClient } from "../lib/backend/client";
import type { BackendWorkspaceSnapshot, SessionStreamEvent } from "../lib/backend/types";
import {
  DEFAULT_RUNTIME,
  DEFAULT_SETTINGS,
  type HistorySession,
  type WorkspaceRuntimeState,
  type WorkspaceSettings,
  type WorkspaceState,
} from "./workspace-types";
import { mapDictionaryEntry, mapHistoryEntry, mapWorkspaceSettings, settingsToSaveInput } from "./workspace-mappers";

// Re-export types and selectors for consumers
export type { HistorySession, StylePreset, WorkspaceSettings, WorkspaceState } from "./workspace-types";
export { summarizeAsrBackend, summarizeDictationMode } from "./workspace-mappers";

// ── Store ─────────────────────────────────────────────────────────────

export const workspaceStore = new Store<WorkspaceState>({
  sidebarCollapsed: false,
  mobileMenuOpen: false,
  historySearch: "",
  dictionaryEntries: [],
  historySessions: [],
  stylePresets: [],
  settings: { ...DEFAULT_SETTINGS },
  runtime: { ...DEFAULT_RUNTIME },
});

// ── Actions ───────────────────────────────────────────────────────────

let errorDismissTimer: ReturnType<typeof setTimeout> | null = null;

export const workspaceActions = {
  toggleSidebar() {
    workspaceStore.setState((s) => ({ ...s, sidebarCollapsed: !s.sidebarCollapsed }));
  },

  toggleMobileMenu() {
    workspaceStore.setState((s) => ({ ...s, mobileMenuOpen: !s.mobileMenuOpen }));
  },

  closeMobileMenu() {
    workspaceStore.setState((s) => ({ ...s, mobileMenuOpen: false }));
  },

  setHistorySearch(value: string) {
    workspaceStore.setState((s) => ({ ...s, historySearch: value }));
  },

  updateSetting<Key extends keyof WorkspaceSettings>(key: Key, value: WorkspaceSettings[Key]) {
    workspaceStore.setState((s) => ({
      ...s,
      settings: { ...s.settings, [key]: value },
    }));
  },

  setLoading(loading: boolean) {
    workspaceStore.setState((s) => ({
      ...s,
      runtime: { ...s.runtime, loading },
    }));
  },

  setError(error: string | null) {
    if (errorDismissTimer) {
      clearTimeout(errorDismissTimer);
      errorDismissTimer = null;
    }
    workspaceStore.setState((s) => ({
      ...s,
      runtime: { ...s.runtime, error },
    }));
    if (error) {
      errorDismissTimer = setTimeout(() => {
        errorDismissTimer = null;
        workspaceStore.setState((s) =>
          s.runtime.error === error ? { ...s, runtime: { ...s.runtime, error: null } } : s,
        );
      }, 8_000);
    }
  },

  hydrate(snapshot: BackendWorkspaceSnapshot) {
    workspaceStore.setState((s) => ({
      ...s,
      dictionaryEntries: snapshot.dictionary.map(mapDictionaryEntry),
      stylePresets: snapshot.stylePresets ?? [],
      historySessions: snapshot.history
        .filter((entry) => entry.transcript)
        .map(mapHistoryEntry)
        .filter((entry): entry is HistorySession => entry !== null),
      settings: mapWorkspaceSettings(snapshot),
      runtime: {
        ipcFlavor: snapshot.ipcFlavor,
        serviceStatus: snapshot.serviceStatus,
        activeSessionId: isActiveServiceStatus(snapshot.serviceStatus)
          ? s.runtime.activeSessionId
          : null,
        liveTranscript: isActiveServiceStatus(snapshot.serviceStatus)
          ? s.runtime.liveTranscript
          : "",
        audioLevel: isActiveServiceStatus(snapshot.serviceStatus) ? s.runtime.audioLevel : 0,
        loaded: true,
        loading: false,
        error: null,
      },
    }));
  },

  applySessionEvent(event: SessionStreamEvent) {
    workspaceStore.setState((s) => {
      const runtime = reduceSessionEvent(s.runtime, event);
      if (runtime === s.runtime) return s;
      return { ...s, runtime };
    });

    // Session timeout watchdog: if we enter an active state, schedule a
    // forced reconciliation in case the backend never sends sessionStopped.
    if (event.type === "sessionStarted") {
      startSessionTimeoutWatchdog();
    } else if (event.type === "sessionStopped" || event.type === "sessionError") {
      clearSessionTimeoutWatchdog();
    }
  },
};

// ── Session timeout watchdog ──────────────────────────────────────────
// If a session is active for over 3 minutes without completing (backend
// hang, missed event, etc.), force-reconcile by reloading the snapshot.
const SESSION_TIMEOUT_MS = 180_000;
let sessionTimeoutTimer: ReturnType<typeof setTimeout> | null = null;

function startSessionTimeoutWatchdog() {
  clearSessionTimeoutWatchdog();
  sessionTimeoutTimer = setTimeout(() => {
    sessionTimeoutTimer = null;
    const { runtime } = workspaceStore.state;
    if (runtime.activeSessionId) {
      console.warn("[workspace] session timeout — forcing state reconciliation");
      workspaceStore.setState((s) => ({
        ...s,
        runtime: {
          ...s.runtime,
          activeSessionId: null,
          serviceStatus: "ready",
          audioLevel: 0,
          error: "Session timed out — backend may be unresponsive",
        },
      }));
    }
  }, SESSION_TIMEOUT_MS);
}

function clearSessionTimeoutWatchdog() {
  if (sessionTimeoutTimer) {
    clearTimeout(sessionTimeoutTimer);
    sessionTimeoutTimer = null;
  }
}

// ── Async operations ──────────────────────────────────────────────────

export async function loadWorkspaceSnapshot() {
  workspaceActions.setLoading(true);
  workspaceActions.setError(null);
  try {
    const snapshot = await backendClient.loadWorkspaceSnapshot();
    workspaceActions.hydrate(snapshot);
    return snapshot;
  } catch (error) {
    workspaceActions.setLoading(false);
    workspaceActions.setError(formatError(error));
    throw error;
  }
}

export async function saveWorkspaceSetting<Key extends keyof WorkspaceSettings>(
  key: Key,
  value: WorkspaceSettings[Key],
): Promise<boolean> {
  const previousSettings = workspaceStore.state.settings;
  workspaceActions.updateSetting(key, value);
  try {
    const snapshot = await backendClient.saveWorkspaceSettings(
      settingsToSaveInput(workspaceStore.state.settings),
    );
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    try {
      const snapshot = await backendClient.loadWorkspaceSnapshot();
      workspaceActions.hydrate(snapshot);
    } catch (reloadError) {
      console.warn("[workspace] failed to reload snapshot after save failure", reloadError);
      workspaceStore.setState((s) => ({ ...s, settings: previousSettings }));
    }
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function addDictionaryEntry(phrase: string, replacement: string, priority = 5) {
  const p = phrase.trim();
  const r = replacement.trim();
  if (!p || !r) return false;
  try {
    const snapshot = await backendClient.createDictionaryEntry(p, r, priority);
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function updateDictionaryEntry(
  id: number,
  phrase: string,
  replacement: string,
  priority: number,
) {
  const p = phrase.trim();
  const r = replacement.trim();
  if (!p || !r) return false;
  try {
    const snapshot = await backendClient.updateDictionaryEntry(id, p, r, priority);
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function removeDictionaryEntry(id: number) {
  try {
    const snapshot = await backendClient.deleteDictionaryEntry(id);
    workspaceActions.hydrate(snapshot);
  } catch (error) {
    workspaceActions.setError(formatError(error));
  }
}

async function dictationAction<T>(action: () => Promise<T>, fallback: T): Promise<T> {
  workspaceActions.setError(null);
  try {
    return await action();
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return fallback;
  }
}

export async function startDictation() {
  const sessionId = await dictationAction(() => backendClient.startDictation(), null);
  if (sessionId) {
    workspaceActions.applySessionEvent({ type: "sessionStarted", sessionId });
  }
  return sessionId;
}

export async function stopDictation() {
  await dictationAction(() => backendClient.stopDictation(), undefined);
}

export async function cancelDictation() {
  await dictationAction(() => backendClient.cancelDictation(), undefined);
}

export async function toggleDictation() {
  workspaceActions.setError(null);
  try {
    const sessionId = await backendClient.toggleDictation();
    workspaceActions.applySessionEvent(
      sessionId ? { type: "sessionStarted", sessionId } : { type: "sessionStopped" },
    );
    return sessionId;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return null;
  }
}

// ── Selectors ─────────────────────────────────────────────────────────

let _visibleCache: { sessions: HistorySession[]; query: string; result: HistorySession[] } | null = null;

export function getVisibleHistorySessions(state: WorkspaceState) {
  const query = state.historySearch.trim().toLowerCase();
  if (!query) return state.historySessions;
  if (_visibleCache && _visibleCache.sessions === state.historySessions && _visibleCache.query === query) {
    return _visibleCache.result;
  }
  const result = state.historySessions.filter((s) => s.transcript.toLowerCase().includes(query));
  _visibleCache = { sessions: state.historySessions, query, result };
  return result;
}

export function getVisibleSessionCount(state: WorkspaceState) {
  return getVisibleHistorySessions(state).length;
}

// Memoized aggregate stats — recomputed only when historySessions reference changes.
export interface SessionStats {
  totalWords: number;
  totalMinutes: number;
  count: number;
}

let _statsCache: { sessions: HistorySession[]; result: SessionStats } | null = null;

export function getSessionStats(state: WorkspaceState): SessionStats {
  const sessions = state.historySessions;
  if (_statsCache && _statsCache.sessions === sessions) return _statsCache.result;
  let words = 0;
  let seconds = 0;
  for (const s of sessions) { words += s.words; seconds += s.seconds; }
  const result = { totalWords: words, totalMinutes: Math.round(seconds / 60), count: sessions.length };
  _statsCache = { sessions, result };
  return result;
}

// Memoized today-stats — recomputed only when historySessions reference or date changes.
export interface TodayStats {
  count: number;
  words: number;
}

let _todayCache: { sessions: HistorySession[]; date: string; result: TodayStats } | null = null;

export function getTodayStats(state: WorkspaceState): TodayStats {
  const sessions = state.historySessions;
  const today = new Date().toDateString();
  if (_todayCache && _todayCache.sessions === sessions && _todayCache.date === today) return _todayCache.result;
  let count = 0;
  let words = 0;
  for (const s of sessions) {
    if (new Date(s.startedAt).toDateString() === today) { count++; words += s.words; }
  }
  const result = { count, words };
  _todayCache = { sessions, date: today, result };
  return result;
}

// ── Internal helpers ──────────────────────────────────────────────────

function formatError(error: unknown) {
  return error instanceof Error ? error.message : "Unable to reach the desktop backend.";
}

function reduceSessionEvent(
  runtime: WorkspaceRuntimeState,
  event: SessionStreamEvent,
): WorkspaceRuntimeState {
  switch (event.type) {
    case "serviceStatus":
      return { ...runtime, serviceStatus: event.status };
    case "sessionStarted":
      return {
        ...runtime,
        activeSessionId: event.sessionId,
        serviceStatus: "listening",
        liveTranscript: "",
        audioLevel: 0,
        error: null,
      };
    case "processing":
      return { ...runtime, serviceStatus: "processing", audioLevel: 0 };
    case "finalizedText":
      return {
        ...runtime,
        serviceStatus: "ready",
        liveTranscript: runtime.liveTranscript
          ? `${runtime.liveTranscript} ${event.text}`
          : event.text,
      };
    case "correctedText":
      return { ...runtime, liveTranscript: event.text };
    case "sessionStopped":
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: runtime.serviceStatus === "error" ? "error" : "ready",
        audioLevel: 0,
      };
    case "sessionError":
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: "error",
        audioLevel: 0,
        error: event.message,
      };
    case "audioLevel":
      return { ...runtime, audioLevel: Math.max(0, Math.min(1, event.level)) };
    case "partialText":
    case "commandDetected":
      return runtime;
  }
}

function isActiveServiceStatus(status: string) {
  return status === "listening" || status === "processing";
}
