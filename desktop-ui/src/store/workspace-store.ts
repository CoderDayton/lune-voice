import { Store } from '@tanstack/store';
import { backendClient } from '../lib/backend/client';
import type { FormatTogglesDto, SessionStreamEvent, WorkspaceSnapshot } from '../lib/backend/types';
import {
  mapDictionaryEntry,
  mapHistoryEntry,
  mapWorkspaceSettings,
  settingsToSaveInput,
} from './workspace-mappers';
import {
  DEFAULT_RUNTIME,
  DEFAULT_SETTINGS,
  type HistorySession,
  type WorkspaceRuntimeState,
  type WorkspaceSettings,
  type WorkspaceState,
} from './workspace-types';

export { summarizeAsrBackend, summarizeDictationMode } from './workspace-mappers';
// Re-export types and selectors for consumers
export type {
  HistorySession,
  StylePreset,
  WorkspaceSettings,
  WorkspaceState,
} from './workspace-types';

// ── Store ─────────────────────────────────────────────────────────────

export const workspaceStore = new Store<WorkspaceState>({
  sidebarCollapsed: false,
  mobileMenuOpen: false,
  historySearch: '',
  historyPage: 1,
  historyPageSize: 20,
  dictionaryEntries: [],
  historySessions: [],
  historyPageItems: [],
  historyTotal: 0,
  historyLoading: false,
  sessionTotals: { sessions: 0, words: 0, seconds: 0 },
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
    workspaceStore.setState((s) => ({ ...s, historySearch: value, historyPage: 1 }));
    // Debounce: one backend query per pause in typing, not per keystroke.
    if (_historySearchDebounce) clearTimeout(_historySearchDebounce);
    _historySearchDebounce = setTimeout(() => {
      void fetchHistoryPage();
    }, 250);
  },

  setHistoryPage(page: number) {
    workspaceStore.setState((s) => {
      const totalPages = Math.max(1, Math.ceil(s.historyTotal / s.historyPageSize));
      const clamped = Math.min(Math.max(1, Math.floor(page)), totalPages);
      return clamped === s.historyPage ? s : { ...s, historyPage: clamped };
    });
    void fetchHistoryPage();
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

  hydrate(snapshot: WorkspaceSnapshot) {
    workspaceStore.setState((s) => ({
      ...s,
      dictionaryEntries: snapshot.dictionary.map(mapDictionaryEntry),
      stylePresets: snapshot.stylePresets ?? [],
      historySessions: snapshot.history
        .filter((entry) => entry.transcript)
        .map(mapHistoryEntry)
        .filter((entry): entry is HistorySession => entry !== null),
      sessionTotals: {
        sessions: snapshot.stats.sessions,
        words: snapshot.stats.words,
        seconds: snapshot.stats.seconds,
      },
      settings: mapWorkspaceSettings(snapshot),
      runtime: {
        ipcFlavor: snapshot.ipcFlavor,
        serviceStatus: snapshot.serviceStatus,
        activeSessionId: isActiveServiceStatus(snapshot.serviceStatus)
          ? s.runtime.activeSessionId
          : null,
        liveTranscript: isActiveServiceStatus(snapshot.serviceStatus)
          ? s.runtime.liveTranscript
          : '',
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
    if (event.type === 'sessionStarted') {
      startSessionTimeoutWatchdog();
    } else if (event.type === 'sessionStopped' || event.type === 'sessionError') {
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
      console.warn('[workspace] session timeout — forcing state reconciliation');
      workspaceStore.setState((s) => ({
        ...s,
        runtime: {
          ...s.runtime,
          activeSessionId: null,
          serviceStatus: 'ready',
          audioLevel: 0,
          error: 'Session timed out — backend may be unresponsive',
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
      console.warn('[workspace] failed to reload snapshot after save failure', reloadError);
      workspaceStore.setState((s) => ({ ...s, settings: previousSettings }));
    }
    workspaceActions.setError(formatError(error));
    return false;
  }
}

/// Update several settings and persist them in a single save. Used when two
/// fields must move together (e.g. the styling switch enables both the style
/// preset and the local engine) so they never land in the store desynced.
export async function saveWorkspaceSettings(patch: Partial<WorkspaceSettings>): Promise<boolean> {
  const previousSettings = workspaceStore.state.settings;
  for (const [key, value] of Object.entries(patch)) {
    workspaceActions.updateSetting(key as keyof WorkspaceSettings, value as never);
  }
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
      console.warn('[workspace] failed to reload snapshot after save failure', reloadError);
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

export async function removeHistorySession(sessionId: string) {
  try {
    const snapshot = await backendClient.deleteSession(sessionId);
    workspaceActions.hydrate(snapshot);
    // The snapshot refreshes the capped `historySessions`/stats, but the
    // server-paged view is separate — refetch so the current page and total
    // reflect the deletion.
    void fetchHistoryPage();
  } catch (error) {
    workspaceActions.setError(formatError(error));
  }
}

// ── History pagination (server-side) ──────────────────────────────────
// The History tab pages the full sessions table on the backend
// (`list_history_page`) instead of the capped snapshot slice, so it can reach
// every session. `_historyReqSeq` discards out-of-order responses when the user
// pages or types quickly.
let _historyReqSeq = 0;
let _historySearchDebounce: ReturnType<typeof setTimeout> | null = null;

export async function fetchHistoryPage() {
  const { historyPage, historyPageSize, historySearch } = workspaceStore.state;
  const seq = ++_historyReqSeq;
  workspaceStore.setState((s) => ({ ...s, historyLoading: true }));
  try {
    const res = await backendClient.listHistoryPage(
      historyPage,
      historyPageSize,
      historySearch.trim(),
    );
    if (seq !== _historyReqSeq) return; // superseded by a newer request
    // The current page can fall past the end (e.g. deleting the last row on the
    // last page). Clamp and refetch so the user never lands on a blank page.
    const maxPage = Math.max(1, Math.ceil(res.total / historyPageSize));
    if (res.total > 0 && historyPage > maxPage) {
      workspaceStore.setState((s) => ({ ...s, historyPage: maxPage }));
      void fetchHistoryPage();
      return;
    }
    const items = res.sessions.map(mapHistoryEntry).filter((e): e is HistorySession => e !== null);
    workspaceStore.setState((s) => ({
      ...s,
      historyPageItems: items,
      historyTotal: res.total,
      historyLoading: false,
    }));
  } catch (error) {
    if (seq !== _historyReqSeq) return;
    workspaceActions.setError(formatError(error));
    workspaceStore.setState((s) => ({ ...s, historyLoading: false }));
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

export async function createCustomStyle(
  name: string,
  description: string,
  instructions: string,
  formatting: FormatTogglesDto,
): Promise<boolean> {
  if (!name.trim()) return false;
  try {
    const snapshot = await backendClient.createCustomStyle(
      name.trim(),
      description.trim(),
      instructions.trim(),
      formatting,
    );
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function updateCustomStyle(
  id: string,
  name: string,
  description: string,
  instructions: string,
  formatting: FormatTogglesDto,
): Promise<boolean> {
  if (!name.trim()) return false;
  try {
    const snapshot = await backendClient.updateCustomStyle(
      id,
      name.trim(),
      description.trim(),
      instructions.trim(),
      formatting,
    );
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function deleteCustomStyle(styleId: string): Promise<boolean> {
  try {
    const snapshot = await backendClient.deleteCustomStyle(styleId);
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
  }
}

export async function generateStyleExamples(styleId: string): Promise<boolean> {
  try {
    const snapshot = await backendClient.generateStyleExamples(styleId);
    workspaceActions.hydrate(snapshot);
    return true;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return false;
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
    workspaceActions.applySessionEvent({ type: 'sessionStarted', sessionId });
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
      sessionId ? { type: 'sessionStarted', sessionId } : { type: 'sessionStopped' },
    );
    return sessionId;
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return null;
  }
}

// ── Selectors ─────────────────────────────────────────────────────────

// Page count derived from the server-reported total (History tab).
export function getHistoryServerPageCount(state: WorkspaceState) {
  return Math.max(1, Math.ceil(state.historyTotal / state.historyPageSize));
}

// Memoized aggregate stats — recomputed only when historySessions reference changes.
export interface SessionStats {
  totalWords: number;
  totalMinutes: number;
  count: number;
}

let _statsCache: { totals: WorkspaceState['sessionTotals']; result: SessionStats } | null = null;

export function getSessionStats(state: WorkspaceState): SessionStats {
  const totals = state.sessionTotals;
  if (_statsCache && _statsCache.totals === totals) return _statsCache.result;
  const result = {
    totalWords: totals.words,
    totalMinutes: Math.round(totals.seconds / 60),
    count: totals.sessions,
  };
  _statsCache = { totals, result };
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
  if (_todayCache && _todayCache.sessions === sessions && _todayCache.date === today)
    return _todayCache.result;
  let count = 0;
  let words = 0;
  for (const s of sessions) {
    if (new Date(s.startedAt).toDateString() === today) {
      count++;
      words += s.words;
    }
  }
  const result = { count, words };
  _todayCache = { sessions, date: today, result };
  return result;
}

// ── Internal helpers ──────────────────────────────────────────────────

function formatError(error: unknown) {
  // Tauri rejects a backend `Err(String)` as a plain string (not an Error), so
  // surface those directly instead of masking them with the generic fallback.
  if (error instanceof Error) return error.message;
  if (typeof error === 'string' && error.trim().length > 0) return error;
  return 'Unable to reach the desktop backend.';
}

function reduceSessionEvent(
  runtime: WorkspaceRuntimeState,
  event: SessionStreamEvent,
): WorkspaceRuntimeState {
  switch (event.type) {
    case 'serviceStatus':
      return { ...runtime, serviceStatus: event.status };
    case 'sessionStarted':
      return {
        ...runtime,
        activeSessionId: event.sessionId,
        serviceStatus: 'listening',
        liveTranscript: '',
        audioLevel: 0,
        error: null,
      };
    case 'processing':
      return { ...runtime, serviceStatus: 'processing', audioLevel: 0 };
    case 'finalizedText':
      return {
        ...runtime,
        serviceStatus: 'ready',
        liveTranscript: runtime.liveTranscript
          ? `${runtime.liveTranscript} ${event.text}`
          : event.text,
      };
    case 'correctedText':
      return { ...runtime, liveTranscript: event.text };
    case 'sessionStopped':
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: runtime.serviceStatus === 'error' ? 'error' : 'ready',
        audioLevel: 0,
      };
    case 'sessionError':
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: 'error',
        audioLevel: 0,
        error: event.message,
      };
    case 'sessionWarning':
      // Non-fatal: surface the message in the toast but leave serviceStatus
      // (the status badge) untouched, so it never flashes red. Session keeps
      // running, so activeSessionId is not cleared.
      return { ...runtime, error: event.message };
    case 'audioLevel':
      return { ...runtime, audioLevel: Math.max(0, Math.min(1, event.level)) };
    case 'partialText':
    case 'commandDetected':
      return runtime;
  }
}

function isActiveServiceStatus(status: string) {
  return status === 'listening' || status === 'processing';
}
