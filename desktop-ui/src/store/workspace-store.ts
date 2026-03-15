import { Store } from "@tanstack/store";
import { backendClient } from "../lib/backend/client";
import type {
  BackendDictionaryEntry,
  BackendHistoryEntry,
  BackendWorkspaceSnapshot,
  SessionStreamEvent,
} from "../lib/backend/types";
import type { DictionaryEntry } from "../features/dictionary/data/dictionary-data";
import {
  labelForAsrBackend,
  labelForDeliveryMethod,
  labelForLlmBackend,
  labelForStylePreset,
  type ActivationModeValue,
  type AsrBackendValue,
  type DictationModeValue,
  type LlmBackendValue,
  type PillVisibilityValue,
} from "../features/setup/data/settings-data";

export interface WorkspaceSettings {
  activationMode: ActivationModeValue;
  asrBackend: AsrBackendValue;
  useGpu: boolean;
  confidenceThreshold: string;
  llmCorrection: boolean;
  llmBackend: LlmBackendValue;
  dictationMode: DictationModeValue;
  stylePreset: string;
  adaptiveVad: boolean;
  speechThreshold: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  alignerEnabled: boolean;
  pillVisibility: PillVisibilityValue;
  // OpenAI ASR fields
  openaiAsrBaseUrl: string;
  openaiAsrApiKey: string;
  openaiAsrModel: string;
  // OpenAI LLM fields
  openaiLlmBaseUrl: string;
  openaiLlmApiKey: string;
  openaiLlmModel: string;
}

export interface HistorySession {
  id: string;
  startedAt: string;
  endedAt: string | null;
  transcript: string;
  backend: string;
  style: string;
  status: string;
  words: number;
  seconds: number;
}

interface WorkspaceRuntimeState {
  ipcFlavor: string;
  serviceStatus: string;
  activeSessionId: string | null;
  liveTranscript: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

interface WorkspaceState {
  sidebarCollapsed: boolean;
  mobileMenuOpen: boolean;
  historySearch: string;
  dictionaryEntries: DictionaryEntry[];
  historySessions: HistorySession[];
  settings: WorkspaceSettings;
  runtime: WorkspaceRuntimeState;
}

const DEFAULT_SETTINGS: WorkspaceSettings = {
  activationMode: "Toggle",
  asrBackend: "OrtQwenAsr",
  useGpu: true,
  confidenceThreshold: "0.50",
  llmCorrection: true,
  llmBackend: "LlamaCppLocal",
  dictationMode: "Clipboard",
  stylePreset: "casual",
  adaptiveVad: false,
  speechThreshold: "0.50",
  hotkeyToggle: "Ctrl+Space",
  hotkeyCancel: "Escape",
  alignerEnabled: true,
  pillVisibility: "Always",
  openaiAsrBaseUrl: "https://api.openai.com/v1",
  openaiAsrApiKey: "",
  openaiAsrModel: "whisper-1",
  openaiLlmBaseUrl: "https://api.openai.com/v1",
  openaiLlmApiKey: "",
  openaiLlmModel: "gpt-4o-mini",
};

const DEFAULT_RUNTIME: WorkspaceRuntimeState = {
  ipcFlavor: "unknown",
  serviceStatus: "Connecting",
  activeSessionId: null,
  liveTranscript: "",
  loaded: false,
  loading: false,
  error: null,
};


export function createInitialWorkspaceState(): WorkspaceState {
  return {
    sidebarCollapsed: false,
    mobileMenuOpen: false,
    historySearch: "",
    dictionaryEntries: [],
    historySessions: [],
    settings: { ...DEFAULT_SETTINGS },
    runtime: { ...DEFAULT_RUNTIME },
  };
}

export const workspaceStore = new Store(createInitialWorkspaceState());

export const workspaceActions = {
  toggleSidebar() {
    workspaceStore.setState((state) => ({ ...state, sidebarCollapsed: !state.sidebarCollapsed }));
  },

  toggleMobileMenu() {
    workspaceStore.setState((state) => ({ ...state, mobileMenuOpen: !state.mobileMenuOpen }));
  },

  closeMobileMenu() {
    workspaceStore.setState((state) => ({ ...state, mobileMenuOpen: false }));
  },

  setHistorySearch(value: string) {
    workspaceStore.setState((state) => ({ ...state, historySearch: value }));
  },

  updateSetting<Key extends keyof WorkspaceSettings>(key: Key, value: WorkspaceSettings[Key]) {
    workspaceStore.setState((state) => ({
      ...state,
      settings: {
        ...state.settings,
        [key]: value,
      },
    }));
  },

  setLoading(loading: boolean) {
    workspaceStore.setState((state) => ({
      ...state,
      runtime: {
        ...state.runtime,
        loading,
      },
    }));
  },

  setError(error: string | null) {
    workspaceStore.setState((state) => ({
      ...state,
      runtime: {
        ...state.runtime,
        error,
      },
    }));
  },

  hydrate(snapshot: BackendWorkspaceSnapshot) {
    workspaceStore.setState((state) => ({
      ...state,
      dictionaryEntries: snapshot.dictionary.map(mapDictionaryEntry),
      historySessions: snapshot.history
        .filter((entry) => entry.transcript)
        .map(mapHistoryEntry)
        .filter((entry): entry is HistorySession => entry !== null),
      settings: mapWorkspaceSettings(snapshot),
      runtime: {
        ipcFlavor: snapshot.ipcFlavor,
        serviceStatus: snapshot.serviceStatus,
        activeSessionId: state.runtime.activeSessionId,
        liveTranscript: state.runtime.liveTranscript,
        loaded: true,
        loading: false,
        error: null,
      },
    }));
  },

  applySessionEvent(event: SessionStreamEvent) {
    workspaceStore.setState((state) => ({
      ...state,
      runtime: reduceSessionEvent(state.runtime, event),
    }));
  },
};

export async function loadWorkspaceSnapshot() {
  workspaceActions.setLoading(true);
  workspaceActions.setError(null);

  try {
    return await hydrateFromSnapshot(() => backendClient.loadWorkspaceSnapshot());
  } catch (error) {
    workspaceActions.setLoading(false);
    workspaceActions.setError(formatError(error));
    throw error;
  }
}

export async function persistWorkspaceSettings() {
  const { settings } = workspaceStore.state;
  workspaceActions.setError(null);

  const snapshot = await backendClient.saveWorkspaceSettings({
    activationMode: settings.activationMode,
    dictationMode: settings.dictationMode,
    useGpu: settings.useGpu,
    asrBackendType: settings.asrBackend,
    llmBackendType: settings.llmBackend,
    llmCorrection: settings.llmCorrection,
    confidenceThreshold: settings.llmCorrection ? Number(settings.confidenceThreshold) : null,
    adaptiveVad: settings.adaptiveVad,
    speechThreshold: Number(settings.speechThreshold),
    stylePreset: settings.stylePreset,
    hotkeyToggle: settings.hotkeyToggle,
    hotkeyCancel: settings.hotkeyCancel,
    alignerEnabled: settings.alignerEnabled,
    pillVisibility: settings.pillVisibility,
    openaiAsrBaseUrl: settings.openaiAsrBaseUrl || null,
    openaiAsrApiKey: settings.openaiAsrApiKey || null,
    openaiAsrModel: settings.openaiAsrModel || null,
    openaiLlmBaseUrl: settings.openaiLlmBaseUrl || null,
    openaiLlmApiKey: settings.openaiLlmApiKey || null,
    openaiLlmModel: settings.openaiLlmModel || null,
  });

  workspaceActions.hydrate(snapshot);
  return snapshot;
}

export async function saveWorkspaceSetting<Key extends keyof WorkspaceSettings>(
  key: Key,
  value: WorkspaceSettings[Key],
) {
  workspaceActions.updateSetting(key, value);

  try {
    await persistWorkspaceSettings();
  } catch (error) {
    workspaceActions.setError(formatError(error));
  }
}

export async function addDictionaryEntry(phrase: string, replacement: string) {
  const normalizedPhrase = phrase.trim();
  const normalizedReplacement = replacement.trim();

  if (!normalizedPhrase || !normalizedReplacement) {
    return false;
  }

  return withWorkspaceError(
    async () => {
      await hydrateFromSnapshot(() =>
        backendClient.createDictionaryEntry(
          normalizedPhrase,
          normalizedReplacement,
          5,
        ),
      );
      return true;
    },
    false,
  );
}

export async function removeDictionaryEntry(id: number) {
  await withWorkspaceError(
    () => hydrateFromSnapshot(() => backendClient.deleteDictionaryEntry(id)),
    undefined,
  );
}

export async function startDictation() {
  return runDictationAction(() => backendClient.startDictation(), null);
}

export async function stopDictation() {
  await runDictationAction(() => backendClient.stopDictation(), undefined);
}

export async function cancelDictation() {
  await runDictationAction(() => backendClient.cancelDictation(), undefined);
}

export async function toggleDictation() {
  return runDictationAction(() => backendClient.toggleDictation(), null);
}

export function getVisibleHistorySessions(state: WorkspaceState) {
  const query = state.historySearch.trim().toLowerCase();

  if (!query) {
    return state.historySessions;
  }

  return state.historySessions.filter((session) =>
    session.transcript.toLowerCase().includes(query),
  );
}

export function summarizeAsrBackend(value: AsrBackendValue) {
  return labelForAsrBackend(value).replace(" (recommended)", "");
}

export function formatBackendLabel(value: string) {
  return labelForAsrBackend(value as AsrBackendValue)
    .replace(" (recommended)", "")
    .replace(" (OpenAI-compatible)", "");
}

export function summarizeDictationMode(value: DictationModeValue) {
  return labelForDeliveryMethod(value).replace(" into app", "");
}

export const summarizeLlmBackend = labelForLlmBackend;
export const summarizeStylePreset = labelForStylePreset;

function mapHistoryEntry(entry: BackendHistoryEntry): HistorySession | null {
  if (!entry.transcript) {
    return null;
  }

  const startedAt = new Date(entry.startedAt);
  const endedAt = entry.endedAt ? new Date(entry.endedAt) : null;
  const seconds = endedAt
    ? Math.max(1, Math.round((endedAt.getTime() - startedAt.getTime()) / 1000))
    : 0;

  return {
    id: entry.id,
    startedAt: entry.startedAt,
    endedAt: entry.endedAt,
    transcript: entry.transcript,
    backend: entry.backend,
    style: entry.style,
    status: entry.status,
    words: countWords(entry.transcript),
    seconds,
  };
}

function mapDictionaryEntry(entry: BackendDictionaryEntry): DictionaryEntry {
  return entry;
}

function mapWorkspaceSettings(snapshot: BackendWorkspaceSnapshot): WorkspaceSettings {
  return {
    activationMode: snapshot.settings.activationMode,
    asrBackend: snapshot.settings.asrBackendType,
    useGpu: snapshot.settings.useGpu,
    confidenceThreshold: formatThreshold(snapshot.settings.confidenceThreshold ?? 0.5),
    llmCorrection: snapshot.settings.confidenceThreshold !== null,
    llmBackend: snapshot.settings.llmBackendType,
    dictationMode: snapshot.settings.dictationMode,
    stylePreset: snapshot.settings.stylePreset,
    adaptiveVad: snapshot.settings.adaptiveVad,
    speechThreshold: formatThreshold(snapshot.settings.speechThreshold),
    hotkeyToggle: snapshot.settings.hotkeyToggle,
    hotkeyCancel: snapshot.settings.hotkeyCancel,
    alignerEnabled: snapshot.settings.alignerEnabled,
    pillVisibility: (snapshot.settings.pillVisibility ?? "Always") as PillVisibilityValue,
    openaiAsrBaseUrl: snapshot.settings.openaiAsrBaseUrl ?? DEFAULT_SETTINGS.openaiAsrBaseUrl,
    openaiAsrApiKey: snapshot.settings.openaiAsrApiKey ?? DEFAULT_SETTINGS.openaiAsrApiKey,
    openaiAsrModel: snapshot.settings.openaiAsrModel ?? DEFAULT_SETTINGS.openaiAsrModel,
    openaiLlmBaseUrl: snapshot.settings.openaiLlmBaseUrl ?? DEFAULT_SETTINGS.openaiLlmBaseUrl,
    openaiLlmApiKey: snapshot.settings.openaiLlmApiKey ?? DEFAULT_SETTINGS.openaiLlmApiKey,
    openaiLlmModel: snapshot.settings.openaiLlmModel ?? DEFAULT_SETTINGS.openaiLlmModel,
  };
}

function countWords(text: string) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function formatThreshold(value: number) {
  return value.toFixed(2);
}

function formatError(error: unknown) {
  return error instanceof Error ? error.message : "Unable to reach the desktop backend.";
}

async function hydrateFromSnapshot(loadSnapshot: () => Promise<BackendWorkspaceSnapshot>) {
  const snapshot = await loadSnapshot();
  workspaceActions.hydrate(snapshot);
  return snapshot;
}

async function withWorkspaceError<T>(action: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await action();
  } catch (error) {
    workspaceActions.setError(formatError(error));
    return fallback;
  }
}

async function runDictationAction<T>(action: () => Promise<T>, fallback: T): Promise<T> {
  workspaceActions.setError(null);
  return withWorkspaceError(action, fallback);
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
        error: null,
      };
    case "processing":
      return { ...runtime, serviceStatus: "processing" };
    case "finalizedText":
      return {
        ...runtime,
        serviceStatus: "ready",
        liveTranscript: [runtime.liveTranscript, event.text].filter(Boolean).join(" ").trim(),
      };
    case "correctedText":
      return { ...runtime, liveTranscript: event.text };
    case "sessionStopped":
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: runtime.serviceStatus === "error" ? "error" : "ready",
      };
    case "sessionError":
      return {
        ...runtime,
        activeSessionId: null,
        serviceStatus: "error",
        error: event.message,
      };
    case "partialText":
    case "commandDetected":
      return runtime;
  }
}
