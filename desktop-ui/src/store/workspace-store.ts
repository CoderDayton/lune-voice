import { Store } from "@tanstack/store";
import { backendClient } from "../lib/backend/client";
import type { BackendHistoryEntry, BackendWorkspaceSnapshot } from "../lib/backend/types";
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
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

interface WorkspaceState {
  sidebarCollapsed: boolean;
  historySearch: string;
  dictionaryEntries: DictionaryEntry[];
  historySessions: HistorySession[];
  settings: WorkspaceSettings;
  runtime: WorkspaceRuntimeState;
}

export function createInitialWorkspaceState(): WorkspaceState {
  return {
    sidebarCollapsed: false,
    historySearch: "",
    dictionaryEntries: [],
    historySessions: [],
    settings: {
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
    },
    runtime: {
      ipcFlavor: "unknown",
      serviceStatus: "Connecting",
      loaded: false,
      loading: false,
      error: null,
    },
  };
}

export const workspaceStore = new Store(createInitialWorkspaceState());

export const workspaceActions = {
  toggleSidebar() {
    workspaceStore.setState((state) => ({ ...state, sidebarCollapsed: !state.sidebarCollapsed }));
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
      dictionaryEntries: snapshot.dictionary.map((entry) => ({
        id: entry.id,
        phrase: entry.phrase,
        replacement: entry.replacement,
        priority: entry.priority,
      })),
      historySessions: snapshot.history
        .filter((entry) => entry.transcript)
        .map(mapHistoryEntry)
        .filter((entry): entry is HistorySession => entry !== null),
      settings: {
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
      },
      runtime: {
        ipcFlavor: snapshot.ipcFlavor,
        serviceStatus: snapshot.serviceStatus,
        loaded: true,
        loading: false,
        error: null,
      },
    }));
  },
};

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

  try {
    const snapshot = await backendClient.createDictionaryEntry(
      normalizedPhrase,
      normalizedReplacement,
      5,
    );
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
  if (value === "OrtQwenAsr") {
    return "Balanced";
  }

  if (value === "ParakeetTdt") {
    return "Fast and light";
  }

  return value;
}

export function summarizeDictationMode(value: DictationModeValue) {
  return labelForDeliveryMethod(value).replace(" into app", "");
}

export function summarizeLlmBackend(value: LlmBackendValue) {
  return labelForLlmBackend(value);
}

export function summarizeStylePreset(value: string) {
  return labelForStylePreset(value);
}

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

function countWords(text: string) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function formatThreshold(value: number) {
  return value.toFixed(2);
}

function formatError(error: unknown) {
  return error instanceof Error ? error.message : "Unable to reach the desktop backend.";
}
