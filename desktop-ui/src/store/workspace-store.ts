import { Store } from "@tanstack/store";
import type { DictionaryEntry } from "../features/dictionary/data/dictionary-data";
import { initialDictionaryEntries } from "../features/dictionary/data/dictionary-data";

export interface WorkspaceSettings {
  activationMode: string;
  asrBackend: string;
  useGpu: boolean;
  confidenceThreshold: string;
  llmCorrection: boolean;
  llmModel: string;
  deliveryMethod: string;
  stylePreset: string;
  adaptiveVad: boolean;
  speechThreshold: string;
}

interface WorkspaceState {
  sidebarCollapsed: boolean;
  historySearch: string;
  dictionaryEntries: DictionaryEntry[];
  settings: WorkspaceSettings;
}

export function createInitialWorkspaceState(): WorkspaceState {
  return {
    sidebarCollapsed: false,
    historySearch: "",
    dictionaryEntries: initialDictionaryEntries.map((entry) => ({ ...entry })),
    settings: {
      activationMode: "Toggle",
      asrBackend: "Balanced (recommended)",
      useGpu: true,
      confidenceThreshold: "0.50",
      llmCorrection: true,
      llmModel: "Smart cleanup (local)",
      deliveryMethod: "Paste into app",
      stylePreset: "Everyday",
      adaptiveVad: false,
      speechThreshold: "0.50",
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

  addDictionaryEntry(phrase: string, replacement: string) {
    const normalizedPhrase = phrase.trim();
    const normalizedReplacement = replacement.trim();

    if (!normalizedPhrase || !normalizedReplacement) {
      return false;
    }

    workspaceStore.setState((state) => ({
      ...state,
      dictionaryEntries: [
        ...state.dictionaryEntries,
        {
          id: createDictionaryEntryId(normalizedPhrase),
          phrase: normalizedPhrase,
          replacement: normalizedReplacement,
          priority: 5,
        },
      ],
    }));

    return true;
  },

  removeDictionaryEntry(id: string) {
    workspaceStore.setState((state) => ({
      ...state,
      dictionaryEntries: state.dictionaryEntries.filter((entry) => entry.id !== id),
    }));
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
};

function createDictionaryEntryId(phrase: string) {
  const slug = phrase
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

  return `${slug || "entry"}-${crypto.randomUUID()}`;
}
