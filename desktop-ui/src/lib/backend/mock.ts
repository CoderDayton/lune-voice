
import { initialDictionaryEntries } from "../../features/dictionary/data/dictionary-data";
import type { BackendClient, BackendWorkspaceSnapshot, SaveWorkspaceSettingsInput } from "./types";

function createMockSnapshot(): BackendWorkspaceSnapshot {
  return {
    ipcFlavor: "mock",
    serviceStatus: "ready",
    settings: {
      activationMode: "Toggle",
      dictationMode: "Clipboard",
      useGpu: true,
      asrBackendType: "OrtQwenAsr",
      llmBackendType: "LlamaCppLocal",
      confidenceThreshold: 0.5,
      adaptiveVad: false,
      speechThreshold: 0.5,
      stylePreset: "casual",
      hotkeyToggle: "Ctrl+Space",
      hotkeyCancel: "Escape",
      openaiAsrBaseUrl: null,
      openaiAsrApiKey: null,
      openaiAsrModel: null,
      openaiLlmBaseUrl: null,
      openaiLlmApiKey: null,
      openaiLlmModel: null,
      alignerEnabled: true,
      pillVisibility: "Always",
      language: "en",
    },
    history: [],
    dictionary: initialDictionaryEntries.map((entry, index) => ({
      id: index + 1,
      phrase: entry.phrase,
      replacement: entry.replacement,
      priority: entry.priority,
    })),
  };
}

let mockSnapshot: BackendWorkspaceSnapshot = createMockSnapshot();
let nextDictionaryId = 1000;

function applySettings(input: SaveWorkspaceSettingsInput) {
  const { llmCorrection, ...fields } = input;
  mockSnapshot = {
    ...mockSnapshot,
    settings: {
      ...mockSnapshot.settings,
      ...fields,
      confidenceThreshold: llmCorrection ? input.confidenceThreshold : null,
    },
  };
}

export const mockBackendClient: BackendClient = {
  async loadWorkspaceSnapshot() {
    return structuredClone(mockSnapshot);
  },

  async saveWorkspaceSettings(input) {
    applySettings(input);
    return structuredClone(mockSnapshot);
  },

  async createDictionaryEntry(phrase, replacement, priority) {
    mockSnapshot = {
      ...mockSnapshot,
      dictionary: [
        ...mockSnapshot.dictionary,
        {
          id: nextDictionaryId++,
          phrase,
          replacement,
          priority,
        },
      ],
    };

    return structuredClone(mockSnapshot);
  },

  async deleteDictionaryEntry(id) {
    mockSnapshot = {
      ...mockSnapshot,
      dictionary: mockSnapshot.dictionary.filter((entry) => entry.id !== id),
    };

    return structuredClone(mockSnapshot);
  },

  async startDictation() {
    return "mock-session";
  },

  async stopDictation() {},

  async cancelDictation() {},

  async toggleDictation() {
    return "mock-session";
  },
};

export function resetMockBackendClient() {
  mockSnapshot = createMockSnapshot();
  nextDictionaryId = 1000;
}
