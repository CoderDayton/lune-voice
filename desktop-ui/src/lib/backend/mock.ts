
import type { DictionaryEntry } from "../../features/dictionary/data/dictionary-data";
import type { BackendClient, BackendWorkspaceSnapshot, SaveWorkspaceSettingsInput } from "./types";

const initialDictionaryEntries: DictionaryEntry[] = [
  { id: 1, phrase: "github", replacement: "GitHub", priority: 5 },
  { id: 2, phrase: "kubernetes", replacement: "Kubernetes", priority: 8 },
  { id: 3, phrase: "lune voice", replacement: "Lune Voice", priority: 10 },
  { id: 4, phrase: "rust lang", replacement: "Rust", priority: 5 },
  { id: 5, phrase: "api", replacement: "API", priority: 6 },
  { id: 6, phrase: "llm", replacement: "LLM", priority: 7 },
];

function createMockSnapshot(): BackendWorkspaceSnapshot {
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
  const { llmCorrection, openaiAsrApiKey: _openaiAsrApiKey, openaiLlmApiKey: _openaiLlmApiKey, ...fields } = input;
  mockSnapshot = {
    ...mockSnapshot,
    settings: {
      ...mockSnapshot.settings,
      ...fields,
      devicePolicy: fields.useGpu ? fields.devicePolicy : "Cpu",
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
