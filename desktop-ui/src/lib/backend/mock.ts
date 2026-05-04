
import type { DictionaryEntry } from "../../features/dictionary/data/dictionary-data";
import type { BackendClient, BackendPillSnapshot, BackendWorkspaceSnapshot, SaveWorkspaceSettingsInput } from "./types";

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
      asrBackendType: "Qwen3Asr",
      confidenceThreshold: 0.5,
      adaptiveVad: false,
      speechThreshold: 0.5,
      stylePreset: "casual",
      hotkeyToggle: "Ctrl+Space",
      hotkeyCancel: "Escape",
      pillVisibility: "Always",
      language: "en",
      openaiAsrBaseUrl: null,
      openaiAsrModel: null,
      openaiAsrApiKeySet: false,
    },
    history: [
      {
        id: "s1",
        startedAt: new Date(Date.now() - 60_000).toISOString(),
        endedAt: new Date(Date.now() - 38_000).toISOString(),
        transcript: "Uh yeah it is working right now but obviously this isn't a very good approach. In particular, the way the data flows through the system needs significant improvement.",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
      {
        id: "s2",
        startedAt: new Date(Date.now() - 300_000).toISOString(),
        endedAt: new Date(Date.now() - 298_000).toISOString(),
        transcript: "Will this work now?",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
      {
        id: "s3",
        startedAt: new Date(Date.now() - 600_000).toISOString(),
        endedAt: new Date(Date.now() - 598_000).toISOString(),
        transcript: "Oh that was exactly right.",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
      {
        id: "s4",
        startedAt: new Date(Date.now() - 900_000).toISOString(),
        endedAt: new Date(Date.now() - 894_000).toISOString(),
        transcript: "This really needs some quality control. It doesn't understand my voice very well.",
        backend: "Qwen3Asr",
        style: "casual",
        status: "completed",
      },
    ],
    dictionary: initialDictionaryEntries.map((entry, index) => ({
      id: index + 1,
      phrase: entry.phrase,
      replacement: entry.replacement,
      priority: entry.priority,
    })),
    stylePresets: [
      { id: "minimal", name: "Minimal", description: "Fix errors only. No style changes." },
      { id: "casual", name: "Casual", description: "Relaxed, conversational tone." },
      { id: "professional", name: "Professional", description: "Polished and clear." },
      { id: "formal", name: "Formal", description: "Official correspondence." },
      { id: "technical", name: "Technical", description: "Code and engineering." },
      { id: "writing", name: "Writing", description: "Long-form content." },
    ],
  };
}

let mockSnapshot: BackendWorkspaceSnapshot = createMockSnapshot();
let nextDictionaryId = 1000;

function applySettings(input: SaveWorkspaceSettingsInput) {
  const { openaiAsrApiKey: _openaiAsrApiKey, ...fields } = input;
  mockSnapshot = {
    ...mockSnapshot,
    settings: {
      ...mockSnapshot.settings,
      ...fields,
      devicePolicy: fields.useGpu ? fields.devicePolicy : "Cpu",
      openaiAsrApiKeySet: !!_openaiAsrApiKey,
    },
  };
}

export const mockBackendClient: BackendClient = {
  async loadWorkspaceSnapshot() {
    return structuredClone(mockSnapshot);
  },

  async loadPillSnapshot(): Promise<BackendPillSnapshot> {
    return {
      serviceStatus: mockSnapshot.serviceStatus,
      activationMode: mockSnapshot.settings.activationMode,
      pillVisibility: mockSnapshot.settings.pillVisibility,
    };
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

  async updateDictionaryEntry(id, phrase, replacement, priority) {
    mockSnapshot = {
      ...mockSnapshot,
      dictionary: mockSnapshot.dictionary.map((entry) =>
        entry.id === id ? { ...entry, phrase, replacement, priority } : entry,
      ),
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
