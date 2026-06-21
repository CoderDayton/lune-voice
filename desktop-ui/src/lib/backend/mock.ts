import type { DictionaryEntry } from '../../features/dictionary/data/dictionary-data';
import type {
  BackendClient,
  CalibrationHandle,
  CalibrationStreamEvent,
  PillSnapshotDto,
  SaveWorkspaceSettingsInput,
  WorkspaceSnapshot,
} from './types';

const initialDictionaryEntries: DictionaryEntry[] = [
  { id: 1, phrase: 'github', replacement: 'GitHub', priority: 5 },
  { id: 2, phrase: 'kubernetes', replacement: 'Kubernetes', priority: 8 },
  { id: 3, phrase: 'lune voice', replacement: 'Lune Voice', priority: 10 },
  { id: 4, phrase: 'rust lang', replacement: 'Rust', priority: 5 },
  { id: 5, phrase: 'api', replacement: 'API', priority: 6 },
  { id: 6, phrase: 'llm', replacement: 'LLM', priority: 7 },
];

function createMockSnapshot(): WorkspaceSnapshot {
  return {
    ipcFlavor: 'mock',
    serviceStatus: 'ready',
    settings: {
      activationMode: 'Toggle',
      dictationMode: 'Clipboard',
      useGpu: true,
      devicePolicy: 'Auto',
      asrBackendType: 'Qwen3Asr',
      llmEnabled: true,
      confidenceThreshold: 0.55,
      llmSkipThreshold: 0.4,
      adaptiveVad: false,
      speechThreshold: 0.5,
      speechThresholdEnd: 0.3,
      stylePreset: 'casual',
      hotkeyToggle: 'Ctrl+Space',
      hotkeyCancel: 'Escape',
      pillVisibility: 'Always',
      language: 'en',
      dictationLanguage: 'auto',
      micPrewarm: true,
      inputDevice: null,
      vadPcen: true,
      startMinimized: false,
      sessionHistoryDays: 30,
      terminalMode: 'Auto',
      openaiAsrBaseUrl: null,
      openaiAsrModel: null,
      openaiAsrApiKeySet: false,
    },
    history: [
      {
        id: 's1',
        startedAt: new Date(Date.now() - 60_000).toISOString(),
        endedAt: new Date(Date.now() - 38_000).toISOString(),
        transcript:
          "Uh yeah it is working right now but obviously this isn't a very good approach. In particular, the way the data flows through the system needs significant improvement.",
        backend: 'Qwen3Asr',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's2',
        startedAt: new Date(Date.now() - 300_000).toISOString(),
        endedAt: new Date(Date.now() - 298_000).toISOString(),
        transcript: 'Will this work now?',
        backend: 'Qwen3Asr',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's3',
        startedAt: new Date(Date.now() - 600_000).toISOString(),
        endedAt: new Date(Date.now() - 598_000).toISOString(),
        transcript: 'Oh that was exactly right.',
        backend: 'Qwen3Asr',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's4',
        startedAt: new Date(Date.now() - 900_000).toISOString(),
        endedAt: new Date(Date.now() - 894_000).toISOString(),
        transcript:
          "This really needs some quality control. It doesn't understand my voice very well.",
        backend: 'Qwen3Asr',
        style: 'casual',
        status: 'completed',
      },
    ],
    dictionary: initialDictionaryEntries.map((entry, index) => ({
      id: index + 1,
      phrase: entry.phrase,
      replacement: entry.replacement,
      priority: entry.priority,
    })),
    stats: { sessions: 0, words: 0, seconds: 0 },
    stylePresets: [
      {
        id: 'minimal',
        name: 'Minimal',
        description: 'Error fixes only, no AI styling.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
      {
        id: 'casual',
        name: 'Casual',
        description: 'Relaxed, conversational tone.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
      {
        id: 'professional',
        name: 'Professional',
        description: 'Polished and clear.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
      {
        id: 'formal',
        name: 'Formal',
        description: 'Official correspondence.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
      {
        id: 'technical',
        name: 'Technical',
        description: 'Code and engineering.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
      {
        id: 'writing',
        name: 'Writing',
        description: 'Long-form content.',
        isBuiltin: true,
        instructions: '',
        formatting: {
          keybinds: true,
          titleAbbreviations: true,
          autoCapitalization: true,
          brandCasing: true,
        },
      },
    ],
  };
}

function computeStats(history: WorkspaceSnapshot['history']) {
  let words = 0;
  let seconds = 0;
  let sessions = 0;
  for (const h of history) {
    if (!h.transcript) continue;
    const w = h.transcript.trim().split(/\s+/).filter(Boolean).length;
    if (w === 0) continue;
    sessions++;
    words += w;
    if (h.endedAt) {
      const dt = (new Date(h.endedAt).getTime() - new Date(h.startedAt).getTime()) / 1000;
      seconds += Math.max(1, Math.round(dt));
    }
  }
  return { sessions, words, seconds };
}

function snapshotWithStats(s: WorkspaceSnapshot): WorkspaceSnapshot {
  return { ...s, stats: computeStats(s.history) };
}

let mockSnapshot: WorkspaceSnapshot = snapshotWithStats(createMockSnapshot());
let nextDictionaryId = 1000;
let nextStyleId = 1000;

function applySettings(input: SaveWorkspaceSettingsInput) {
  const { openaiAsrApiKey: _openaiAsrApiKey, ...fields } = input;
  mockSnapshot = {
    ...mockSnapshot,
    settings: {
      ...mockSnapshot.settings,
      ...fields,
      devicePolicy: fields.useGpu ? fields.devicePolicy : 'Cpu',
      openaiAsrApiKeySet: !!_openaiAsrApiKey,
    },
  };
}

export const mockBackendClient: BackendClient = {
  async loadWorkspaceSnapshot() {
    return structuredClone(mockSnapshot);
  },

  async loadPillSnapshot(): Promise<PillSnapshotDto> {
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

  async deleteSession(sessionId) {
    mockSnapshot = snapshotWithStats({
      ...mockSnapshot,
      history: mockSnapshot.history.filter((entry) => entry.id !== sessionId),
    });
    return structuredClone(mockSnapshot);
  },

  async createCustomStyle(name, description, instructions, formatting) {
    mockSnapshot = {
      ...mockSnapshot,
      stylePresets: [
        ...mockSnapshot.stylePresets,
        {
          id: `custom-${nextStyleId++}`,
          name,
          description,
          instructions,
          isBuiltin: false,
          formatting,
        },
      ],
    };
    return structuredClone(mockSnapshot);
  },
  async updateCustomStyle(id, name, description, instructions, formatting) {
    mockSnapshot = {
      ...mockSnapshot,
      stylePresets: mockSnapshot.stylePresets.map((p) =>
        p.id === id ? { ...p, name, description, instructions, formatting } : p,
      ),
    };
    return structuredClone(mockSnapshot);
  },
  async deleteCustomStyle(styleId) {
    const wasActive = mockSnapshot.settings.stylePreset === styleId;
    mockSnapshot = {
      ...mockSnapshot,
      stylePresets: mockSnapshot.stylePresets.filter((p) => p.id !== styleId),
      settings: wasActive
        ? { ...mockSnapshot.settings, stylePreset: 'casual' }
        : mockSnapshot.settings,
    };
    return structuredClone(mockSnapshot);
  },
  async generateStyleExamples(_styleId) {
    return structuredClone(mockSnapshot);
  },

  async startDictation() {
    return 'mock-session';
  },

  async stopDictation() {},

  async cancelDictation() {},

  async toggleDictation() {
    return 'mock-session';
  },

  async calibrateStart(onEvent): Promise<CalibrationHandle> {
    let cancelled = false;
    let resolveAdvance: (() => void) | null = null;
    const advancePromise = new Promise<void>((resolve) => {
      resolveAdvance = resolve;
    });

    const emit = (e: CalibrationStreamEvent) => {
      if (!cancelled) onEvent(e);
    };

    const tick = (phase: 'noise' | 'speech', durationMs: number, level: number) =>
      new Promise<void>((resolve) => {
        const start = performance.now();
        const id = setInterval(() => {
          if (cancelled) {
            clearInterval(id);
            resolve();
            return;
          }
          const elapsed = performance.now() - start;
          const jitter = (Math.random() - 0.5) * 0.1;
          emit({
            type: 'frame',
            phase,
            vadScore: Math.max(0, Math.min(1, level + jitter)),
            energy: Math.max(0, Math.min(1, level * 0.4 + jitter * 0.2)),
          });
          if (elapsed >= durationMs) {
            clearInterval(id);
            resolve();
          }
        }, 50);
      });

    (async () => {
      emit({ type: 'phaseChanged', phase: 'noise' });
      await tick('noise', 3000, 0.05);
      if (cancelled) return emit({ type: 'phaseChanged', phase: 'cancelled' });
      emit({ type: 'phaseChanged', phase: 'waiting' });
      await advancePromise;
      if (cancelled) return emit({ type: 'phaseChanged', phase: 'cancelled' });
      emit({ type: 'phaseChanged', phase: 'speech' });
      await tick('speech', 8000, 0.85);
      if (cancelled) return emit({ type: 'phaseChanged', phase: 'cancelled' });
      emit({ type: 'phaseChanged', phase: 'done' });
      emit({
        type: 'result',
        speechThreshold: 0.55,
        speechThresholdEnd: 0.2,
        diagnostics: { noiseP95: 0.07, speechP10: 0.78, pauseP20: 0.18 },
      });
    })();

    return {
      async advance() {
        resolveAdvance?.();
      },
      async cancel() {
        cancelled = true;
        resolveAdvance?.(); // unblock the waiting phase if still pending
      },
    };
  },
};
