import type { DictionaryEntry } from '../../features/dictionary/data/dictionary-data';
import type {
  BackendClient,
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
      copyToClipboard: false,
      soundCuesEnabled: true,
      soundCueVolume: 1.0,
      useGpu: true,
      devicePolicy: 'Auto',
      asrBackendType: 'WhisperCpp',
      whisperModel: 'Medium',
      llmEnabled: true,
      confidenceThreshold: 0.55,
      llmSkipThreshold: 0.4,
      stylePreset: null,
      hotkeyToggle: 'Ctrl+Space',
      hotkeyCancel: 'Escape',
      pillVisibility: 'Always',
      pillAnchor: 'BottomCenter',
      pillLocked: false,
      pillEdgeMargin: 72,
      pillMonitor: null,
      pillAllWorkspaces: true,
      pillSize: 'Standard',
      pillOrientation: 'Horizontal',
      pillCorner: 'Rounded',
      language: 'en',
      dictationLanguage: 'auto',
      micPrewarm: true,
      inputDevice: null,
      vadPcen: true,
      startMinimized: false,
      checkUpdatesOnLaunch: false,
      sessionHistoryDays: 30,
      terminalMode: 'Auto',
      openaiAsrBaseUrl: null,
      openaiAsrModel: null,
      openaiAsrApiKeySet: false,
      openaiLlmBaseUrl: 'https://api.openai.com/v1',
      openaiLlmModel: 'gpt-4o-mini',
      openaiLlmApiKeySet: false,
      openaiLlmActive: false,
    },
    history: [
      {
        id: 's1',
        startedAt: new Date(Date.now() - 60_000).toISOString(),
        endedAt: new Date(Date.now() - 38_000).toISOString(),
        transcript:
          "Uh yeah it is working right now but obviously this isn't a very good approach. In particular, the way the data flows through the system needs significant improvement.",
        backend: 'WhisperCpp',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's2',
        startedAt: new Date(Date.now() - 300_000).toISOString(),
        endedAt: new Date(Date.now() - 298_000).toISOString(),
        transcript: 'Will this work now?',
        backend: 'WhisperCpp',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's3',
        startedAt: new Date(Date.now() - 600_000).toISOString(),
        endedAt: new Date(Date.now() - 598_000).toISOString(),
        transcript: 'Oh that was exactly right.',
        backend: 'WhisperCpp',
        style: 'casual',
        status: 'completed',
      },
      {
        id: 's4',
        startedAt: new Date(Date.now() - 900_000).toISOString(),
        endedAt: new Date(Date.now() - 894_000).toISOString(),
        transcript:
          "This really needs some quality control. It doesn't understand my voice very well.",
        backend: 'WhisperCpp',
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
    snippets: [
      { id: 'snip-1', triggerPhrase: 'signature', body: 'Best regards,\nDayton' },
      {
        id: 'snip-2',
        triggerPhrase: 'meeting header',
        body: '## Meeting — {date}\n\nAttendees:\n- ',
      },
    ],
    stats: { sessions: 0, words: 0, seconds: 0 },
    stylePresets: [
      {
        id: 'casual',
        name: 'Casual',
        description: 'Relaxed, conversational tone.',
        isBuiltin: true,
        instructions: '',
        fewShotDirty: false,
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
        fewShotDirty: false,
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
        fewShotDirty: false,
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
        fewShotDirty: false,
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
        fewShotDirty: false,
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
let nextSnippetId = 1000;

/// Reset the mock backend to its seeded initial state. `mockSnapshot` and the id
/// counters are module-level mutable state shared across every test file in a run,
/// so a test that creates/deletes dictionary or style entries otherwise leaks into
/// later files. Call this in a test's `beforeEach` for deterministic,
/// order-independent state.
export function resetMockBackend() {
  mockSnapshot = snapshotWithStats(createMockSnapshot());
  nextDictionaryId = 1000;
  nextStyleId = 1000;
  nextSnippetId = 1000;
}

function applySettings(input: SaveWorkspaceSettingsInput) {
  const {
    openaiAsrApiKey: _openaiAsrApiKey,
    openaiLlmApiKey: _openaiLlmApiKey,
    openaiLlmBaseUrl: _openaiLlmBaseUrl,
    openaiLlmModel: _openaiLlmModel,
    llmBackendType: _llmBackendType,
    ...fields
  } = input;
  // Mirror settings_to_dto: the remote LLM fields are populated whenever
  // styling is enabled — OpenAI-compatible is the only engine now.
  const remote = input.llmEnabled;
  mockSnapshot = {
    ...mockSnapshot,
    settings: {
      ...mockSnapshot.settings,
      ...fields,
      devicePolicy: fields.useGpu ? fields.devicePolicy : 'Cpu',
      whisperModel: fields.whisperModel ?? mockSnapshot.settings.whisperModel,
      openaiAsrApiKeySet: !!_openaiAsrApiKey,
      // Provider persists regardless of the active engine (blank keeps saved).
      openaiLlmBaseUrl:
        _openaiLlmBaseUrl && _openaiLlmBaseUrl.length > 0
          ? _openaiLlmBaseUrl
          : mockSnapshot.settings.openaiLlmBaseUrl,
      openaiLlmModel:
        _openaiLlmModel && _openaiLlmModel.length > 0
          ? _openaiLlmModel
          : mockSnapshot.settings.openaiLlmModel,
      openaiLlmApiKeySet:
        _openaiLlmApiKey && _openaiLlmApiKey.length > 0
          ? true
          : mockSnapshot.settings.openaiLlmApiKeySet,
      openaiLlmActive: remote,
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
      pillLocked: mockSnapshot.settings.pillLocked,
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

  async saveSnippet(id, triggerPhrase, body) {
    const trigger = triggerPhrase.trim();
    // Mirror the backend's case-insensitive trigger-collision rejection.
    const clash = mockSnapshot.snippets.some(
      (s) => s.id !== id && s.triggerPhrase.trim().toLowerCase() === trigger.toLowerCase(),
    );
    if (clash) throw new Error('another snippet already uses this trigger phrase');
    mockSnapshot = {
      ...mockSnapshot,
      snippets: id
        ? mockSnapshot.snippets.map((s) =>
            s.id === id ? { ...s, triggerPhrase: trigger, body } : s,
          )
        : [
            ...mockSnapshot.snippets,
            { id: `snip-${nextSnippetId++}`, triggerPhrase: trigger, body },
          ],
    };
    return structuredClone(mockSnapshot);
  },

  async deleteSnippet(id) {
    mockSnapshot = {
      ...mockSnapshot,
      snippets: mockSnapshot.snippets.filter((s) => s.id !== id),
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

  async listHistoryPage(page, pageSize, search) {
    const q = search.trim().toLowerCase();
    const matched = mockSnapshot.history
      .filter((entry) => entry.transcript)
      .filter((entry) => (q ? entry.transcript.toLowerCase().includes(q) : true))
      .sort((a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime());
    const start = Math.max(0, (page - 1) * pageSize);
    return {
      sessions: structuredClone(matched.slice(start, start + pageSize)),
      total: matched.length,
    };
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
          fewShotDirty: false,
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
      settings: wasActive ? { ...mockSnapshot.settings, stylePreset: null } : mockSnapshot.settings,
    };
    return structuredClone(mockSnapshot);
  },
  async generateStyleExamples(_styleId) {
    return structuredClone(mockSnapshot);
  },

  async startDictation() {
    return 'mock-session';
  },

  // The store simulates the capture event sequence in mock mode (it owns the
  // event pipeline); this only hands back a session id.
  async startTriggerCapture() {
    return 'mock-capture';
  },

  async stopDictation() {},

  async cancelDictation() {},

  async toggleDictation() {
    return 'mock-session';
  },

  async nudgePill() {},

  // Browser mock has no window to move — report no native tracking so the
  // drag hook stays on the (equally mocked) nudge path.
  async beginPillDrag() {
    return false;
  },

  async dragPillTick() {},

  async endPillDrag() {},

  async savePillPosition() {},

  async resetPillPosition() {},

  // One believable monitor so the picker renders in the browser mock.
  async listMonitors() {
    return [{ name: 'Mock-1', width: 1920, height: 1080, isPrimary: true }];
  },

  // Nothing to update in the browser mock — there is no packaged app to
  // replace, so the check is a no-op rather than a fake "up to date".
  async appVersion() {
    return '0.0.0-mock';
  },

  async startUpdateCheck() {},

  async gpuBackendStatus() {
    // The browser/mock runtime has no native backend at all, so the onboarding
    // GPU affordance and the Settings GPU section both stay hidden.
    return {
      state: 'unsupported' as const,
      devices: [],
      reason: null,
      downloadBytes: 0,
      downloadAvailable: false,
    };
  },
};
