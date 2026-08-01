import type { DictionaryEntry } from '../features/dictionary/data/dictionary-data';
import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  LlmBackendValue,
  PillAnchorValue,
  PillCornerValue,
  PillOrientationValue,
  PillSizeValue,
  PillVisibilityValue,
  TerminalModeValue,
  WhisperModelValue,
} from '../features/settings/data/settings-data';
import type { FormatTogglesDto, SnippetItemDto } from '../lib/backend/types';

export interface WorkspaceSettings {
  activationMode: ActivationModeValue;
  asrBackend: AsrBackendValue;
  whisperModel: WhisperModelValue;
  useGpu: boolean;
  devicePolicy: 'Cpu' | 'Cuda' | 'Auto';
  confidenceThreshold: string;
  llmSkipThreshold: string;
  dictationMode: DictationModeValue;
  copyToClipboard: boolean;
  soundCuesEnabled: boolean;
  soundCueVolume: number;
  /** Selected style preset id, or `null` for no styling (raw transcript). */
  stylePreset: string | null;
  hotkeyToggle: string;
  hotkeyCancel: string;
  pillVisibility: PillVisibilityValue;
  pillAnchor: PillAnchorValue;
  pillLocked: boolean;
  pillEdgeMargin: number;
  /** Monitor connector name the pill is pinned to; null = automatic. */
  pillMonitor: string | null;
  pillAllWorkspaces: boolean;
  pillSize: PillSizeValue;
  pillOrientation: PillOrientationValue;
  pillCorner: PillCornerValue;
  language: string;
  dictationLanguage: string;
  micPrewarm: boolean;
  /** Preferred microphone device name; null = system default. */
  inputDevice: string | null;
  vadPcen: boolean;
  startMinimized: boolean;
  /** Contact the release endpoint for a newer version at startup. */
  checkUpdatesOnLaunch: boolean;
  sessionHistoryDays: number | null;
  terminalMode: TerminalModeValue;
  llmEnabled: boolean;
  openaiAsrBaseUrl: string;
  openaiAsrApiKey: string;
  openaiAsrModel: string;
  // Styling LLM engine + remote provider config (editable via the Styling
  // engine settings section). OpenAI-compatible is the only engine; the API
  // key is masked ('••••••••') when a key is saved.
  llmBackend: LlmBackendValue;
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

export interface WorkspaceRuntimeState {
  ipcFlavor: string;
  serviceStatus: string;
  activeSessionId: string | null;
  liveTranscript: string;
  audioLevel: number;
  /** Active capture-only session (snippet trigger recorder), if any. */
  captureSessionId: string | null;
  /** Last raw transcript returned by a capture session ('' = heard nothing). */
  captureText: string | null;
  /** Bumps on every capture result so equal texts still notify consumers. */
  captureSeq: number;
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

export interface StylePreset {
  id: string;
  name: string;
  description: string;
  isBuiltin: boolean;
  instructions: string;
  formatting: FormatTogglesDto;
  fewShotDirty: boolean;
}

export interface SessionTotals {
  sessions: number;
  words: number;
  seconds: number;
}

export interface WorkspaceState {
  sidebarCollapsed: boolean;
  mobileMenuOpen: boolean;
  historySearch: string;
  historyPage: number;
  historyPageSize: number;
  dictionaryEntries: DictionaryEntry[];
  snippets: SnippetItemDto[];
  historySessions: HistorySession[];
  // Server-paged view for the History tab (fetched via `list_history_page`),
  // independent of the capped `historySessions` snapshot slice.
  historyPageItems: HistorySession[];
  historyTotal: number;
  historyLoading: boolean;
  sessionTotals: SessionTotals;
  stylePresets: StylePreset[];
  settings: WorkspaceSettings;
  runtime: WorkspaceRuntimeState;
}

export const DEFAULT_SETTINGS: WorkspaceSettings = {
  activationMode: 'Toggle',
  asrBackend: 'WhisperCpp',
  whisperModel: 'Medium',
  useGpu: true,
  devicePolicy: 'Auto',
  confidenceThreshold: '0.55',
  llmSkipThreshold: '0.40',
  dictationMode: 'Clipboard',
  copyToClipboard: false,
  soundCuesEnabled: true,
  soundCueVolume: 1.0,
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
  llmEnabled: true,
  openaiAsrBaseUrl: 'https://api.openai.com/v1',
  openaiAsrApiKey: '',
  openaiAsrModel: 'whisper-1',
  llmBackend: 'OpenAiCompatible',
  openaiLlmBaseUrl: 'https://api.openai.com/v1',
  openaiLlmApiKey: '',
  openaiLlmModel: 'gpt-4o-mini',
};

export const DEFAULT_RUNTIME: WorkspaceRuntimeState = {
  ipcFlavor: 'unknown',
  serviceStatus: 'Connecting',
  activeSessionId: null,
  liveTranscript: '',
  audioLevel: 0,
  captureSessionId: null,
  captureText: null,
  captureSeq: 0,
  loaded: false,
  loading: false,
  error: null,
};
