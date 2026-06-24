import type { DictionaryEntry } from '../features/dictionary/data/dictionary-data';
import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  PillVisibilityValue,
  TerminalModeValue,
} from '../features/settings/data/settings-data';
import type { FormatTogglesDto } from '../lib/backend/types';

export interface WorkspaceSettings {
  activationMode: ActivationModeValue;
  asrBackend: AsrBackendValue;
  useGpu: boolean;
  devicePolicy: 'Cpu' | 'Cuda' | 'Auto';
  confidenceThreshold: string;
  llmSkipThreshold: string;
  dictationMode: DictationModeValue;
  copyToClipboard: boolean;
  soundCuesEnabled: boolean;
  soundCueVolume: number;
  stylePreset: string;
  adaptiveVad: boolean;
  speechThreshold: string;
  speechThresholdEnd: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  pillVisibility: PillVisibilityValue;
  language: string;
  dictationLanguage: string;
  micPrewarm: boolean;
  /** Preferred microphone device name; null = system default. */
  inputDevice: string | null;
  vadPcen: boolean;
  startMinimized: boolean;
  sessionHistoryDays: number | null;
  terminalMode: TerminalModeValue;
  llmEnabled: boolean;
  openaiAsrBaseUrl: string;
  openaiAsrApiKey: string;
  openaiAsrModel: string;
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
  historySessions: HistorySession[];
  sessionTotals: SessionTotals;
  stylePresets: StylePreset[];
  settings: WorkspaceSettings;
  runtime: WorkspaceRuntimeState;
}

export const DEFAULT_SETTINGS: WorkspaceSettings = {
  activationMode: 'Toggle',
  asrBackend: 'Qwen3Asr',
  useGpu: true,
  devicePolicy: 'Auto',
  confidenceThreshold: '0.55',
  llmSkipThreshold: '0.40',
  dictationMode: 'Clipboard',
  copyToClipboard: false,
  soundCuesEnabled: true,
  soundCueVolume: 1.0,
  stylePreset: 'casual',
  adaptiveVad: false,
  speechThreshold: '0.50',
  speechThresholdEnd: '0.30',
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
  llmEnabled: true,
  openaiAsrBaseUrl: 'https://api.openai.com/v1',
  openaiAsrApiKey: '',
  openaiAsrModel: 'whisper-1',
};

export const DEFAULT_RUNTIME: WorkspaceRuntimeState = {
  ipcFlavor: 'unknown',
  serviceStatus: 'Connecting',
  activeSessionId: null,
  liveTranscript: '',
  audioLevel: 0,
  loaded: false,
  loading: false,
  error: null,
};
