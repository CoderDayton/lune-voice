// SPDX-License-Identifier: GPL-3.0-or-later

import type { DictionaryEntry } from '../features/dictionary/data/dictionary-data';
import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  LlmBackendValue,
  ParakeetModelValue,
  PillAnchorValue,
  PillCornerValue,
  PillOrientationValue,
  PillSizeValue,
  PillVisibilityValue,
  PrivacyModeValue,
  TerminalModeValue,
  WhisperModelValue,
} from '../features/settings/data/settings-data';
import type {
  FormatTogglesDto,
  PlatformCapabilitiesDto,
  SnippetItemDto,
} from '../lib/backend/types';

export interface WorkspaceSettings {
  activationMode: ActivationModeValue;
  asrBackend: AsrBackendValue;
  whisperModel: WhisperModelValue;
  parakeetModel: ParakeetModelValue;
  useGpu: boolean;
  devicePolicy: 'Cpu' | 'Cuda' | 'Auto';
  confidenceThreshold: string;
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
  /** Minutes idle before the ASR model is freed; null = keep it loaded. */
  modelIdleUnloadMinutes: number | null;
  /** Preferred microphone device name; null = system default. */
  inputDevice: string | null;
  vadPcen: boolean;
  startMinimized: boolean;
  /** Contact the release endpoint for a newer version at startup. */
  checkUpdatesOnLaunch: boolean;
  /** Capitalize known names ("github" → "GitHub"). A style can override it. */
  brandCasing: boolean;
  sessionHistoryDays: number | null;
  terminalMode: TerminalModeValue;
  privacyMode: PrivacyModeValue;
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
  /** What the host platform's window system supports. Controls for an
   *  unsupported capability are not rendered. */
  capabilities: PlatformCapabilitiesDto;
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
  // Today's share of the same totals. Aggregated by the backend over every
  // row: `historySessions` is a capped recent slice, so deriving these from it
  // silently under-reports a day with more dictations than the cap.
  sessionsToday: number;
  wordsToday: number;
  secondsToday: number;
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
  parakeetModel: 'TdtV3Q8',
  useGpu: true,
  devicePolicy: 'Auto',
  confidenceThreshold: '0.55',
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
  modelIdleUnloadMinutes: null,
  inputDevice: null,
  vadPcen: true,
  startMinimized: false,
  checkUpdatesOnLaunch: false,
  brandCasing: true,
  sessionHistoryDays: 30,
  terminalMode: 'Auto',
  privacyMode: 'LocalOnly',
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
  capabilities: { pillAllWorkspaces: true },
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
