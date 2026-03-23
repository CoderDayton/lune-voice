import type { DictionaryEntry } from "../features/dictionary/data/dictionary-data";
import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  LlmBackendValue,
  PillVisibilityValue,
} from "../features/settings/data/settings-data";

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
  alignerEnabled: boolean;
  pillVisibility: PillVisibilityValue;
  language: string;
  openaiAsrBaseUrl: string;
  openaiAsrApiKey: string;
  openaiAsrModel: string;
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
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

export interface WorkspaceState {
  sidebarCollapsed: boolean;
  mobileMenuOpen: boolean;
  historySearch: string;
  dictionaryEntries: DictionaryEntry[];
  historySessions: HistorySession[];
  settings: WorkspaceSettings;
  runtime: WorkspaceRuntimeState;
}

export const DEFAULT_SETTINGS: WorkspaceSettings = {
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
  alignerEnabled: true,
  pillVisibility: "Always",
  language: "en",
  openaiAsrBaseUrl: "https://api.openai.com/v1",
  openaiAsrApiKey: "",
  openaiAsrModel: "whisper-1",
  openaiLlmBaseUrl: "https://api.openai.com/v1",
  openaiLlmApiKey: "",
  openaiLlmModel: "gpt-4o-mini",
};

export const DEFAULT_RUNTIME: WorkspaceRuntimeState = {
  ipcFlavor: "unknown",
  serviceStatus: "Connecting",
  activeSessionId: null,
  liveTranscript: "",
  loaded: false,
  loading: false,
  error: null,
};
