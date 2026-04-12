import type { DictionaryEntry } from "../features/dictionary/data/dictionary-data";
import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  PillVisibilityValue,
} from "../features/settings/data/settings-data";

export interface WorkspaceSettings {
  activationMode: ActivationModeValue;
  asrBackend: AsrBackendValue;
  useGpu: boolean;
  devicePolicy: "Cpu" | "Cuda" | "Auto";
  confidenceThreshold: string;
  dictationMode: DictationModeValue;
  stylePreset: string;
  adaptiveVad: boolean;
  speechThreshold: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  pillVisibility: PillVisibilityValue;
  language: string;
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
  asrBackend: "ParakeetTdt",
  useGpu: true,
  devicePolicy: "Auto",
  confidenceThreshold: "0.50",
  dictationMode: "Clipboard",
  stylePreset: "casual",
  adaptiveVad: false,
  speechThreshold: "0.50",
  hotkeyToggle: "Ctrl+Space",
  hotkeyCancel: "Escape",
  pillVisibility: "Always",
  language: "en",
  openaiAsrBaseUrl: "https://api.openai.com/v1",
  openaiAsrApiKey: "",
  openaiAsrModel: "whisper-1",
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
