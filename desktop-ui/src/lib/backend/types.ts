import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  LlmBackendValue,
} from "../../features/setup/data/settings-data";

export interface BackendDictionaryEntry {
  id: number;
  phrase: string;
  replacement: string;
  priority: number;
}

export interface BackendHistoryEntry {
  id: string;
  startedAt: string;
  endedAt: string | null;
  transcript: string | null;
  backend: string;
  style: string;
  status: string;
}

export interface BackendWorkspaceSettings {
  activationMode: ActivationModeValue;
  dictationMode: DictationModeValue;
  useGpu: boolean;
  asrBackendType: AsrBackendValue;
  llmBackendType: LlmBackendValue;
  confidenceThreshold: number | null;
  adaptiveVad: boolean;
  speechThreshold: number;
  stylePreset: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  alignerEnabled: boolean;
  pillVisibility: string;
  language: string;
  // OpenAI ASR fields (present when asrBackendType == "OpenAiCompatible")
  openaiAsrBaseUrl?: string | null;
  openaiAsrApiKey?: string | null;
  openaiAsrModel?: string | null;
  // OpenAI LLM fields (present when llmBackendType == "OpenAiCompatible")
  openaiLlmBaseUrl?: string | null;
  openaiLlmApiKey?: string | null;
  openaiLlmModel?: string | null;
}

export interface BackendWorkspaceSnapshot {
  ipcFlavor: string;
  serviceStatus: string;
  settings: BackendWorkspaceSettings;
  history: BackendHistoryEntry[];
  dictionary: BackendDictionaryEntry[];
}

export interface SaveWorkspaceSettingsInput {
  activationMode: ActivationModeValue;
  dictationMode: DictationModeValue;
  useGpu: boolean;
  asrBackendType: AsrBackendValue;
  llmBackendType: LlmBackendValue;
  llmCorrection: boolean;
  confidenceThreshold: number | null;
  adaptiveVad: boolean;
  speechThreshold: number;
  stylePreset: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  alignerEnabled: boolean;
  pillVisibility: string;
  language: string;
  // OpenAI ASR fields (sent when asrBackendType == "OpenAiCompatible")
  openaiAsrBaseUrl?: string | null;
  openaiAsrApiKey?: string | null;
  openaiAsrModel?: string | null;
  // OpenAI LLM fields (sent when llmBackendType == "OpenAiCompatible")
  openaiLlmBaseUrl?: string | null;
  openaiLlmApiKey?: string | null;
  openaiLlmModel?: string | null;
}

export interface BackendClient {
  loadWorkspaceSnapshot(): Promise<BackendWorkspaceSnapshot>;
  saveWorkspaceSettings(input: SaveWorkspaceSettingsInput): Promise<BackendWorkspaceSnapshot>;
  createDictionaryEntry(
    phrase: string,
    replacement: string,
    priority: number,
  ): Promise<BackendWorkspaceSnapshot>;
  deleteDictionaryEntry(id: number): Promise<BackendWorkspaceSnapshot>;
  startDictation(): Promise<string>;
  stopDictation(): Promise<void>;
  cancelDictation(): Promise<void>;
  toggleDictation(): Promise<string | null>;
}

export type SessionStreamEvent =
  | { type: "serviceStatus"; status: string }
  | { type: "sessionStarted"; sessionId: string }
  | { type: "partialText" }
  | { type: "processing" }
  | { type: "finalizedText"; text: string }
  | { type: "correctedText"; text: string }
  | { type: "audioLevel"; level: number }
  | { type: "commandDetected"; payload: string }
  | { type: "sessionStopped" }
  | { type: "sessionError"; message: string };
