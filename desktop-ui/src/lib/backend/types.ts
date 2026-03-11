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
}
