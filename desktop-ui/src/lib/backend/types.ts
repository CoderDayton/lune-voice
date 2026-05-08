import type {
  ActivationModeValue,
  AsrBackendValue,
  DictationModeValue,
  PillVisibilityValue,
} from '../../features/settings/data/settings-data';

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
  devicePolicy: 'Cpu' | 'Cuda' | 'Auto';
  asrBackendType: AsrBackendValue;
  confidenceThreshold: number | null;
  adaptiveVad: boolean;
  speechThreshold: number;
  speechThresholdEnd: number;
  stylePreset: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  pillVisibility: PillVisibilityValue;
  language: string;
  dictationLanguage: string;
  micPrewarm: boolean;
  vadPcen: boolean;
  // OpenAI ASR fields (present when asrBackendType == "OpenAiCompatible")
  openaiAsrBaseUrl?: string | null;
  openaiAsrModel?: string | null;
  openaiAsrApiKeySet: boolean;
}

export interface BackendStylePreset {
  id: string;
  name: string;
  description: string;
  isBuiltin: boolean;
  instructions: string;
}

export interface BackendWorkspaceSnapshot {
  ipcFlavor: string;
  serviceStatus: string;
  settings: BackendWorkspaceSettings;
  history: BackendHistoryEntry[];
  dictionary: BackendDictionaryEntry[];
  stylePresets: BackendStylePreset[];
}

export interface BackendPillSnapshot {
  serviceStatus: string;
  activationMode: ActivationModeValue;
  pillVisibility: PillVisibilityValue;
}

export interface SaveWorkspaceSettingsInput {
  activationMode: ActivationModeValue;
  dictationMode: DictationModeValue;
  useGpu: boolean;
  devicePolicy: 'Cpu' | 'Cuda' | 'Auto';
  asrBackendType: AsrBackendValue;
  confidenceThreshold: number | null;
  adaptiveVad: boolean;
  speechThreshold: number;
  speechThresholdEnd: number;
  stylePreset: string;
  hotkeyToggle: string;
  hotkeyCancel: string;
  pillVisibility: PillVisibilityValue;
  language: string;
  dictationLanguage: string;
  micPrewarm: boolean;
  vadPcen: boolean;
  // OpenAI ASR fields (sent when asrBackendType == "OpenAiCompatible")
  openaiAsrBaseUrl?: string | null;
  openaiAsrApiKey?: string | null;
  openaiAsrModel?: string | null;
}

export type CalibrationPhase =
  | 'idle'
  | 'noise'
  | 'waiting'
  | 'speech'
  | 'done'
  | 'cancelled'
  | 'failed';

export interface CalibrationDiagnostics {
  noiseP95: number;
  speechP10: number;
  pauseP20: number;
}

export type CalibrationStreamEvent =
  | { type: 'phaseChanged'; phase: CalibrationPhase }
  | { type: 'frame'; phase: CalibrationPhase; vadScore: number; energy: number }
  | {
      type: 'result';
      speechThreshold: number;
      speechThresholdEnd: number;
      diagnostics: CalibrationDiagnostics;
    }
  | { type: 'error'; message: string };

export interface CalibrationHandle {
  advance(): Promise<void>;
  cancel(): Promise<void>;
}

export interface BackendClient {
  loadWorkspaceSnapshot(): Promise<BackendWorkspaceSnapshot>;
  loadPillSnapshot(): Promise<BackendPillSnapshot>;
  saveWorkspaceSettings(input: SaveWorkspaceSettingsInput): Promise<BackendWorkspaceSnapshot>;
  createDictionaryEntry(
    phrase: string,
    replacement: string,
    priority: number,
  ): Promise<BackendWorkspaceSnapshot>;
  updateDictionaryEntry(
    id: number,
    phrase: string,
    replacement: string,
    priority: number,
  ): Promise<BackendWorkspaceSnapshot>;
  deleteDictionaryEntry(id: number): Promise<BackendWorkspaceSnapshot>;
  deleteSession(sessionId: string): Promise<BackendWorkspaceSnapshot>;
  createCustomStyle(
    name: string,
    description: string,
    instructions: string,
  ): Promise<BackendWorkspaceSnapshot>;
  updateCustomStyle(
    id: string,
    name: string,
    description: string,
    instructions: string,
  ): Promise<BackendWorkspaceSnapshot>;
  deleteCustomStyle(styleId: string): Promise<BackendWorkspaceSnapshot>;
  generateStyleExamples(styleId: string): Promise<BackendWorkspaceSnapshot>;
  startDictation(): Promise<string>;
  stopDictation(): Promise<void>;
  cancelDictation(): Promise<void>;
  toggleDictation(): Promise<string | null>;
  calibrateStart(onEvent: (event: CalibrationStreamEvent) => void): Promise<CalibrationHandle>;
}

export type SessionStreamEvent =
  | { type: 'serviceStatus'; status: string }
  | { type: 'sessionStarted'; sessionId: string }
  | { type: 'partialText' }
  | { type: 'processing' }
  | { type: 'finalizedText'; text: string }
  | { type: 'correctedText'; text: string }
  | { type: 'audioLevel'; level: number }
  | { type: 'commandDetected'; payload: string }
  | { type: 'sessionStopped' }
  | { type: 'sessionError'; message: string };
