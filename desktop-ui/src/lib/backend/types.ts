// DTO shapes are generated from `desktop-ui/src-tauri/src/dto.rs` via ts-rs.
// Run `cargo test -p lune-ui` after editing any Rust DTO to regenerate.
//
// The historical `Backend*` aliases below are kept so downstream imports
// (workspace store, mappers, components, tests) don't need a sweeping rename.

import type {
  DictionaryItemDto,
  HistoryItemDto,
  PillSnapshotDto,
  SaveWorkspaceSettingsInput as SaveWorkspaceSettingsInputDto,
  StylePresetDto,
  WorkspaceSettingsDto,
  WorkspaceSnapshot,
} from './generated';

export type {
  ActivationModeDto,
  AsrBackendTypeDto,
  DevicePolicyDto,
  DictationModeDto,
  PillVisibilityDto,
} from './generated';

export type BackendDictionaryEntry = DictionaryItemDto;
export type BackendHistoryEntry = HistoryItemDto;
export type BackendPillSnapshot = PillSnapshotDto;
export type BackendStylePreset = StylePresetDto;
export type BackendWorkspaceSettings = WorkspaceSettingsDto;
export type BackendWorkspaceSnapshot = WorkspaceSnapshot;
export type SaveWorkspaceSettingsInput = SaveWorkspaceSettingsInputDto;

// ── Hand-rolled (not yet ts-rs-generated) ────────────────────────────────
// These remain hand-rolled because the underlying Rust definitions are
// tagged enums whose ts-rs output needs more thought (tag/content layout),
// or because they describe client-side interfaces that have no Rust DTO.

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
