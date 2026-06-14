// DTO shapes are generated from `desktop-ui/src-tauri/src/dto.rs` via ts-rs.
// Run `cargo test -p lune-ui` after editing any Rust DTO to regenerate.
//
// Consumers should import DTO types by their canonical generated names
// (e.g. `WorkspaceSnapshot`, `DictionaryItemDto`). The hand-rolled types
// below describe client-side interfaces and tagged enums that have no Rust
// counterpart yet.

import type { PillSnapshotDto, SaveWorkspaceSettingsInput, WorkspaceSnapshot } from './generated';

export type {
  ActivationModeDto,
  AsrBackendTypeDto,
  DevicePolicyDto,
  DictationModeDto,
  DictionaryItemDto,
  HistoryItemDto,
  PillSnapshotDto,
  PillVisibilityDto,
  SaveWorkspaceSettingsInput,
  SessionStatsDto,
  StylePresetDto,
  WorkspaceSettingsDto,
  WorkspaceSnapshot,
} from './generated';

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
  loadWorkspaceSnapshot(): Promise<WorkspaceSnapshot>;
  loadPillSnapshot(): Promise<PillSnapshotDto>;
  saveWorkspaceSettings(input: SaveWorkspaceSettingsInput): Promise<WorkspaceSnapshot>;
  createDictionaryEntry(
    phrase: string,
    replacement: string,
    priority: number,
  ): Promise<WorkspaceSnapshot>;
  updateDictionaryEntry(
    id: number,
    phrase: string,
    replacement: string,
    priority: number,
  ): Promise<WorkspaceSnapshot>;
  deleteDictionaryEntry(id: number): Promise<WorkspaceSnapshot>;
  deleteSession(sessionId: string): Promise<WorkspaceSnapshot>;
  createCustomStyle(
    name: string,
    description: string,
    instructions: string,
  ): Promise<WorkspaceSnapshot>;
  updateCustomStyle(
    id: string,
    name: string,
    description: string,
    instructions: string,
  ): Promise<WorkspaceSnapshot>;
  deleteCustomStyle(styleId: string): Promise<WorkspaceSnapshot>;
  generateStyleExamples(styleId: string): Promise<WorkspaceSnapshot>;
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
  | { type: 'sessionError'; message: string }
  | { type: 'sessionWarning'; message: string };
