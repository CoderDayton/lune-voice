// DTO shapes are generated from `desktop-ui/src-tauri/src/dto.rs` via ts-rs.
// Run `cargo test -p lune-ui` after editing any Rust DTO to regenerate.
//
// Consumers should import DTO types by their canonical generated names
// (e.g. `WorkspaceSnapshot`, `DictionaryItemDto`). The hand-rolled types
// below describe client-side interfaces and tagged enums that have no Rust
// counterpart yet.

import type {
  FormatTogglesDto,
  GpuBackendStatusDto,
  HistoryPageDto,
  MonitorInfoDto,
  PillSnapshotDto,
  SaveWorkspaceSettingsInput,
  WorkspaceSnapshot,
} from './generated';

export type {
  ActivationModeDto,
  AsrBackendTypeDto,
  DevicePolicyDto,
  DictationModeDto,
  DictionaryItemDto,
  FormatTogglesDto,
  GpuBackendStateDto,
  GpuBackendStatusDto,
  GpuInstallProgressDto,
  HistoryItemDto,
  HistoryPageDto,
  MonitorInfoDto,
  PillAnchorDto,
  PillCornerDto,
  PillOrientationDto,
  PillSizeDto,
  PillSnapshotDto,
  PillVisibilityDto,
  SaveWorkspaceSettingsInput,
  SessionStatsDto,
  SnippetItemDto,
  StylePresetDto,
  TerminalModeDto,
  WorkspaceSettingsDto,
  WorkspaceSnapshot,
} from './generated';

export interface BackendClient {
  /** Version of the running build, as the updater compares it. */
  appVersion(): Promise<string>;
  /** Fire the update check. Resolves immediately; the outcome is a native dialog. */
  startUpdateCheck(): Promise<void>;
  /** Quit the app for real, bypassing the close-to-tray behaviour. */
  quitApp(): Promise<void>;
  /** Whether this build was compiled with a feedback proxy endpoint. */
  feedbackAvailable(): Promise<boolean>;
  /** Seconds left before another report is accepted; 0 when ready. */
  feedbackCooldownRemaining(): Promise<number>;
  /** Send one report. Rejects with a user-facing message on failure. */
  submitFeedback(category: string, message: string): Promise<void>;
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
  /** Create (`id: null`) or update (`id` set) a voice snippet. */
  saveSnippet(id: string | null, triggerPhrase: string, body: string): Promise<WorkspaceSnapshot>;
  deleteSnippet(id: string): Promise<WorkspaceSnapshot>;
  deleteSession(sessionId: string): Promise<WorkspaceSnapshot>;
  listHistoryPage(page: number, pageSize: number, search: string): Promise<HistoryPageDto>;
  createCustomStyle(
    name: string,
    description: string,
    instructions: string,
    formatting: FormatTogglesDto,
  ): Promise<WorkspaceSnapshot>;
  updateCustomStyle(
    id: string,
    name: string,
    description: string,
    instructions: string,
    formatting: FormatTogglesDto,
  ): Promise<WorkspaceSnapshot>;
  deleteCustomStyle(styleId: string): Promise<WorkspaceSnapshot>;
  generateStyleExamples(styleId: string): Promise<WorkspaceSnapshot>;
  startDictation(): Promise<string>;
  /** Start a capture-only session: speech is transcribed and streamed back as
   *  a `captureText` event; nothing is typed into the focused app. */
  startTriggerCapture(): Promise<string>;
  stopDictation(): Promise<void>;
  cancelDictation(): Promise<void>;
  toggleDictation(): Promise<string | null>;
  /** Move the pill overlay by a drag delta (logical px). */
  nudgePill(dx: number, dy: number): Promise<void>;
  /** Persist the pill overlay's current position so it reopens there. */
  savePillPosition(): Promise<void>;
  /** Clear the saved pill position and re-place it at its default anchor. */
  resetPillPosition(): Promise<void>;
  /** Connected monitors, for the pill's monitor picker. */
  listMonitors(): Promise<MonitorInfoDto[]>;
  /** Runtime GPU state: driver present, plugin installed, device registered. */
  gpuBackendStatus(): Promise<GpuBackendStatusDto>;
  /**
   * Whether this app installed the `/dev/uinput` udev rule itself. False on a
   * .deb/.rpm install, where the package manager owns the file.
   */
  uinputRuleInstalled(): Promise<boolean>;
  /** Remove that rule. Raises a system password prompt. */
  removeUinputRule(): Promise<boolean>;
}

export type SessionStreamEvent =
  | { type: 'serviceStatus'; status: string }
  | { type: 'sessionStarted'; sessionId: string }
  | { type: 'partialText' }
  | { type: 'processing' }
  | { type: 'finalizedText'; text: string }
  | { type: 'correctedText'; text: string }
  | { type: 'captureText'; text: string }
  | { type: 'audioLevel'; level: number }
  | { type: 'commandDetected'; payload: string }
  | { type: 'sessionStopped' }
  | { type: 'sessionError'; message: string }
  | { type: 'sessionWarning'; message: string };
