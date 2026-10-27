// SPDX-License-Identifier: GPL-3.0-or-later

import { invoke } from '@tauri-apps/api/core';
import { mockBackendClient } from './mock';
import { isTauriRuntime } from './runtime';
import type {
  BackendClient,
  GpuBackendStatusDto,
  HistoryPageDto,
  MonitorInfoDto,
  PillSnapshotDto,
  UpdateStateDto,
  WorkspaceSnapshot,
} from './types';

const tauriBackendClient: BackendClient = {
  appVersion() {
    return invoke<string>('app_version');
  },

  startUpdateCheck() {
    return invoke<boolean>('start_update_check');
  },

  updateState() {
    return invoke<UpdateStateDto>('update_state');
  },

  uinputAccessReady() {
    return invoke<boolean>('uinput_access_ready');
  },

  grantUinputAccess() {
    return invoke<boolean>('grant_uinput_access');
  },

  holdToTalkAvailable() {
    return invoke<boolean>('hold_to_talk_available');
  },

  listAudioInputDevices() {
    return invoke<string[]>('list_audio_input_devices');
  },

  activeModelDownload() {
    return invoke<string | null>('active_model_download');
  },

  cancelModelDownload() {
    return invoke<void>('cancel_model_download');
  },

  retryRuntime() {
    return invoke<void>('retry_runtime');
  },

  installPendingUpdate() {
    return invoke<void>('install_pending_update');
  },

  dismissUpdate() {
    return invoke<void>('dismiss_update');
  },

  quitApp() {
    return invoke<void>('quit_app');
  },

  feedbackAvailable() {
    return invoke<boolean>('feedback_available');
  },

  feedbackCooldownRemaining() {
    return invoke<number>('feedback_cooldown_remaining');
  },

  submitFeedback(category, message) {
    return invoke<void>('submit_feedback', { category, message });
  },

  loadWorkspaceSnapshot() {
    return invoke<WorkspaceSnapshot>('load_workspace_snapshot');
  },

  loadPillSnapshot() {
    return invoke<PillSnapshotDto>('load_pill_snapshot');
  },

  saveWorkspaceSettings(input) {
    return invoke<WorkspaceSnapshot>('save_workspace_settings', { input });
  },

  createDictionaryEntry(phrase, replacement, priority) {
    return invoke<WorkspaceSnapshot>('create_dictionary_entry', {
      input: {
        phrase,
        replacement,
        priority,
      },
    });
  },

  updateDictionaryEntry(id, phrase, replacement, priority) {
    return invoke<WorkspaceSnapshot>('update_dictionary_entry', {
      input: { id, phrase, replacement, priority },
    });
  },

  deleteDictionaryEntry(id) {
    return invoke<WorkspaceSnapshot>('delete_dictionary_entry', { entryId: id });
  },

  saveSnippet(id, triggerPhrase, body) {
    return invoke<WorkspaceSnapshot>('save_snippet', {
      input: { id, triggerPhrase, body },
    });
  },

  deleteSnippet(id) {
    return invoke<WorkspaceSnapshot>('delete_snippet', { snippetId: id });
  },

  deleteSession(sessionId) {
    return invoke<WorkspaceSnapshot>('delete_session', { sessionId });
  },

  deleteSessions(sessionIds) {
    return invoke<WorkspaceSnapshot>('delete_sessions', { sessionIds });
  },

  listHistoryPage(page, pageSize, search) {
    return invoke<HistoryPageDto>('list_history_page', { page, pageSize, search });
  },

  createCustomStyle(name, description, instructions, formatting) {
    return invoke<WorkspaceSnapshot>('create_custom_style', {
      input: { name, description, instructions, formatting },
    });
  },
  updateCustomStyle(id, name, description, instructions, formatting) {
    return invoke<WorkspaceSnapshot>('update_custom_style', {
      input: { id, name, description, instructions, formatting },
    });
  },
  deleteCustomStyle(styleId) {
    return invoke<WorkspaceSnapshot>('delete_custom_style', { styleId });
  },
  generateStyleExamples(styleId) {
    return invoke<WorkspaceSnapshot>('generate_style_examples', { styleId });
  },
  startDictation() {
    return invoke<string>('start_dictation');
  },

  startTriggerCapture() {
    return invoke<string>('start_trigger_capture');
  },

  stopDictation() {
    return invoke<void>('stop_dictation');
  },

  cancelDictation() {
    return invoke<void>('cancel_dictation');
  },

  toggleDictation() {
    return invoke<string | null>('toggle_dictation');
  },

  nudgePill(dx, dy) {
    return invoke<void>('nudge_pill', { dx, dy });
  },

  savePillPosition() {
    return invoke<void>('save_pill_position');
  },

  resetPillPosition() {
    return invoke<void>('reset_pill_position');
  },

  listMonitors() {
    return invoke<MonitorInfoDto[]>('list_monitors');
  },

  gpuBackendStatus() {
    return invoke<GpuBackendStatusDto>('gpu_backend_status');
  },

  uinputRuleInstalled() {
    return invoke<boolean>('uinput_rule_installed');
  },

  removeUinputRule() {
    return invoke<boolean>('remove_uinput_rule');
  },
};

// Resolved at module load — safe because Tauri injects __TAURI_INTERNALS__ before scripts execute.
// In dev mode with HMR, the module re-evaluates and picks up the correct runtime.
export const backendClient = isTauriRuntime() ? tauriBackendClient : mockBackendClient;
