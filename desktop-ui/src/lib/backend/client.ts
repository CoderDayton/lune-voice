import { invoke } from '@tauri-apps/api/core';
import { mockBackendClient } from './mock';
import { isTauriRuntime } from './runtime';
import type {
  BackendClient,
  GpuSupportDto,
  HistoryPageDto,
  PillSnapshotDto,
  WorkspaceSnapshot,
} from './types';

const tauriBackendClient: BackendClient = {
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

  deleteSession(sessionId) {
    return invoke<WorkspaceSnapshot>('delete_session', { sessionId });
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

  gpuSupport() {
    return invoke<GpuSupportDto>('gpu_support');
  },
};

// Resolved at module load — safe because Tauri injects __TAURI_INTERNALS__ before scripts execute.
// In dev mode with HMR, the module re-evaluates and picks up the correct runtime.
export const backendClient = isTauriRuntime() ? tauriBackendClient : mockBackendClient;
