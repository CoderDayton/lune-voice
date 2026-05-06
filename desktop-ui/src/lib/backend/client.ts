import { Channel, invoke } from '@tauri-apps/api/core';
import { mockBackendClient } from './mock';
import { isTauriRuntime } from './runtime';
import type {
  BackendClient,
  BackendPillSnapshot,
  BackendWorkspaceSnapshot,
  CalibrationHandle,
  CalibrationStreamEvent,
} from './types';

const tauriBackendClient: BackendClient = {
  loadWorkspaceSnapshot() {
    return invoke<BackendWorkspaceSnapshot>('load_workspace_snapshot');
  },

  loadPillSnapshot() {
    return invoke<BackendPillSnapshot>('load_pill_snapshot');
  },

  saveWorkspaceSettings(input) {
    return invoke<BackendWorkspaceSnapshot>('save_workspace_settings', { input });
  },

  createDictionaryEntry(phrase, replacement, priority) {
    return invoke<BackendWorkspaceSnapshot>('create_dictionary_entry', {
      input: {
        phrase,
        replacement,
        priority,
      },
    });
  },

  updateDictionaryEntry(id, phrase, replacement, priority) {
    return invoke<BackendWorkspaceSnapshot>('update_dictionary_entry', {
      input: { id, phrase, replacement, priority },
    });
  },

  deleteDictionaryEntry(id) {
    return invoke<BackendWorkspaceSnapshot>('delete_dictionary_entry', { entryId: id });
  },

  deleteSession(sessionId) {
    return invoke<BackendWorkspaceSnapshot>('delete_session', { sessionId });
  },

  createCustomStyle(name, description, instructions) {
    return invoke<BackendWorkspaceSnapshot>('create_custom_style', {
      input: { name, description, instructions },
    });
  },
  updateCustomStyle(id, name, description, instructions) {
    return invoke<BackendWorkspaceSnapshot>('update_custom_style', {
      input: { id, name, description, instructions },
    });
  },
  deleteCustomStyle(styleId) {
    return invoke<BackendWorkspaceSnapshot>('delete_custom_style', { styleId });
  },
  generateStyleExamples(styleId) {
    return invoke<BackendWorkspaceSnapshot>('generate_style_examples', { styleId });
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

  async calibrateStart(onEvent): Promise<CalibrationHandle> {
    const channel = new Channel<CalibrationStreamEvent>((event) => {
      try {
        onEvent(event);
      } catch (e) {
        console.error('[calibrate] event handler threw', e);
      }
    });
    await invoke<void>('calibrate_start', { channel });
    return {
      advance: () => invoke<void>('calibrate_advance'),
      cancel: () => invoke<void>('calibrate_cancel'),
    };
  },
};

// Resolved at module load — safe because Tauri injects __TAURI_INTERNALS__ before scripts execute.
// In dev mode with HMR, the module re-evaluates and picks up the correct runtime.
export const backendClient = isTauriRuntime() ? tauriBackendClient : mockBackendClient;
