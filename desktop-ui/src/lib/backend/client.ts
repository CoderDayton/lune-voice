import { invoke } from "@tauri-apps/api/core";
import { mockBackendClient } from "./mock";
import type { BackendClient, BackendWorkspaceSnapshot } from "./types";

import { isTauriRuntime } from "./runtime";

const tauriBackendClient: BackendClient = {
  loadWorkspaceSnapshot() {
    return invoke<BackendWorkspaceSnapshot>("load_workspace_snapshot");
  },

  saveWorkspaceSettings(input) {
    return invoke<BackendWorkspaceSnapshot>("save_workspace_settings", { input });
  },

  createDictionaryEntry(phrase, replacement, priority) {
    return invoke<BackendWorkspaceSnapshot>("create_dictionary_entry", {
      input: {
        phrase,
        replacement,
        priority,
      },
    });
  },

  deleteDictionaryEntry(id) {
    return invoke<BackendWorkspaceSnapshot>("delete_dictionary_entry", { entryId: id });
  },

  startDictation() {
    return invoke<string>("start_dictation");
  },

  stopDictation() {
    return invoke<void>("stop_dictation");
  },

  cancelDictation() {
    return invoke<void>("cancel_dictation");
  },

  toggleDictation() {
    return invoke<string | null>("toggle_dictation");
  },
};

// Resolved at module load — safe because Tauri injects __TAURI_INTERNALS__ before scripts execute.
// In dev mode with HMR, the module re-evaluates and picks up the correct runtime.
export const backendClient = isTauriRuntime() ? tauriBackendClient : mockBackendClient;
