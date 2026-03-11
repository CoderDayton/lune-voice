import { invoke } from "@tauri-apps/api/core";
import { mockBackendClient } from "./mock";
import type { BackendClient, BackendWorkspaceSnapshot } from "./types";

function isTauriRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

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
};

export const backendClient = isTauriRuntime() ? tauriBackendClient : mockBackendClient;
