// SPDX-License-Identifier: GPL-3.0-or-later

export function isTauriRuntime() {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}
