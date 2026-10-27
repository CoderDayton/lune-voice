// SPDX-License-Identifier: GPL-3.0-or-later

import { useStore } from '@tanstack/react-store';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { backendClient } from '../../lib/backend/client';
import { isTauriRuntime } from '../../lib/backend/runtime';
import { createDownloadRun, type DownloadEvent, type SavedRun } from './download-run';

/** Where an interrupted run is parked between window lifetimes. */
const QUEUE_KEY = 'lune-setup-queue';

const storage = {
  load(): SavedRun | null {
    try {
      const raw = window.localStorage?.getItem(QUEUE_KEY);
      if (!raw) return null;
      const parsed: unknown = JSON.parse(raw);
      // Hand-checked rather than trusted: this outlives the app version that
      // wrote it, and a shape change would otherwise crash the first screen.
      if (
        typeof parsed !== 'object' ||
        parsed === null ||
        !Array.isArray((parsed as { pending?: unknown }).pending) ||
        !(parsed as { pending: unknown[] }).pending.every((n) => typeof n === 'string')
      ) {
        return null;
      }
      const record = parsed as { pending: string[]; total?: unknown; wantsGpu?: unknown };
      return {
        pending: record.pending,
        total: typeof record.total === 'number' ? record.total : record.pending.length,
        tag: record.wantsGpu === true ? 'gpu' : null,
      };
    } catch {
      return null;
    }
  },
  save(run: SavedRun): void {
    try {
      const { pending, total, tag } = run;
      window.localStorage?.setItem(
        QUEUE_KEY,
        JSON.stringify({ pending, total, wantsGpu: tag === 'gpu' }),
      );
    } catch {
      // Storage can be unavailable or full. Losing the resume record only costs
      // the user the clicks it would have saved, so it is never worth failing on.
    }
  },
  clear(): void {
    try {
      window.localStorage?.removeItem(QUEUE_KEY);
    } catch {
      // Same reasoning as `save`.
    }
  },
};

/**
 * The app's model downloads. A module-level singleton, so a run keeps going —
 * and its progress keeps showing — while the user moves between Settings pages
 * or the wizard's screens.
 */
export const modelDownloads = createDownloadRun({
  download: (localName) => invoke<void>('download_model_cmd', { localName }),
  cancel: () => backendClient.cancelModelDownload(),
  missingBytes: async (localName) => {
    const library =
      await invoke<{ localName: string; downloadBytes: number | null }[]>('check_model_status');
    return library.find((m) => m.localName === localName)?.downloadBytes ?? null;
  },
  storage,
});

if (isTauriRuntime()) {
  const unlisten = listen<DownloadEvent>('lune://download-progress', (event) => {
    modelDownloads.onProgress(event.payload);
  });
  // A hot reload re-runs this module; without this each reload stacks another
  // listener and one report advances the queue several times.
  import.meta.hot?.dispose(() => {
    void unlisten.then((stop) => stop());
  });
}

/** The run's state, re-rendering as it changes. */
export function useModelDownloads() {
  return useStore(modelDownloads.store);
}
