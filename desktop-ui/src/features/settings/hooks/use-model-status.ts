// SPDX-License-Identifier: GPL-3.0-or-later

import { invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useRef, useState } from 'react';
import { isTauriRuntime } from '../../../lib/backend/runtime';
import { useModelDownloads } from '../../model-downloads/model-downloads';

export interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
  description: string;
  required: boolean;
  /** Whether the active backend selection currently uses this model. */
  inUse: boolean;
  /** Pipeline role for library grouping: currently always `'speech-to-text'`. */
  role: string;
  /** Actual on-disk size in bytes when installed; `null` otherwise. */
  sizeBytes: number | null;
  /** Bytes a download would fetch, VAD included; `null` when unknown. */
  downloadBytes: number | null;
}

export interface ModelStatusApi {
  models: ModelStatus[];
  /** Lookup by on-disk directory (`local_name`). */
  byName: Map<string, ModelStatus>;
  loading: boolean;
  refresh: () => void;
  remove: (localName: string) => Promise<void>;
}

/**
 * `check_model_status` reports the full installable library (every Whisper
 * size), each flagged with `inUse` for the active config. Call
 * `refresh` when the backend or selected model changes to keep `inUse` current.
 * Downloads are not handled here: they run on the app's shared queue, and a
 * finished one makes this list reload.
 */
export function useModelStatus(): ModelStatusApi {
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const completions = useModelDownloads().completions;

  const refresh = useCallback(() => {
    if (!isTauriRuntime()) {
      setLoading(false);
      return;
    }
    invoke<ModelStatus[]>('check_model_status')
      .then(setModels)
      .catch((e) => console.error('check_model_status:', e))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const seen = useRef(completions);
  useEffect(() => {
    if (completions === seen.current) return;
    seen.current = completions;
    refresh();
  }, [refresh, completions]);

  const remove = useCallback(
    async (localName: string) => {
      try {
        await invoke('delete_model_cmd', { localName });
        refresh();
      } catch (e) {
        console.error('delete_model_cmd:', e);
      }
    },
    [refresh],
  );

  const byName = new Map(models.map((m) => [m.localName, m]));

  return { models, byName, loading, refresh, remove };
}
